"""
Constant Elasticity of Substitution (CES) model.

Participants compare two baskets of three goods and report their relative preference
on a continuous scale y ∈ [0, 1]. 

Theta representation
--------------------
theta has shape [..., 5]:
    theta[..., 0:3]  = alpha  in simplex   via Dirichlet(1,1,1)
    theta[..., 3]    = rho    in (0.01,1)  via 0.01 + 0.99 * Beta(1,1)
    theta[..., 4]    = slope  > 0          via LogNormal(1, 3)

Design
------
xi has shape [..., 6] in unconstrained R^6.  Before evaluation the policy output
is mapped to constrained basket quantities via:
    transform_to(constraints.interval(1e-6, 100.))(xi)
which gives each good a value in approximately (0, 100).

Observation model
-----------------
    U(x) = exp(-rho * log(sum_i  x_i^rho * alpha_i))    [= Z^{-rho}]
    mean  = slope * (U(x) - U(x'))
    sd    = slope * tau * (1 + ||x - x'||_2) + 1e-6      (tau = 0.005)
    eta   ~ Normal(mean, sd^2)
    y     = clip(sigmoid(eta), eps, 1 - eps)              (eps = 2^{-22})


Prior
-----
    alpha  ~  Dirichlet(1, 1, 1)
    rho    =  0.01 + 0.99 * rho_raw,   rho_raw ~ Beta(1, 1)
    slope  ~  LogNormal(1, 3)
"""

from __future__ import annotations

import numbers
import torch
import torch.nn as nn
import torch.distributions as dist
from torch import Tensor
from torch.distributions.transforms import SigmoidTransform


# ---------------------------------------------------------------------------
# CensoredSigmoidNormal
# ---------------------------------------------------------------------------

class CensoredSigmoidNormal(dist.Distribution):
    """Normal passed through sigmoid and censored at [lower_lim, upper_lim].

    y is drawn as:
        eta ~ Normal(loc, scale)
        y   = clip(sigmoid(eta), lower_lim, upper_lim)

    log_prob places point mass at the boundaries (CDF tails) and uses the
    change-of-variables pdf in the interior.

    Args:
        loc:       [...] location of the underlying Normal.
        scale:     [...] scale of the underlying Normal (> 0).
        upper_lim: upper censoring boundary (default 1 - 2^{-22}).
        lower_lim: lower censoring boundary (default 2^{-22}).
    """

    def __init__(
        self,
        loc: Tensor,
        scale: Tensor,
        upper_lim: float,
        lower_lim: float,
        validate_args: bool = False,
    ) -> None:
        self.upper_lim = upper_lim
        self.lower_lim = lower_lim
        self.transform = SigmoidTransform()
        normal = dist.Normal(loc, scale, validate_args=validate_args)
        self.base_dist = dist.TransformedDistribution(normal, [self.transform])
        super().__init__(
            self.base_dist.batch_shape,
            self.base_dist.event_shape,
            validate_args=validate_args,
        )

    def sample(self, sample_shape: torch.Size = torch.Size()) -> Tensor:
        with torch.no_grad():
            x = self.base_dist.sample(sample_shape)
            x = x.clamp(self.lower_lim, self.upper_lim)
            return x

    def log_prob(self, value: Tensor) -> Tensor:
        """Log probability under the censored measure.

        For interior points: standard change-of-variables pdf.
        At lower_lim: log P(eta <= logit(lower_lim)).
        At upper_lim: log P(eta >= logit(upper_lim)).
        Outside bounds: -inf.

        Uses an asymptotic approximation for tail probabilities that are
        numerically indistinguishable from zero (< 1e-40).
        """
        log_p = self.base_dist.log_prob(value)

        # Handle shape mismatch when log_p has extra leading dim (L+1, B, ...)
        if log_p.dim() > value.dim():
            value = value.unsqueeze(0).expand_as(log_p)

        crit = 1e-40

        upper_cdf = 1.0 - self.base_dist.cdf(
            torch.tensor(self.upper_lim, device=log_p.device, dtype=log_p.dtype)
        )
        lower_cdf = self.base_dist.cdf(
            torch.tensor(self.lower_lim, device=log_p.device, dtype=log_p.dtype)
        )

        # Asymptotic log-prob for very small CDF values
        shape = self.base_dist.batch_shape
        z_upper = self._z(torch.tensor(self.upper_lim, device=log_p.device, dtype=log_p.dtype))
        z_lower = self._z(torch.tensor(self.lower_lim, device=log_p.device, dtype=log_p.dtype))
        asym_upper = self.base_dist.log_prob(
            torch.tensor(self.upper_lim, device=log_p.device, dtype=log_p.dtype).expand(shape)
        ) - (crit + z_upper.abs()).log()
        asym_lower = self.base_dist.log_prob(
            torch.tensor(self.lower_lim, device=log_p.device, dtype=log_p.dtype).expand(shape)
        ) - (crit + z_lower.abs()).log()

        mask_upper = upper_cdf < crit
        mask_lower = lower_cdf < crit

        upper_cdf = upper_cdf.masked_fill(mask_upper, 1.0).log()
        upper_cdf = upper_cdf.masked_scatter(mask_upper, asym_upper[mask_upper])

        lower_cdf = lower_cdf.masked_fill(mask_lower, 1.0).log()
        lower_cdf = lower_cdf.masked_scatter(mask_lower, asym_lower[mask_lower])

        log_p = log_p.clone()
        log_p[value == self.upper_lim] = upper_cdf.expand_as(log_p)[value == self.upper_lim]
        log_p[value > self.upper_lim] = float("-inf")
        log_p[value == self.lower_lim] = lower_cdf.expand_as(log_p)[value == self.lower_lim]
        log_p[value < self.lower_lim] = float("-inf")

        return log_p

    def _z(self, value: Tensor) -> Tensor:
        """Standardised value in Normal space: (logit(value) - loc) / scale."""
        return (self.transform.inv(value) - self.base_dist.base_dist.loc) / self.base_dist.base_dist.scale


# ---------------------------------------------------------------------------
# Prior
# ---------------------------------------------------------------------------

class CESPrior(nn.Module):
    """Prior over (alpha, rho, slope).

    Samples have shape [n, 5]:
        [:, 0:3] = alpha from Dirichlet(1,1,1)
        [:, 3]   = rho   = 0.01 + 0.99 * rho_raw,  rho_raw ~ Beta(1,1)
        [:, 4]   = slope ~ LogNormal(1, 3)

    Args:
        device: torch device.
    """

    def __init__(self, device: torch.device | str = "cpu") -> None:
        super().__init__()
        self.device = torch.device(device)
        self.theta_dim = 5

        self._alpha_dist = dist.Dirichlet(torch.ones(3, device=self.device))
        self._rho_dist = dist.Beta(
            torch.tensor(1.0, device=self.device),
            torch.tensor(1.0, device=self.device),
        )
        self._slope_dist = dist.LogNormal(
            torch.tensor(1.0, device=self.device),
            torch.tensor(3.0, device=self.device),
        )

    def sample(self, n: int) -> Tensor:
        """Return n samples of shape [n, 5]."""
        alpha = self._alpha_dist.sample((n,))      # [n, 3]
        rho_raw = self._rho_dist.sample((n,))      # [n]
        rho = 0.01 + 0.99 * rho_raw               # rho in (0.01, 1)
        slope = self._slope_dist.sample((n,))      # [n]

        out = torch.empty(n, 5, device=self.device)
        out[:, 0:3] = alpha
        out[:, 3] = rho
        out[:, 4] = slope
        return out

    def log_prob(self, theta: Tensor) -> Tensor:
        """Log prior density.

        Note: log p(rho) is evaluated by passing the constrained rho value
        directly to Beta.log_prob (Beta is defined on [0,1] and rho ∈ (0.01,1)).
        This matches the original implementation.

        Args:
            theta: [n, 5]
        Returns:
            [n]
        """
        alpha = theta[..., 0:3]
        rho = theta[..., 3]
        slope = theta[..., 4]

        lp_alpha = self._alpha_dist.log_prob(alpha.clamp(min=1e-6))
        lp_rho = self._rho_dist.log_prob(rho)
        lp_slope = self._slope_dist.log_prob(slope)

        return lp_alpha + lp_rho + lp_slope


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

_DESIGN_CONSTRAINT = dist.constraints.interval(1e-6, 100.0)


class CESModel(nn.Module):
    """CES generative model.

    Args:
        prior: CESPrior.
        design_net: policy network; called as design_net(designs, outcomes).
        T: total experiment steps.
        tau: noise scale multiplier.
        eps: censoring level (default 2^{-22}).
        device: torch device.
    """

    def __init__(
        self,
        prior: CESPrior,
        design_net: nn.Module,
        T: int = 10,
        tau: float = 0.005,
        eps: float = 2 ** -22,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        self.prior = prior
        self.design_net = design_net
        self.T = T
        self.tau = tau
        self.eps = eps
        self.device = torch.device(device)
        self.design_dim = 6

    # ------------------------------------------------------------------
    # Design transform
    # ------------------------------------------------------------------

    def _transform_xi(self, xi: Tensor) -> Tensor:
        """Map unconstrained xi [..., 6] to constrained basket values in (1e-6, 100).

        Uses the canonical PyTorch interval transform: y = a + (b - a)*sigmoid(x).
        """
        return dist.transform_to(_DESIGN_CONSTRAINT)(xi)

    # ------------------------------------------------------------------
    # Likelihood
    # ------------------------------------------------------------------

    def outcome_likelihood(self, theta: Tensor, xi: Tensor) -> CensoredSigmoidNormal:
        """Return p(y | theta, xi).

        The design xi is transformed to constrained basket space before evaluation.

        Args:
            theta: [..., 5]  — (alpha[3], rho, slope)
            xi:    [..., 6]  — unconstrained design (policy output)
        Returns:
            CensoredSigmoidNormal distribution.
        """
        alpha = theta[..., 0:3]
        rho = theta[..., 3:4]     # [..., 1]
        slope = theta[..., 4:5]   # [..., 1]

        d = self._transform_xi(xi)   # [..., 6] in (1e-6, 100)
        d1, d2 = d[..., 0:3], d[..., 3:6]

        # Utility: Z^{-rho} where Z = sum_i d_i^rho * alpha_i
        # Using log-space for numerical stability: exp(-rho * log(Z))
        U1 = torch.exp(-rho * torch.log((d1.pow(rho) * alpha).sum(-1, keepdim=True)))
        U2 = torch.exp(-rho * torch.log((d2.pow(rho) * alpha).sum(-1, keepdim=True)))

        mean = slope * (U1 - U2)
        sd = slope * self.tau * (1.0 + torch.norm(d1 - d2, dim=-1, p=2, keepdim=True)) + 1e-6

        return CensoredSigmoidNormal(
            mean.squeeze(-1),
            sd.squeeze(-1),
            upper_lim=1.0 - self.eps,
            lower_lim=self.eps,
        )

    def log_likelihood(self, theta: Tensor, designs: Tensor, outcomes: Tensor) -> Tensor:
        """Evaluate sum_t log p(y_t | theta_n, xi_t^n).

        Designs are assumed to be in *constrained* space (already transformed).
        This method calls outcome_likelihood with unconstrained xi, so we apply
        the inverse transform before passing through the model.

        Args:
            theta:    [N, 5]
            designs:  [N, T, 6]   — already in constrained (1e-6, 100) space
            outcomes: [N, T, 1]
        Returns:
            [N]
        """
        N, T, _ = designs.shape
        # Invert the design transform to get unconstrained xi
        xi_unconstrained = dist.transform_to(_DESIGN_CONSTRAINT).inv(designs)
        log_lk = torch.zeros(N, device=theta.device)
        for t in range(T):
            log_lk += self.outcome_likelihood(
                theta, xi_unconstrained[:, t, :]
            ).log_prob(outcomes[:, t, :].squeeze(-1))
        return log_lk

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def forward(
        self,
        batch_size: int,
        past_designs: Tensor | None = None,
        past_outcomes: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Simulate a batch of T-step experiments.

        Designs stored in history are in *constrained* space.

        Returns:
            theta:    [B, 5]
            designs:  [B, T_new, 6]   — constrained
            outcomes: [B, T_new, 1]
        """
        theta = self.prior.sample(batch_size).to(self.device)

        designs = (
            torch.empty(batch_size, 0, 6, device=self.device)
            if past_designs is None else past_designs
        )
        outcomes = (
            torch.empty(batch_size, 0, 1, device=self.device)
            if past_outcomes is None else past_outcomes
        )
        n_past = designs.shape[1]

        for _ in range(n_past, self.T):
            xi_raw = self.design_net(designs, outcomes)       # [B, 6] unconstrained
            xi = self._transform_xi(xi_raw)                   # [B, 6] constrained

            # outcome_likelihood expects unconstrained xi
            yi = self.outcome_likelihood(theta, xi_raw).sample().unsqueeze(-1)

            designs = torch.cat([designs, xi.unsqueeze(1)], dim=1)
            outcomes = torch.cat([outcomes, yi.unsqueeze(1)], dim=1)

        return theta, designs[:, n_past:], outcomes[:, n_past:]

    @torch.no_grad()
    def run_policy(
        self,
        theta: Tensor,
        past_designs: Tensor | None = None,
        past_outcomes: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Roll out the policy for a fixed theta.

        Returns:
            designs:  [B, T, 6]   — constrained
            outcomes: [B, T, 1]
        """
        B = theta.shape[0]
        designs = (
            torch.empty(B, 0, 6, device=self.device)
            if past_designs is None else past_designs
        )
        outcomes = (
            torch.empty(B, 0, 1, device=self.device)
            if past_outcomes is None else past_outcomes
        )
        n_past = designs.shape[1]

        for _ in range(n_past, self.T + n_past):
            xi_raw = self.design_net(designs, outcomes)
            xi = self._transform_xi(xi_raw)
            yi = self.outcome_likelihood(theta, xi_raw).sample().unsqueeze(-1)
            designs = torch.cat([designs, xi.unsqueeze(1)], dim=1)
            outcomes = torch.cat([outcomes, yi.unsqueeze(1)], dim=1)

        return designs, outcomes
