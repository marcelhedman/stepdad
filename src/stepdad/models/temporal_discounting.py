"""
Hyperbolic temporal discounting model.

Participants choose between a small immediate reward R and a delayed reward
of £100 after D days.  The latent parameters theta = (log_k, alpha) capture
the individual's discount rate and noise level.

Design and unconstrained parameterisation
-----------------------------------------
The design xi = (xi_r, xi_d) lives in unconstrained R^2 and is mapped to
the constrained (R, D) space via:
    R = 100 * sigmoid(xi_r)        so R in (0, 100)
    D = exp(xi_d - log_k_mean)     so D > 0

The log_k_mean offset in D helps with initialisation.

Observation model
-----------------
    V0 = R                          (immediate reward)
    V1 = 100 / (1 + exp(log_k) * D)  (delayed reward, discounted)
    psi = epsilon + (1 - 2*epsilon) * Phi((V0 - V1) / |alpha|)
    y | theta, xi  ~  Bernoulli(psi)      y = 1 if participant chooses delayed

Prior
-----
    log_k  ~  Normal(-4.25, 1.5)
    alpha  ~  Normal(0, 2) 
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.distributions as dist
from torch import Tensor


class TemporalDiscountingPrior(nn.Module):
    """Prior over (log_k, alpha).

    Args:
        log_k_loc: mean of log_k prior (default -4.25).
        log_k_scale: std of log_k prior (default 1.5).
        alpha_loc: mean of alpha prior (default 0.0).
        alpha_scale: std of alpha prior (default 2.0).
        device: torch device.
    """

    def __init__(
        self,
        log_k_loc: float = -4.25,
        log_k_scale: float = 1.5,
        alpha_loc: float = 0.0,
        alpha_scale: float = 2.0,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        self.log_k_loc = log_k_loc
        self.log_k_scale = log_k_scale
        self.device = torch.device(device)
        self.theta_dim = 2

        self._log_k_dist = dist.Normal(
            torch.tensor([log_k_loc], device=self.device),
            torch.tensor([log_k_scale], device=self.device),
        )
        self._alpha_dist = dist.Normal(
            torch.tensor([alpha_loc], device=self.device),
            torch.tensor([alpha_scale], device=self.device),
        )

    def sample(self, n: int) -> Tensor:
        """Return n samples of shape [n, 2] = [log_k, alpha]."""
        log_k = self._log_k_dist.sample((n,)).squeeze(-1)  # [n]
        alpha = self._alpha_dist.sample((n,)).squeeze(-1)   # [n]
        return torch.stack([log_k, alpha], dim=-1)           # [n, 2]

    def log_prob(self, theta: Tensor) -> Tensor:
        """Log prior density.

        Args:
            theta: [n, 2]  — (log_k, alpha)
        Returns:
            [n]
        """
        log_k = theta[..., 0:1]
        alpha = theta[..., 1:2]
        return (
            self._log_k_dist.log_prob(log_k) + self._alpha_dist.log_prob(alpha)
        ).squeeze(-1)


class TemporalDiscountingModel(nn.Module):
    """Hyperbolic temporal discounting generative model.

    Args:
        prior: TemporalDiscountingPrior.
        design_net: policy network; called as design_net(designs, outcomes).
        T: total experiment steps.
        long_reward: fixed delayed reward amount (default 100).
        short_delay: delay for the immediate option (default 0).
        epsilon: lapse rate / floor/ceiling on choice probability (default 0.01).
        device: torch device.
    """

    def __init__(
        self,
        prior: TemporalDiscountingPrior,
        design_net: nn.Module,
        T: int = 20,
        long_reward: float = 100.0,
        short_delay: float = 0.0,
        epsilon: float = 0.01,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        self.prior = prior
        self.design_net = design_net
        self.T = T
        self.long_reward = long_reward
        self.short_delay = short_delay
        self.epsilon = epsilon
        self.device = torch.device(device)
        # log_k_mean used as shift in the delay transform (see module docstring)
        self.log_k_mean: float = prior.log_k_loc + 0.5 * prior.log_k_scale ** 2
        self.design_dim = 2

    # ------------------------------------------------------------------
    # Design transform
    # ------------------------------------------------------------------

    def _transform_xi(self, xi: Tensor) -> tuple[Tensor, Tensor]:
        """Map unconstrained xi -> constrained (R, D).

        Args:
            xi: [..., 2]  — (xi_r, xi_d) unconstrained
        Returns:
            R: [..., 1]  in (0, 100)
            D: [..., 1]  > 0
        """
        xi_r, xi_d = xi[..., 0:1], xi[..., 1:2]
        R = self.long_reward * torch.sigmoid(xi_r)
        D = torch.exp(xi_d - self.log_k_mean)
        return R, D

    # ------------------------------------------------------------------
    # Likelihood
    # ------------------------------------------------------------------

    def transform_design(self, xi: Tensor) -> Tensor:
        return xi

    def outcome_likelihood(self, theta: Tensor, xi: Tensor) -> dist.Distribution:
        """Return p(y | theta, xi) = Bernoulli(psi).

        Args:
            theta: [..., 2]  — (log_k, alpha)
            xi:    [..., 2]  — unconstrained design
        Returns:
            Bernoulli distribution with prob psi.
        """
        log_k = theta[..., 0:1]
        alpha = theta[..., 1:2]
        R, D = self._transform_xi(xi)

        v0 = R / (1.0 + torch.exp(log_k) * self.short_delay)
        v1 = self.long_reward / (1.0 + torch.exp(log_k) * D)
        erf_arg = (v0 - v1) / (alpha.abs() + 1e-3)
        psi = self.epsilon + (1.0 - 2.0 * self.epsilon) * (
            0.5 + 0.5 * torch.erf(erf_arg / math.sqrt(2.0))
        )
        return dist.Bernoulli(probs=psi)

    def log_likelihood(self, theta: Tensor, designs: Tensor, outcomes: Tensor) -> Tensor:
        """Evaluate sum_t log p(y_t | theta_n, xi_t^n).

        Args:
            theta:    [N, 2]
            designs:  [N, T, 2]
            outcomes: [N, T, 1]
        Returns:
            [N]
        """
        N, T, _ = designs.shape
        log_lk = torch.zeros(N, device=theta.device)
        for t in range(T):
            log_lk += self.outcome_likelihood(
                theta, designs[:, t, :]
            ).log_prob(outcomes[:, t, :]).squeeze(-1)
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

        Returns:
            theta:    [B, 2]
            designs:  [B, T_new, 2]
            outcomes: [B, T_new, 1]
        """
        theta = self.prior.sample(batch_size).to(self.device)

        designs = (
            torch.empty(batch_size, 0, 2, device=self.device)
            if past_designs is None else past_designs
        )
        outcomes = (
            torch.empty(batch_size, 0, 1, device=self.device)
            if past_outcomes is None else past_outcomes
        )
        n_past = designs.shape[1]

        for _ in range(n_past, self.T):
            xi = self.design_net(designs, outcomes)
            yi = self.outcome_likelihood(theta, xi).sample()
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
            designs:  [B, T, 2]
            outcomes: [B, T, 1]
        """
        B = theta.shape[0]
        designs = (
            torch.empty(B, 0, 2, device=self.device)
            if past_designs is None else past_designs
        )
        outcomes = (
            torch.empty(B, 0, 1, device=self.device)
            if past_outcomes is None else past_outcomes
        )
        n_past = designs.shape[1]

        for _ in range(n_past, self.T + n_past):
            xi = self.design_net(designs, outcomes)
            yi = self.outcome_likelihood(theta, xi).sample()
            designs = torch.cat([designs, xi.unsqueeze(1)], dim=1)
            outcomes = torch.cat([outcomes, yi.unsqueeze(1)], dim=1)

        return designs, outcomes
