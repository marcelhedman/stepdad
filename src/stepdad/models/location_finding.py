"""
Source location finding model

Objective: infer the 2-D locations of K hidden sources from noisy signal-strength
measurements.  Each source emits a signal that attenuates with the inverse-square law.

Theta representation
--------------------
theta has shape [..., K*p].  The K source locations are stored contiguously:
    theta[..., 0:p]       = location of source 0
    theta[..., p:2p]      = location of source 1
    ...
    theta[..., (K-1)*p:]  = location of source K-1

Use theta.view(..., K, p) to recover the per-source structure.

Observation model
-----------------
    mu(theta, xi) = b + sum_{k=1}^K  alpha_k / (m + ||theta_k - xi||^2)
    log y | theta, xi  ~  Normal( log mu(theta, xi),  sigma^2 )

    m > 0 is a near-field constant that prevents division by zero when a design
    coincides exactly with a source location (Table 8 of the paper: m = 1e-4).

Prior
-----
    theta_k  i.i.d.  ~  Normal(0, I_d)   for k = 1, ..., K
    => theta  ~  Normal(0, I_{K*p})

log_likelihood interface
------------------------
log_likelihood(theta [N, K*p], designs [N, T, p], outcomes [N, T, 1]) -> [N]

Each theta[n] is evaluated against its paired trajectory (designs[n], outcomes[n]).
The caller is responsible for tiling designs/outcomes when evaluating against many
theta samples (e.g. IS posterior inference, sPCE contrastive samples).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.distributions as dist
from torch import Tensor


class LocationFindingPrior(nn.Module):
    """Factorised Normal prior over all K source locations.

    Samples have shape [n, K*p].  The first p entries correspond to source 0,
    the next p to source 1, and so on 

    Args:
        K: number of sources.
        p: spatial dimension of each source (default 2).
        loc: prior mean for every coordinate (default 0).
        scale: prior std for every coordinate (default 1).
        device: torch device.
    """

    def __init__(
        self,
        K: int = 1,
        p: int = 2,
        loc: float = 0.0,
        scale: float = 1.0,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        self.K = K
        self.p = p
        self.theta_dim = K * p
        self.device = torch.device(device)
        self._dist = dist.Normal(
            torch.zeros(self.theta_dim, device=self.device) + loc,
            torch.ones(self.theta_dim, device=self.device) * scale,
        )

    def sample(self, n: int) -> Tensor:
        """Return n prior samples of shape [n, K*p]."""
        return self._dist.sample((n,))

    def log_prob(self, theta: Tensor) -> Tensor:
        """Log prior density, summed over K*p coordinates.

        Args:
            theta: [n, K*p]
        Returns:
            [n]
        """
        return self._dist.log_prob(theta).sum(-1)


class LocationFindingModel(nn.Module):
    """Source location finding generative model.

    Args:
        prior: LocationFindingPrior — defines K, p, and the prior density.
        design_net: policy network; called as design_net(designs, outcomes).
        T: total experiment steps.
        alpha: per-source signal strengths, length K (default all-ones).
        b: background signal level (default 0.1).
        m: near-field constant preventing 1/0 at zero distance (default 1e-4).
        sigma: observation noise std (default 0.5).
        device: torch device.
    """

    def __init__(
        self,
        prior: LocationFindingPrior,
        design_net: nn.Module,
        T: int = 10,
        alpha: list[float] | None = None,
        b: float = 0.1,
        m: float = 1e-4,
        sigma: float = 0.5,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        self.prior = prior
        self.design_net = design_net
        self.T = T
        self.b = b
        self.m = m
        self.sigma = sigma
        self.device = torch.device(device)
        self.K = prior.K
        self.p = prior.p

        if alpha is None:
            alpha = [1.0] * self.K
        self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32, device=self.device))  # [K]

    # ------------------------------------------------------------------
    # Likelihood helpers
    # ------------------------------------------------------------------

    def _signal_mean(self, theta: Tensor, xi: Tensor) -> Tensor:
        """Compute mu(theta, xi) = b + sum_k alpha_k / (m + ||theta_k - xi||^2).

        theta is reshaped from [..., K*p] → [..., K, p] so that
        theta[..., k, :] is the p-dimensional location of source k.

        The denominator (m + sq_dist) >= m > 0 so no division-by-zero occurs.

        Args:
            theta: [..., K*p]
            xi:    [..., p]
        Returns:
            [..., 1]
        """
        theta_k = theta.view(*theta.shape[:-1], self.K, self.p)         # [..., K, p]
        xi_exp = xi.unsqueeze(-2)                                        # [..., 1, p]
        sq_dist = ((theta_k - xi_exp) ** 2).sum(-1)                     # [..., K]
        signal = (self.alpha / (self.m + sq_dist)).sum(-1, keepdim=True) # [..., 1]
        return self.b + signal

    def outcome_likelihood(self, theta: Tensor, xi: Tensor) -> dist.Distribution:
        """Return p(y | theta, xi) = Normal(log mu(theta, xi), sigma^2).

        Args:
            theta: [..., K*p]
            xi:    [..., p]
        Returns:
            Normal distribution over log y.
        """
        mu = self._signal_mean(theta, xi)
        return dist.Normal(torch.log(mu), self.sigma)

    def log_likelihood(self, theta: Tensor, designs: Tensor, outcomes: Tensor) -> Tensor:
        """Evaluate sum_t log p(y_t | theta_n, xi_t^n) — one value per sample n.

        Each theta[n] is paired with its own trajectory (designs[n], outcomes[n]).
        The caller tiles designs/outcomes when evaluating many theta samples against
        a single experimental history (e.g. IS or sPCE contrastive samples).

        Args:
            theta:    [N, K*p]
            designs:  [N, T, p]
            outcomes: [N, T, 1]
        Returns:
            [N] — summed log-likelihood over T steps.
        """
        N, T, _ = designs.shape
        log_lk = torch.zeros(N, device=theta.device)
        for t in range(T):
            log_lk += self.outcome_likelihood(
                theta, designs[:, t, :]
            ).log_prob(outcomes[:, t, :]).squeeze(-1)   # [N]
        return log_lk

    # ------------------------------------------------------------------
    # Simulation (used for training the policy)
    # ------------------------------------------------------------------

    def forward(
        self,
        batch_size: int,
        past_designs: Tensor | None = None,
        past_outcomes: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Simulate a batch of T-step experiments by running design_net.

        Returns:
            theta:    [B, K*p]
            designs:  [B, T_new, p]
            outcomes: [B, T_new, 1]
        """
        theta = self.prior.sample(batch_size).to(self.device)

        designs = (
            torch.empty(batch_size, 0, self.p, device=self.device)
            if past_designs is None else past_designs
        )
        outcomes = (
            torch.empty(batch_size, 0, 1, device=self.device)
            if past_outcomes is None else past_outcomes
        )
        n_past = designs.shape[1]

        for _ in range(n_past, self.T):
            xi = self.design_net(designs, outcomes)
            yi = self.outcome_likelihood(theta, xi).rsample()
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
        """Roll out the current policy for a fixed theta (live experiment).

        Returns:
            designs:  [B, T, d]
            outcomes: [B, T, 1]
        """
        B = theta.shape[0]
        designs = (
            torch.empty(B, 0, self.p, device=self.device)
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
