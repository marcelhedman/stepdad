"""
Sequential Prior Contrastive Estimation (sPCE) bounds on EIG.

Given a model with interface:
    model.log_likelihood(theta [N, theta_dim], designs [N, T, p], outcomes [N, T, obs_dim]) -> [N]

and a prior with interface:
    model.prior.sample(n) -> [n, theta_dim]

this module provides:
- sPCE lower bound  
- sNMC upper bound  
- A differentiable training loss for the lower bound

The convention throughout:
    B = batch size (number of independent trajectories per gradient step)
    L = number of contrastive samples per trajectory
    T = number of experiment steps

For each batch element b:
    theta_0[b]         ~ prior   (the "true" parameter for that trajectory)
    theta_l[b, l]      ~ prior   (L contrastive samples, l = 1..L)
    (designs[b], outcomes[b])  simulated under theta_0[b] and the policy

"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
from torch import Tensor
from typing import Literal


def _contrastive_log_prob(
    model: nn.Module,
    theta_0: Tensor,
    designs: Tensor,
    outcomes: Tensor,
    L: int,
    lower_bound: bool,
) -> tuple[Tensor, Tensor]:
    """Compute primary and denominator log-likelihoods for sPCE / sNMC.

    Args:
        model:       generative model with .log_likelihood and .prior.sample.
        theta_0:     [B, theta_dim] — primary (true) parameters.
        designs:     [B, T, p]
        outcomes:    [B, T, obs_dim]
        L:           number of contrastive samples.
        lower_bound: if True, include theta_0 in the denominator (sPCE lower);
                     if False, exclude it (sNMC upper).

    Returns:
        log_p0:      [B] — log p(y | theta_0).
        denominator: [B] — logsumexp of log p(y | theta_l) over contrastive set.
    """
    B = theta_0.shape[0]
    T = designs.shape[1]

    # Primary log-likelihood: pair theta_0[b] with trajectory b
    log_p0 = model.log_likelihood(theta_0, designs, outcomes)  # [B]

    # Contrastive samples: L samples per batch element
    theta_l = model.prior.sample(L * B).to(theta_0.device)    # [L*B, theta_dim]
    # Tile designs/outcomes to match: each of the L*B samples is paired with
    # the trajectory from its corresponding batch element b = (sample_idx % B)
    designs_tiled = designs.unsqueeze(0).expand(L, B, T, -1).reshape(L * B, T, -1)
    outcomes_tiled = outcomes.unsqueeze(0).expand(L, B, T, -1).reshape(L * B, T, -1)

    log_p_l = model.log_likelihood(theta_l, designs_tiled, outcomes_tiled)  # [L*B]
    log_p_l = log_p_l.view(L, B)                                             # [L, B]

    if lower_bound:
        # Include theta_0 in denominator: [L+1, B]
        log_stack = torch.cat([log_p0.unsqueeze(0), log_p_l], dim=0)
        n_denom = L + 1
    else:
        # Exclude theta_0 from denominator: [L, B]
        log_stack = log_p_l
        n_denom = L

    denominator = torch.logsumexp(log_stack, dim=0)  # [B]
    return log_p0, denominator, n_denom


class SPCELoss(nn.Module):
    """Differentiable sPCE training loss for policy optimisation.

    Minimising this loss maximises the sPCE lower bound on EIG.

    gradient_estimator:
        - "rparam": pathwise gradient through outcomes/designs
        - "reinforce": score-function-style estimator for outcome sampling


    Args:
        model:       generative model.
        L:           number of contrastive samples per trajectory (default 1023).
        lower_bound: use sPCE lower bound (True) or sNMC upper bound (False).
    """

    def __init__(self, model: nn.Module, gradient_estimator: Literal["rparam", "reinforce"],
                 L: int = 1023, lower_bound: bool = True,
                 ) -> None:
        super().__init__()
        self.model = model
        self.L = L
        self.lower_bound = lower_bound
        self.gradient_estimator = gradient_estimator

    def forward(
        self,
        theta_0: Tensor,
        designs: Tensor,
        outcomes: Tensor,
    ) -> Tensor:
        """Compute the differentiable training loss.

        Args:
            theta_0:  [B, theta_dim]
            designs:  [B, T, d]
            outcomes: [B, T, obs_dim]
        Returns:
            Scalar loss (to be minimised by the optimiser).
        """
        log_p0, denominator, _ = _contrastive_log_prob(
            self.model, theta_0, designs, outcomes, self.L, self.lower_bound
        )

   
        if self.gradient_estimator == "rparam":
            # Pathwise estimator: directly differentiate sampled objective.
            if not outcomes.requires_grad:
                raise RuntimeError(
                    "gradient_estimator='rparam' but outcomes.requires_grad is False. "
                    "This suggests rsample()/pathwise sampling was not used."
                )
            loss = -(log_p0 - denominator).mean()

        elif self.gradient_estimator == "reinforce":
            # Score-function-style surrogate.
            reward = (log_p0 - denominator).detach()
            loss = -(reward * log_p0 - denominator).mean()

        else:
            raise ValueError(
                f"Unknown gradient_estimator={self.gradient_estimator!r}. "
                "Expected 'rparam' or 'reinforce'."
            )

        if not torch.isfinite(loss):
            raise RuntimeError("Loss became NaN or Inf.")

        return loss


@torch.no_grad()
def estimate_eig(
    model: nn.Module,
    theta_0: Tensor,
    designs: Tensor,
    outcomes: Tensor,
    L: int = 100_000,
    lower_bound: bool = True,
) -> float:
    """Evaluate a tight sPCE lower (or sNMC upper) bound on EIG.

    Use a large L (default 100K) for a tight evaluation bound.

    Args:
        model:       generative model.
        theta_0:     [B, theta_dim]
        designs:     [B, T, p]
        outcomes:    [B, T, obs_dim]
        L:           number of contrastive samples (large → tight bound).
        lower_bound: True for sPCE lower, False for sNMC upper.
    Returns:
        Scalar EIG estimate.
    """
    log_p0, denominator, n_denom = _contrastive_log_prob(
        model, theta_0, designs, outcomes, L, lower_bound
    )
    return (log_p0 - denominator).mean().item() + math.log(n_denom)
