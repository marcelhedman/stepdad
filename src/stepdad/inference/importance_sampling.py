"""
Importance sampling (IS) posterior inference.

Uses the prior as the proposal distribution — self-normalised IS:

    weights[n] ∝ p(y_{1:tau} | theta[n], xi_{1:tau})
    theta[n] ~ p(theta)   for n = 1, ..., N


The effective sample size (ESS) is reported as a diagnostic.  If ESS falls
below `min_ess_fraction * n_samples`, a warning is printed.

Usage
-----
    samples, log_weights, ESS = importance_sample(
        model, designs, outcomes, n_samples=20_000
    )
    posterior_samples = resample(samples, log_weights, n_resample=1000)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


def importance_sample(
    model: nn.Module,
    designs: Tensor,
    outcomes: Tensor,
    n_samples: int = 20_000,
    min_ess_fraction: float = 0.05,
    device: torch.device | str | None = None,
) -> tuple[Tensor, Tensor, float]:
    """Draw posterior samples via importance sampling.

    Args:
        model:             generative model with .prior.sample(n) and
                           .log_likelihood(theta, designs, outcomes).
        designs:           [1, T, p]  — single experimental history.
        outcomes:          [1, T, obs_dim]
        n_samples:         number of IS samples (default 20 000).
        min_ess_fraction:  warn if ESS / n_samples < this fraction (default 5%).
        device:            device for sampling; defaults to designs.device.

    Returns:
        samples:     [n_samples, theta_dim] — samples drawn from the prior.
        log_weights: [n_samples]            — normalised log importance weights.
        ESS:         float                  — effective sample size.
    """
    if device is None:
        device = designs.device

    # Sample from prior
    theta = model.prior.sample(n_samples).to(device)       # [N, theta_dim]

    # Tile the single history to match all N samples
    N = n_samples
    T = designs.shape[1]
    p = designs.shape[2]
    obs_dim = outcomes.shape[2]

    designs_tiled = designs.expand(N, T, p)                # [N, T, p]
    outcomes_tiled = outcomes.expand(N, T, obs_dim)        # [N, T, obs_dim]

    # Log-likelihood for each sample
    with torch.no_grad():
        log_lk = model.log_likelihood(theta, designs_tiled, outcomes_tiled)  # [N]

    # Normalised log importance weights (prior = proposal, so prior log_prob cancels)
    log_weights = log_lk - torch.logsumexp(log_lk, dim=0)  # [N]

    # Effective sample size
    weights = log_weights.exp()
    ESS = (weights.sum() ** 2 / (weights ** 2).sum()).item()

    if ESS < min_ess_fraction * n_samples:
        print(
            f"[IS] Warning: ESS = {ESS:.1f} / {n_samples} "
            f"({100 * ESS / n_samples:.1f}%) is below "
            f"{100 * min_ess_fraction:.0f}% threshold."
        )

    return theta, log_weights, ESS


def resample(
    samples: Tensor,
    log_weights: Tensor,
    n_resample: int,
) -> Tensor:
    """Draw n_resample samples from the IS approximation via multinomial resampling.

    Args:
        samples:     [N, theta_dim]
        log_weights: [N] — normalised log weights (from importance_sample).
        n_resample:  number of samples to draw.
    Returns:
        [n_resample, theta_dim]
    """
    weights = log_weights.exp()
    indices = torch.multinomial(weights, num_samples=n_resample, replacement=True)
    return samples[indices]
