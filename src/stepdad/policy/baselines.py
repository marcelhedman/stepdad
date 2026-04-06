"""
Baseline design networks: static (learnable fixed designs) and random.

Both implement the same interface as DADPolicy:
    forward(designs [B, t, design_dim], outcomes [B, t, obs_dim]) -> [B, design_dim]

StaticDesignNetwork
-------------------
Holds a learnable parameter of shape [T, design_dim].  At each step t the
network returns the t-th row, broadcast over the batch.  Training via
train_dad optimises these fixed designs jointly via the sPCE objective.

RandomDesignNetwork
-------------------
Returns a fresh Normal(0, 1) sample each call.  For models that apply a
constraint transform (e.g. CES), the sample passes through that transform
before being used.  No parameters, no training required.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class StaticDesignNetwork(nn.Module):
    """Learnable fixed-design baseline.

    Stores T design vectors as a trainable parameter.  The network always
    proposes design[t] regardless of the observed history, making it
    non-adaptive by construction.

    Args:
        design_dim: dimensionality of each design.
        T:          total number of experiment steps.
        init_std:   standard deviation for random initialisation (default 0.01).
    """

    def __init__(self, design_dim: int, T: int, init_std: float = 0.01) -> None:
        super().__init__()
        self.design_dim = design_dim
        self.T = T
        self.designs = nn.Parameter(torch.randn(T, design_dim) * init_std)

    def forward(self, designs: Tensor, outcomes: Tensor) -> Tensor:
        """Return the fixed design for the current step t.

        Args:
            designs:  [B, t, design_dim] — history (used only to infer t)
            outcomes: [B, t, obs_dim]    — history (ignored)
        Returns:
            [B, design_dim]
        """
        B = designs.shape[0]
        t = designs.shape[1]
        return self.designs[t].unsqueeze(0).expand(B, -1)


class RandomDesignNetwork(nn.Module):
    """Random design baseline.

    Samples i.i.d. Normal(0, 1) designs at each step, independent of all
    history.  For models that apply a constraint transform (e.g. CES applies
    an interval transform), the unconstrained Normal sample is used directly
    so the model can transform it into the feasible region.

    No learnable parameters.

    Args:
        design_dim: dimensionality of each design.
    """

    def __init__(self, design_dim: int) -> None:
        super().__init__()
        self.design_dim = design_dim

    def forward(self, designs: Tensor, outcomes: Tensor) -> Tensor:
        """Return a random design.

        Args:
            designs:  [B, t, design_dim] — history (used only to infer B and device)
            outcomes: [B, t, obs_dim]    — history (ignored)
        Returns:
            [B, design_dim]
        """
        B = designs.shape[0]
        return torch.randn(B, self.design_dim, device=designs.device)
