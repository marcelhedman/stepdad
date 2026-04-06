"""
Deep Adaptive Design (DAD) policy networks.

Two variants:
- DADPolicy: simple sum-aggregation used for location finding and temporal discounting.
- CESDADPolicy: separate design/outcome encoders with LayerNorm, used for CES.

Both implement the same interface:
    forward(designs [B, t, design_dim], outcomes [B, t, obs_dim]) -> next_design [B, design_dim]

The policy is permutation-invariant in the history via sum-aggregation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# Simple DAD (location finding, temporal discounting)
# ---------------------------------------------------------------------------

class _Encoder(nn.Module):
    """Maps a (design, outcome) pair to a fixed-dim embedding."""

    def __init__(self, design_dim: int, obs_dim: int, hidden_dim: int, encoding_dim: int) -> None:
        super().__init__()
        self.encoding_dim = encoding_dim
        self.net = nn.Sequential(
            nn.Linear(design_dim + obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, encoding_dim),
        )

    def forward(self, xi: Tensor, y: Tensor) -> Tensor:
        """
        Args:
            xi: [B, design_dim]
            y:  [B, obs_dim]
        Returns:
            [B, encoding_dim]
        """
        return self.net(torch.cat([xi, y], dim=-1))


class _Emitter(nn.Module):
    """Maps an aggregated encoding to the next design."""

    def __init__(self, encoding_dim: int, design_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(encoding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 16),
            nn.ReLU(),
            nn.Linear(16, design_dim),
        )

    def forward(self, embedding: Tensor) -> Tensor:
        """
        Args:
            embedding: [B, encoding_dim]
        Returns:
            [B, design_dim]
        """
        return self.net(embedding)


class DADPolicy(nn.Module):
    """Permutation-invariant DAD policy via sum-aggregation.


    Args:
        design_dim:   dimensionality of each design.
        obs_dim:      dimensionality of each observation.
        hidden_dim:   hidden width of the encoder first layer (default 256).
        encoding_dim: width of the encoder output / emitter input (default 16).
    """

    def __init__(
        self,
        design_dim: int,
        obs_dim: int,
        hidden_dim: int = 256,
        encoding_dim: int = 16,
    ) -> None:
        super().__init__()
        self.encoding_dim = encoding_dim
        self.encoder = _Encoder(design_dim, obs_dim, hidden_dim, encoding_dim)
        self.emitter = _Emitter(encoding_dim, design_dim)
        self.register_parameter(
            "empty_value", nn.Parameter(torch.zeros(encoding_dim))
        )

    def forward(self, designs: Tensor, outcomes: Tensor) -> Tensor:
        """Propose the next design given the history.

        Args:
            designs:  [B, t, design_dim]
            outcomes: [B, t, obs_dim]
        Returns:
            [B, design_dim]
        """
        B = designs.shape[0]
        t = designs.shape[1]

        if t == 0:
            # No history — use learned empty value broadcast over batch
            enc = self.empty_value.unsqueeze(0).expand(B, -1)
        else:
            enc = sum(
                self.encoder(designs[:, i, :], outcomes[:, i, :])
                for i in range(t)
            )

        return self.emitter(enc)


# ---------------------------------------------------------------------------
# CES DAD (separate encoders, LayerNorm, optional time embedding)
# ---------------------------------------------------------------------------

class CESDADPolicy(nn.Module):
    """DAD policy for CES.

    Differences from DADPolicy:
    - Separate Linear+LayerNorm+ReLU encoders for designs and outcomes.
    - A two-block 'head' network processes the concatenation.
    - Optional per-timestep time embeddings.
    - Zero-initialised final decode layer for stable early training.

    Args:
        design_dim:    dimensionality of each design (6 for CES).
        obs_dim:       dimensionality of each observation (1 for CES).
        T:             total number of experiment steps (needed for time embeddings).
        hidden_dim:    hidden width in head and decoder (default 256).
        embedding_dim: width of per-pair embeddings (default 32).
        time_embedding: whether to concatenate a learned time embedding (default True).
    """

    def __init__(
        self,
        design_dim: int,
        obs_dim: int,
        T: int,
        hidden_dim: int = 256,
        embedding_dim: int = 32,
        time_embedding: bool = True,
    ) -> None:
        super().__init__()
        self.design_dim = design_dim
        self.T = T
        self.time_embedding = time_embedding

        # Separate encoders for designs and outcomes
        self.encode_designs = nn.Sequential(
            nn.Linear(design_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
        )
        self.encode_outcomes = nn.Sequential(
            nn.Linear(obs_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
        )

        # Two-block head
        self.head = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
        )

        # Decoder
        self.decode_designs = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, design_dim),
        )
        # Zero-init the final layer for training stability
        nn.init.zeros_(self.decode_designs[-1].weight)
        nn.init.zeros_(self.decode_designs[-1].bias)

        if time_embedding:
            self.time_projection = nn.Linear(embedding_dim * 2, embedding_dim)
            self.register_parameter(
                "time_embeddings", nn.Parameter(torch.rand(T, embedding_dim))
            )

    def forward(self, designs: Tensor, outcomes: Tensor) -> Tensor:
        """Propose the next design given the history.

        Args:
            designs:  [B, t, design_dim]  — in constrained space for CES
            outcomes: [B, t, obs_dim]
        Returns:
            [B, design_dim]  — in unconstrained space (model applies transform)
        """
        B = designs.shape[0]
        t = designs.shape[1]

        if t == 0:
            x = torch.zeros(B, self.head[-2].normalized_shape[0], device=designs.device)
        else:
            enc_d = self.encode_designs(designs)   # [B, t, embedding_dim]
            enc_y = self.encode_outcomes(outcomes)  # [B, t, embedding_dim]
            x = torch.cat([enc_d, enc_y], dim=-1)  # [B, t, 2*embedding_dim]
            x = self.head(x)                        # [B, t, embedding_dim]
            x = x.sum(dim=1)                        # [B, embedding_dim]

        if self.time_embedding:
            # time_embeddings[t-1]: at t=0 this wraps to the last embedding
            # (matches original bed_core behaviour)
            time_emb = self.time_embeddings[t - 1].unsqueeze(0).expand(B, -1)
            x = torch.cat([x, time_emb], dim=-1)
            x = self.time_projection(x)

        return self.decode_designs(x)
