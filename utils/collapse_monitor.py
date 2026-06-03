"""
Collapse Monitor for CAD-JEPA

Tracks effective feature rank of encoder representations.
Activates VICReg if rank drops below threshold.
Per paper: EMA + stop-grad is sufficient; this fires infrequently.
"""

import torch
import torch.nn.functional as F


class CollapseMonitor:

    def __init__(self, encoder_dim: int, rank_threshold: float = 0.70):
        self.encoder_dim     = encoder_dim
        self.rank_threshold  = rank_threshold
        self._last_rank_frac = 1.0

    def effective_rank(self, h: torch.Tensor) -> float:
        h_flat = h.detach().reshape(-1, self.encoder_dim).float()
        if h_flat.shape[0] < 2:
            return 1.0
        # _, s, _ = torch.pca_lowrank(h_flat, q=min(64, self.encoder_dim))
        _, s, _ = torch.pca_lowrank(h_flat, q=min(self.encoder_dim, h_flat.shape[0] - 1))
        s = s ** 2
        cumvar = torch.cumsum(s / s.sum(), dim=0)
        rank = int((cumvar < 0.99).sum().item()) + 1
        self._last_rank_frac = rank / self.encoder_dim
        return self._last_rank_frac

    def is_collapsing(self) -> bool:
        return self._last_rank_frac < self.rank_threshold

    def regularization_loss(
        self,
        h: torch.Tensor,
        lambda_v: float = 25.0,
        lambda_c: float = 1.0,
    ) -> torch.Tensor:
        """VICReg variance + covariance regularization (Bardes et al. 2022)."""
        z  = h.reshape(-1, self.encoder_dim).float()
        N, D = z.shape
        std    = torch.sqrt(z.var(dim=0) + 1e-4)
        v_loss = torch.mean(F.relu(1.0 - std))
        z_norm = z - z.mean(dim=0)
        cov    = (z_norm.T @ z_norm) / (N - 1)
        off    = cov ** 2
        off.fill_diagonal_(0.0)
        c_loss = off.sum() / D
        return lambda_v * v_loss + lambda_c * c_loss
