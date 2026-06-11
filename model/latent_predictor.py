"""
Latent Predictor — Stage 2B

Q-Former output [B, 64, 512] → predicted JEPA latents [B, 60, 512]

60 learned position queries cross-attend to Q-Former output
over 6 transformer blocks. Output lives in same space as
GT JEPA encoder output — decoder receives it identically.
"""

import torch
import torch.nn as nn


class LatentPredictorBlock(nn.Module):
    def __init__(self, d_model, n_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.self_attn  = nn.MultiheadAttention(d_model, n_heads,
                                                 batch_first=True, dropout=dropout)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads,
                                                 batch_first=True, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, d_model),
        )
        self.norm1   = nn.LayerNorm(d_model)
        self.norm2   = nn.LayerNorm(d_model)
        self.norm3   = nn.LayerNorm(d_model)
        self.drop    = nn.Dropout(dropout)

    def forward(self, q, memory):
        """
        q      : [B, 60, d]  position queries
        memory : [B, 64, d]  Q-Former output
        """
        q = q + self.drop(
            self.self_attn(self.norm1(q), self.norm1(q), self.norm1(q))[0])
        q = q + self.drop(
            self.cross_attn(self.norm2(q), memory, memory)[0])
        q = q + self.drop(self.ffn(self.norm3(q)))
        return q


class LatentPredictor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d_model  = cfg.d_model               # 512
        n_seq    = cfg.max_total_len         # 60
        n_heads  = cfg.n_heads               # 8
        ff_dim   = cfg.dim_feedforward       # 2048
        n_blocks = cfg.n_predictor_blocks    # 6
        dropout  = cfg.dropout               # 0.1

        # 60 learned position queries
        self.pos_queries = nn.Parameter(
            torch.randn(1, n_seq, d_model) * 0.02)

        self.blocks = nn.ModuleList([
            LatentPredictorBlock(d_model, n_heads, ff_dim, dropout)
            for _ in range(n_blocks)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, qformer_output):
        """
        qformer_output : [B, 64, 512]
        returns        : [B, 60, 512]
        """
        B = qformer_output.size(0)
        q = self.pos_queries.expand(B, -1, -1).clone()   # [B, 60, 512]
        for block in self.blocks:
            q = block(q, qformer_output)
        return self.norm(q)