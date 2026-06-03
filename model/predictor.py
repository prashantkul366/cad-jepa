"""
CAD-JEPA Predictor P_phi

Uses TransformerDecoderLayerImproved from model/layers/improved_transformer.py:
  - Self-attention among masked position queries
  - Cross-attention to context encoder output (keys / values)

Intentionally narrow (predictor_d < d_model) to prevent shortcutting encoder.
"""

import torch
import torch.nn as nn
from torch.nn.modules.normalization import LayerNorm

from model.layers.transformer import TransformerDecoder
from model.layers.improved_transformer import TransformerDecoderLayerImproved


class CADPredictor(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        d_ctx  = cfg.d_model        # 256
        d_pred = cfg.predictor_d    # 128

        self.context_proj = nn.Linear(d_ctx, d_pred)
        self.mask_token   = nn.Embedding(cfg.max_total_len, d_pred)

        # TransformerDecoderLayerImproved: queries cross-attend to context
        # decoder_layer = TransformerDecoderLayerImproved(
        #     d_model=d_pred, nhead=4,
        #     dim_feedforward=d_pred * 4,
        #     dropout=0.0,          # no dropout in predictor
        # )
        decoder_layer = TransformerDecoderLayerImproved(
            d_model=d_pred, nhead=cfg.predictor_heads,
            dim_feedforward=d_pred * 4,
            dropout=0.0,          # no dropout in predictor
        )
        self.decoder    = TransformerDecoder(decoder_layer, cfg.predictor_layers,
                                             LayerNorm(d_pred))
        self.output_proj = nn.Linear(d_pred, d_ctx)

    def forward(self, context_reps: torch.Tensor,
                masked_positions: torch.Tensor) -> torch.Tensor:
        """
        context_reps    : [N, S, d_model]
        masked_positions: [N, n_mask]   int64 — original sequence indices
        returns         : [N, n_mask, d_model]
        """
        memory  = self.context_proj(context_reps)          # [N, S, d_pred]
        queries = self.mask_token(masked_positions)        # [N, n_mask, d_pred]

        # TransformerDecoder expects seq-first
        mem_sf = memory.permute(1, 0, 2)                   # [S, N, d_pred]
        q_sf   = queries.permute(1, 0, 2)                  # [n_mask, N, d_pred]

        out_sf = self.decoder(q_sf, mem_sf)                # [n_mask, N, d_pred]
        out    = out_sf.permute(1, 0, 2)                   # [N, n_mask, d_pred]

        return self.output_proj(out)                        # [N, n_mask, d_model]