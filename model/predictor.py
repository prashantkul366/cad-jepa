"""
CAD-JEPA Predictor  (P_phi)

Paper section 3.2:
  Intentionally NARROW: 3 layers, d=256  (encoder is d=768)
  Input  : context latents [B, T_vis, 768] + positional queries for masked blocks
  Output : predicted latents [B, n_masked, 768]
  Loss   : L1( predicted, sg(target_encoder_output_at_masked_positions) )

Ablation (Table 4): 3-layer is optimal. 4-layer slightly worse (can shortcut encoder).
"""

from typing import Optional

import torch
import torch.nn as nn

from config.configJEPA import ConfigJEPA


class CADPredictor(nn.Module):

    def __init__(self, cfg: ConfigJEPA):
        super().__init__()
        # TODO
        # self.input_proj  = nn.Linear(cfg.encoder_dim, cfg.predictor_dim)
        # self.mask_tokens = nn.Embedding(cfg.max_seq_len, cfg.predictor_dim)
        #   learnable query per masked-block position index
        # layer = nn.TransformerEncoderLayer(
        #     d_model=cfg.predictor_dim, nhead=4,
        #     dim_feedforward=cfg.predictor_dim * 4,
        #     batch_first=True)
        # self.transformer = nn.TransformerEncoder(layer, cfg.predictor_layers)
        # self.output_proj = nn.Linear(cfg.predictor_dim, cfg.encoder_dim)
        pass

    def forward(
        self,
        context_latents: torch.Tensor,
        mask_positions: torch.Tensor,
    ) -> torch.Tensor:
        # context_latents : [B, T_visible, encoder_dim]
        # mask_positions  : [B, n_masked]  integer position indices
        # returns         : [B, n_masked, encoder_dim]
        # TODO:
        #   1. project context down to predictor_dim
        #   2. look up learnable queries for each masked position
        #   3. concat context tokens + query tokens -> transformer
        #   4. select query positions -> project back up to encoder_dim
        pass
