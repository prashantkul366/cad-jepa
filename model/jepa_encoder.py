"""
CAD-JEPA Context Encoder  (E_theta)

Paper section 3.2:
  Input  : visible (unmasked) operation tokens  [B, T_visible, 17]
  Embed  : type embedding + per-param embeddings (256-bin vocab)  -> d=768
  Pos enc: RoPE via rotary-embedding-torch
  Body   : 12-layer TransformerEncoder, d=768, 8 heads
  Output : contextualized latents  [B, T_visible, 768]

Only this encoder receives gradients.
The target encoder is an EMA copy — see model/ema.py.
"""

from typing import Optional

import torch
import torch.nn as nn
from rotary_embedding_torch import RotaryEmbedding

from config.configJEPA import ConfigJEPA


class CADTokenEmbedding(nn.Module):
    """
    [B, T, 17] int tokens  ->  [B, T, encoder_dim] float embeddings

    Paper Eq.(1):
        e_i = LayerNorm( W_type(type_i) || W_p0(p0) || ... || W_p15(p15) )
              projected to encoder_dim.

    Check cadlib/macro.py for exact command-type integer values.
    Check dataset/json2vec.py for how params are quantized to 0-255.
    """

    def __init__(self, cfg: ConfigJEPA):
        super().__init__()
        # TODO
        # d_type  = 64   (TUNE)
        # d_param = 32   (TUNE)
        # self.type_emb   = nn.Embedding(cfg.num_commands, d_type)
        # self.param_embs = nn.ModuleList([
        #     nn.Embedding(cfg.param_vocab_size, d_param) for _ in range(cfg.num_params)
        # ])
        # self.proj = nn.Linear(d_type + cfg.num_params * d_param, cfg.encoder_dim)
        # self.norm = nn.LayerNorm(cfg.encoder_dim)
        pass

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [B, T, 17]  col-0=type, col1-16=params
        # TODO: embed type + each param, concat, project, layernorm
        # return: [B, T, encoder_dim]
        pass


class CADEncoder(nn.Module):

    def __init__(self, cfg: ConfigJEPA):
        super().__init__()
        # TODO
        # self.token_emb = CADTokenEmbedding(cfg)
        # self.rope = RotaryEmbedding(dim = cfg.encoder_dim // cfg.encoder_heads)
        # layer = nn.TransformerEncoderLayer(
        #     d_model=cfg.encoder_dim, nhead=cfg.encoder_heads,
        #     dim_feedforward=cfg.encoder_dim * 4,
        #     dropout=cfg.encoder_dropout, batch_first=True)
        # self.transformer = nn.TransformerEncoder(layer, cfg.encoder_layers)
        pass

    def forward(
        self,
        tokens: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # tokens : [B, T, 17]
        # returns: [B, T, 768]
        # TODO:
        #   x = self.token_emb(tokens)
        #   apply RoPE inside attention heads (rotary_embedding_torch handles this)
        #   return self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        pass
