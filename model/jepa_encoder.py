"""
CAD-JEPA Context Encoder

CADEmbedding  : copied verbatim from autoencoder_deepcad_reference.py
CADJEPAEncoder: adapted from Encoder class, 3 JEPA-specific changes marked below
All transformer layers from model/layers/ — unchanged
"""

import torch
import torch.nn as nn
from torch.nn.modules.normalization import LayerNorm

from model.layers.transformer import TransformerEncoder
from model.layers.improved_transformer import TransformerEncoderLayerImproved
from model.layers.positional_encoding import PositionalEncodingLUT
from model.model_utils import (
    _make_seq_first, _make_batch_first, _get_key_padding_mask
)


class CADEmbedding(nn.Module):
    """Copied verbatim from autoencoder_deepcad_reference.py."""
    def __init__(self, cfg, seq_len):
        super().__init__()
        self.command_embed = nn.Embedding(cfg.n_commands, cfg.d_model)
        args_dim = cfg.args_dim + 1          # 257: PAD(-1) → index 0
        self.arg_embed  = nn.Embedding(args_dim, 64, padding_idx=0)
        self.embed_fcn  = nn.Linear(64 * cfg.n_args, cfg.d_model)
        self.pos_encoding = PositionalEncodingLUT(cfg.d_model, max_len=seq_len + 2)

    def forward(self, commands, args):
        S, N = commands.shape
        src = self.command_embed(commands.long()) + \
              self.embed_fcn(self.arg_embed((args + 1).long()).view(S, N, -1))
        return self.pos_encoding(src)        # [S, N, d_model]


class CADJEPAEncoder(nn.Module):
    """
    Context encoder E_theta.
    Adapted from DeepCAD's Encoder — only 3 changes from original (marked JEPA).
    """
    def __init__(self, cfg):
        super().__init__()
        self.embedding = CADEmbedding(cfg, cfg.max_total_len)
        encoder_layer  = TransformerEncoderLayerImproved(
            cfg.d_model, cfg.n_heads, cfg.dim_feedforward, cfg.dropout)
        self.encoder   = TransformerEncoder(encoder_layer, cfg.n_layers, LayerNorm(cfg.d_model))

    def forward(self, commands, args, jepa_mask=None):
        """
        commands  : [N, S]      int64
        args      : [N, S, 16]  int64
        jepa_mask : [N, S]      bool — True = hide from context encoder

        returns   : [N, S, d_model]
        """
        commands_, args_ = _make_seq_first(commands, args)       # [S,N], [S,N,16]
        key_padding_mask = _get_key_padding_mask(commands_, seq_dim=0)  # [N,S]

        # JEPA CHANGE 1: hide semantically masked blocks from context encoder
        if jepa_mask is not None:
            key_padding_mask = key_padding_mask | jepa_mask

        src    = self.embedding(commands_, args_)                  # [S,N,d]
        memory = self.encoder(src, mask=None,
                              src_key_padding_mask=key_padding_mask)

        # JEPA CHANGE 2: no bottleneck — return full d_model
        # JEPA CHANGE 3: return ALL positions, not mean-pooled
        return _make_batch_first(memory)                           # [N,S,d]

    @torch.no_grad()
    def encode_mean(self, commands, args):
        """Mean-pool over valid positions → single vector. Used for retrieval / Stage 2."""
        h    = self.forward(commands, args)                        # [N,S,d]
        valid = (commands != 3).float().unsqueeze(-1)              # [N,S,1]
        return (h * valid).sum(1) / valid.sum(1).clamp(min=1)     # [N,d]