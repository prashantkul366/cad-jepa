"""
CAD AR Decoder — Stage 3

Cross-attends to JEPA latent sequence [B, 60, 512].
Autoregressively generates command + args tokens.

Stage 3A: memory = GT JEPA encoder output   (ceiling experiment)
Stage 3B: memory = LatentPredictor output   (text-conditioned)
Decoder architecture is identical in both cases.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from cadlib.macro import (
    SOL_IDX, EOS_IDX, EXT_IDX,
    LINE_IDX, ARC_IDX, CIRCLE_IDX,
    CMD_ARGS_MASK, N_ARGS, ARGS_DIM,
)

# Grammar: valid next command tokens
VALID_NEXT = {
    SOL_IDX    : {LINE_IDX, ARC_IDX, CIRCLE_IDX},
    LINE_IDX   : {LINE_IDX, ARC_IDX, CIRCLE_IDX, EXT_IDX},
    ARC_IDX    : {LINE_IDX, ARC_IDX, CIRCLE_IDX, EXT_IDX},
    CIRCLE_IDX : {EXT_IDX},
    EXT_IDX    : {SOL_IDX, EOS_IDX},
    EOS_IDX    : {EOS_IDX},
}

# CMD_ARGS_MASK[cmd, arg] = 1 means arg is active for this command
# shape [6, 16], already defined in cadlib.macro
_CMD_VALID_ARGS = torch.tensor(CMD_ARGS_MASK, dtype=torch.bool)  # [6, 16]


def cmd_valid_args(device):
    return _CMD_VALID_ARGS.to(device)


class CADDecoderBlock(nn.Module):
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
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x, memory, causal_mask=None):
        """
        x          : [B, S, d]
        memory     : [B, 60, d]  JEPA latents
        causal_mask: [S, S] upper-triangular -inf mask
        """
        x = x + self.drop(
            self.self_attn(
                self.norm1(x), self.norm1(x), self.norm1(x),
                attn_mask=causal_mask
            )[0]
        )
        x = x + self.drop(
            self.cross_attn(self.norm2(x), memory, memory)[0])
        x = x + self.drop(self.ffn(self.norm3(x)))
        return x


class CADDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d      = cfg.d_model           # 512
        n_h    = cfg.n_heads           # 8
        ff     = cfg.dim_feedforward   # 2048
        n_lay  = cfg.n_decoder_layers  # 12
        drop   = cfg.dropout           # 0.1

        # ── input embedding ──────────────────────────────────────────
        self.cmd_embed  = nn.Embedding(cfg.n_commands, d)
        self.arg_embed  = nn.Embedding(cfg.args_dim + 1, 64, padding_idx=0)
        self.embed_fcn  = nn.Linear(64 * cfg.n_args, d)

        # ── transformer blocks ────────────────────────────────────────
        self.layers = nn.ModuleList([
            CADDecoderBlock(d, n_h, ff, drop) for _ in range(n_lay)
        ])
        self.norm = nn.LayerNorm(d)

        # ── output heads ──────────────────────────────────────────────
        self.cmd_head  = nn.Linear(d, cfg.n_commands)            # → 6
        self.args_head = nn.Linear(d, cfg.n_args * cfg.args_dim) # → 16*256

        self.d        = d
        self.n_args   = cfg.n_args    # 16
        self.args_dim = cfg.args_dim  # 256
        self.max_len  = cfg.max_total_len  # 60

    # ── helpers ───────────────────────────────────────────────────────
    def _embed(self, commands, args):
        """commands [B,S], args [B,S,16] → [B,S,d]"""
        c = self.cmd_embed(commands.long())
        a = self.embed_fcn(
            self.arg_embed((args + 1).long())
                .view(args.size(0), args.size(1), -1)
        )
        return c + a

    @staticmethod
    def _causal_mask(S, device):
        return torch.triu(
            torch.full((S, S), float('-inf'), device=device), diagonal=1)

    # ── training forward ──────────────────────────────────────────────
    def forward(self, commands, args, memory):
        """
        commands : [B, S]       int64  full sequence (teacher forcing)
        args     : [B, S, 16]   int64
        memory   : [B, 60, 512] float  JEPA latents (GT or predicted)

        returns:
          cmd_logits  [B, S, 6]
          args_logits [B, S, 16, 256]
        """
        B, S = commands.shape
        x    = self._embed(commands, args)           # [B, S, d]
        mask = self._causal_mask(S, commands.device)

        for layer in self.layers:
            x = layer(x, memory, mask)
        x = self.norm(x)

        cl = self.cmd_head(x)                              # [B, S, 6]
        al = self.args_head(x).view(B, S, self.n_args,
                                    self.args_dim)         # [B, S, 16, 256]
        return cl, al

    # ── autoregressive generation ─────────────────────────────────────
    @torch.no_grad()
    def generate(self, memory, max_len=None, temperature=1.0, constrained=True):
        """
        memory : [B, 60, 512]
        returns: commands [B, L], args [B, L, 16]
        """
        B      = memory.size(0)
        device = memory.device
        L      = max_len or self.max_len
        cva    = cmd_valid_args(device)               # [6, 16]

        cmd_seq  = torch.full((B, 1), SOL_IDX,
                              dtype=torch.long, device=device)
        args_seq = torch.full((B, 1, self.n_args), -1,
                              dtype=torch.long, device=device)
        done = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(1, L):
            cl, al = self.forward(cmd_seq, args_seq, memory)

            next_cl = cl[:, -1, :]      # [B, 6]
            next_al = al[:, -1, :, :]   # [B, 16, 256]

            # grammar masking
            if constrained:
                inf_mask = torch.full_like(next_cl, float('-inf'))
                for b in range(B):
                    prev  = cmd_seq[b, -1].item()
                    valid = VALID_NEXT.get(prev, set(range(6)))
                    for v in valid:
                        inf_mask[b, v] = 0.0
                next_cl = next_cl + inf_mask

            next_cmd = torch.multinomial(
                F.softmax(next_cl / temperature, dim=-1), 1
            ).squeeze(-1)  # [B]

            # args
            next_args = torch.full((B, self.n_args), -1,
                                   dtype=torch.long, device=device)
            valid_mask = cva[next_cmd]          # [B, 16]
            sampled    = torch.multinomial(
                F.softmax(next_al / temperature, dim=-1)
                  .view(B * self.n_args, self.args_dim), 1
            ).view(B, self.n_args)              # [B, 16] values in [0, 255]
            next_args  = torch.where(valid_mask, sampled, next_args)

            cmd_seq  = torch.cat([cmd_seq,  next_cmd.unsqueeze(1)],  dim=1)
            args_seq = torch.cat([args_seq, next_args.unsqueeze(1)], dim=1)

            done = done | (next_cmd == EOS_IDX)
            if done.all():
                break

        return cmd_seq, args_seq