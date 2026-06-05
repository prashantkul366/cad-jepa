"""
model/decoder_nar.py
NAR Decoder for CAD-JEPA Stage 3 — Drawing2CAD architecture.

z* [B, 512] → cmd_logits [B, 60, 6] + arg_logits [B, 60, 16, 257]

Key properties:
  - All 60 positions predicted in ONE forward pass (no teacher forcing)
  - z* injected via Linear(512→d_model) added at EVERY decoder layer
  - Command hidden states used as guidance for args (Drawing2CAD trick)
  - Training == Inference: no exposure bias, no distribution shift
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers.improved_transformer import TransformerDecoderLayerGlobalImproved
from .layers.transformer import TransformerDecoder
from .layers.positional_encoding import PositionalEncodingLUT


# ── Constant embedding (60 learned positional queries) ────────────────────────

class ConstEmbedding(nn.Module):
    """
    60 learned constant embeddings — one per output position.
    z* is passed in only to infer batch size + device.
    The actual conditioning happens inside each decoder layer.
    """
    def __init__(self, d_model: int, max_len: int = 60):
        super().__init__()
        self.max_len = max_len
        self.d_model = d_model
        self.PE = PositionalEncodingLUT(d_model, max_len=max_len)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: [1, B, dim_z]  (seq_first)
        N = z.size(1)
        src = self.PE(z.new_zeros(self.max_len, N, self.d_model))  # [60, B, d_model]
        return src


# ── Command decoder ────────────────────────────────────────────────────────────

class CommandDecoder(nn.Module):
    """
    Predicts command types for all 60 positions in parallel.
    Returns both logits AND hidden states (used as guidance for args).
    """
    def __init__(self, d_model: int, dim_z: int, n_heads: int,
                 dim_feedforward: int, n_layers: int, dropout: float,
                 n_commands: int, max_len: int = 60):
        super().__init__()
        self.embedding = ConstEmbedding(d_model, max_len)

        layer = TransformerDecoderLayerGlobalImproved(
            d_model, dim_z, n_heads, dim_feedforward, dropout
        )
        self.decoder = TransformerDecoder(layer, n_layers)

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.ReLU(),
            nn.Linear(d_model // 2, d_model // 4), nn.ReLU(),
            nn.Linear(d_model // 4, n_commands)
        )

    def forward(self, z: torch.Tensor):
        # z: [1, B, dim_z]
        src = self.embedding(z)                  # [60, B, d_model]
        out = self.decoder(src, z)               # [60, B, d_model]
        cmd_logits = self.head(out)              # [60, B, n_commands]
        cmd_logits = cmd_logits.permute(1, 0, 2) # [B, 60, n_commands]
        return cmd_logits, out                   # out = guidance for args


# ── Args decoder ───────────────────────────────────────────────────────────────

class ArgsDecoder(nn.Module):
    """
    Predicts args for all 60 positions in parallel.
    Receives guidance (command hidden states) — key Drawing2CAD trick.
    """
    def __init__(self, d_model: int, dim_z: int, n_heads: int,
                 dim_feedforward: int, n_layers: int, dropout: float,
                 n_args: int, args_dim: int, max_len: int = 60):
        super().__init__()
        self.n_args   = n_args
        self.args_dim = args_dim + 1   # +1 for PAD class

        self.embedding = ConstEmbedding(d_model, max_len)

        layer = TransformerDecoderLayerGlobalImproved(
            d_model, dim_z, n_heads, dim_feedforward, dropout
        )
        self.decoder = TransformerDecoder(layer, n_layers)

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.ReLU(),
            nn.Linear(d_model * 4, d_model * 2), nn.ReLU(),
            nn.Linear(d_model * 2, n_args * self.args_dim)
        )

    def forward(self, z: torch.Tensor, guidance: torch.Tensor):
        # z: [1, B, dim_z]   guidance: [60, B, d_model] from CommandDecoder
        src = self.embedding(z)                       # [60, B, d_model]
        out = self.decoder(src, z)                    # [60, B, d_model]
        out = out + guidance                          # command-guided args
        arg_logits = self.head(out)                   # [60, B, n_args*(args_dim+1)]

        S, N, _ = out.shape
        arg_logits = arg_logits.reshape(S, N, self.n_args, self.args_dim)
        arg_logits = arg_logits.permute(1, 0, 2, 3)  # [B, 60, n_args, args_dim+1]
        return arg_logits


# ── Main NAR decoder ───────────────────────────────────────────────────────────

class CADDecoderNAR(nn.Module):
    """
    Non-Autoregressive CAD sequence decoder.

    Input : z* [B, 512]
    Output: cmd_logits [B, 60, 6], arg_logits [B, 60, 16, 257]

    Usage (training):
        cmd_l, arg_l = decoder(z_star)
        loss, metrics = nar_loss(cmd_l, arg_l, tgt_commands, tgt_args)

    Usage (inference):
        cmd_l, arg_l = decoder(z_star)
        cmds = cmd_l.argmax(-1)          # [B, 60]
        args = arg_l.argmax(-1) - 1      # [B, 60, 16], -1=PAD
    """

    def __init__(
        self,
        latent_d       : int   = 512,
        d_model        : int   = 256,
        n_heads        : int   = 8,
        n_layers_decode: int   = 4,
        dim_feedforward : int  = 512,
        dropout        : float = 0.1,
        n_commands     : int   = 6,
        n_args         : int   = 16,
        args_dim       : int   = 256,
        max_len        : int   = 60,
    ):
        super().__init__()
        self.n_commands = n_commands
        self.n_args     = n_args
        self.args_dim   = args_dim
        self.max_len    = max_len

        # Project JEPA z* to decoder's internal dim if needed
        self.z_proj = nn.Linear(latent_d, latent_d) \
                      if latent_d != d_model else nn.Identity()
        # NOTE: dim_z passed to decoder layers = latent_d (512)
        # because linear_global inside each layer does Linear(512→d_model)

        self.command_decoder = CommandDecoder(
            d_model, latent_d, n_heads, dim_feedforward,
            n_layers_decode, dropout, n_commands, max_len
        )
        self.args_decoder = ArgsDecoder(
            d_model, latent_d, n_heads, dim_feedforward,
            n_layers_decode, dropout, n_args, args_dim, max_len
        )

    def forward(
        self,
        z_star: torch.Tensor,           # [B, 512]
    ):
        # Drawing2CAD uses seq_first format internally: [1, B, dim_z]
        z = z_star.unsqueeze(0)         # [1, B, 512]

        cmd_logits, guidance = self.command_decoder(z)   # [B,60,6], [60,B,d_model]
        arg_logits = self.args_decoder(z, guidance)       # [B,60,16,257]

        return cmd_logits, arg_logits

    def param_summary(self):
        n = sum(p.numel() for p in self.parameters())
        print(f"CADDecoderNAR — {n:,} parameters")
        for name, m in [('command_decoder', self.command_decoder),
                        ('args_decoder',    self.args_decoder)]:
            print(f"  {name}: {sum(p.numel() for p in m.parameters()):,}")


# ── Loss ──────────────────────────────────────────────────────────────────────

def nar_loss(
    cmd_logits  : torch.Tensor,   # [B, 60, 6]
    arg_logits  : torch.Tensor,   # [B, 60, 16, 257]
    tgt_commands: torch.Tensor,   # [B, 60]
    tgt_args    : torch.Tensor,   # [B, 60, 16]
    eos_idx     : int   = 3,
    label_smooth: float = 0.05,
):
    B, L = tgt_commands.shape

    # ── Structural weights (same as V2 — penalise premature SOL) ─────────────
    sw = _structural_weights(tgt_commands).to(cmd_logits.device)

    ce_cmd = F.cross_entropy(
        cmd_logits.reshape(-1, cmd_logits.size(-1)),
        tgt_commands.reshape(-1),
        reduction='none',
        label_smoothing=label_smooth,
    )
    cmd_loss = (ce_cmd * sw.reshape(-1)).sum() / sw.reshape(-1).sum()

    # ── Args loss (ignore PAD = class 0 after +1 shift) ──────────────────────
    args_target = (tgt_args + 1).clamp(min=0)
    arg_loss = F.cross_entropy(
        arg_logits.reshape(-1, arg_logits.size(-1)),
        args_target.reshape(-1),
        ignore_index=0,
        label_smoothing=label_smooth,
    )

    loss = cmd_loss + arg_loss

    with torch.no_grad():
        non_eos  = (tgt_commands != eos_idx)
        cmd_acc  = (cmd_logits.argmax(-1)[non_eos] == tgt_commands[non_eos]).float().mean()
        non_pad  = (tgt_args != -1)
        arg_acc  = ((arg_logits.argmax(-1) - 1)[non_pad] == tgt_args[non_pad]).float().mean()

    return loss, {
        'loss_total': loss.item(), 'loss_cmd': cmd_loss.item(),
        'loss_args': arg_loss.item(), 'acc_cmd': cmd_acc.item(), 'acc_args': arg_acc.item()
    }


def _structural_weights(tgt_commands: torch.Tensor, upweight: float = 5.0):
    B, L = tgt_commands.shape
    w = torch.ones(B, L, dtype=torch.float, device=tgt_commands.device)
    cmds_cpu = tgt_commands.cpu().tolist()
    for b in range(B):
        n_la = 0; has_c = False; first = True; n_cif = 0
        for t in range(L):
            cmd = cmds_cpu[b][t]
            if cmd == 3: break
            elif cmd == 5: n_cif = 0; n_la = 0; has_c = False
            elif cmd == 4:
                if not first:
                    if n_la < 2 and not has_c: w[b, t] = upweight
                    if has_c and n_cif >= 1:   w[b, t] = upweight
                n_cif += 1; n_la = 0; has_c = False; first = False
            elif cmd == 2: has_c = True
            elif cmd in (0, 1): n_la += 1
    return w