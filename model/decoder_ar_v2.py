"""
model/decoder_ar_v2.py

8-block autoregressive Transformer decoder for DeepCAD sequences.
Architecture: Text2CAD-style conditioning + Drawing2CAD dual heads.

Key design decisions:
  - z* (512-d) → 8 memory tokens via linear projection
  - Blocks 0-1: self-attention only  (Text2CAD: "first two blocks do not use cross-attn")
  - Blocks 2-7: self-attention + cross-attention to z* memory tokens
  - Dual heads: command (6-way CE) + args (16×257 soft-target KL)
  - Command embedding added into args head input (command-guided args)
  - Grammar-constrained generation: NO single-curve loops, EVER

Grammar rules (the fix for mode collapse):
  SOL  → only when in NEED_SOL or AFTER_EXT state, OR when current loop is VALID
  LINE/ARC → only when loop is not a circle loop and under curve cap
  CIRCLE → only as first curve of a loop (IN_LOOP_EMPTY)
  EXT  → only when feature has ≥1 valid loop (≥3 LINE/ARC or 1 CIRCLE)
  EOS  → only after ≥1 EXT
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

# ── DeepCAD command indices ────────────────────────────────────────────────
LINE, ARC, CIRCLE, EOS, SOL, EXT = 0, 1, 2, 3, 4, 5
N_COMMANDS   = 6
N_ARGS       = 16
ARGS_DIM     = 256       # quantization bins: values 0-255
PAD_VAL      = -1        # unused arg slot in DeepCAD format
BOS_IDX      = 6         # distinct from any real command
MAX_LEN      = 60        # DeepCAD max sequence length

# ══════════════════════════════════════════════════════════════════════════
# Grammar
# ══════════════════════════════════════════════════════════════════════════

class CADGrammar:
    """
    Clean DeepCAD sequence grammar.

    The bug in prior decoder: SOL was allowed after any single curve,
    producing [SOL LINE SOL LINE SOL LINE SOL LINE EXT] — 4 single-line
    loops, each geometrically invalid (can't close a wire with 1 edge).

    This grammar prevents it: SOL is only allowed to start an INNER loop
    when the CURRENT loop is already valid (≥3 curves or 1 circle).

    State machine:
        NEED_SOL       → waiting to start a feature
        IN_LOOP_EMPTY  → SOL just emitted, no curves yet
        IN_LOOP_ACTIVE → ≥1 curve in current loop
        AFTER_EXT      → EXT just emitted, can start new feature or stop
        DONE           → EOS emitted
    """

    NEED_SOL       = 0
    IN_LOOP_EMPTY  = 1
    IN_LOOP_ACTIVE = 2
    AFTER_EXT      = 3
    DONE           = 4

    def __init__(self, n_ext_target: Optional[int] = None,
                 max_curves_per_loop: int = 12):
        self.n_ext_target        = n_ext_target    # force EOS when reached
        self.max_curves_per_loop = max_curves_per_loop
        self.reset()

    def reset(self):
        self.state              = self.NEED_SOL
        self.n_ext              = 0
        self.n_curves_in_loop   = 0
        self.has_circle         = False   # current loop has a circle
        self.loop_valid         = False   # current loop is closeable
        self.feature_has_valid  = False   # feature has ≥1 valid loop

    @property
    def _ext_limit(self) -> bool:
        return self.n_ext_target is not None and self.n_ext >= self.n_ext_target

    def allowed(self) -> frozenset:
        """Frozenset of command indices allowed at the current position."""

        if self.state == self.NEED_SOL:
            if self._ext_limit:
                return frozenset({EOS})
            cmds = {SOL}
            if self.n_ext > 0:
                cmds.add(EOS)          # can stop if at least one feature done
            return frozenset(cmds)

        if self.state == self.IN_LOOP_EMPTY:
            return frozenset({LINE, ARC, CIRCLE})

        if self.state == self.IN_LOOP_ACTIVE:
            cmds = set()
            # More LINE/ARC: only if not a circle loop and under cap
            if not self.has_circle and self.n_curves_in_loop < self.max_curves_per_loop:
                cmds.update({LINE, ARC})
            # EXT: only if this feature already has a valid closed loop
            if self.feature_has_valid:
                cmds.add(EXT)
            # Inner loop (hole): only if CURRENT loop is already valid
            # This is the critical fix — prevents single-curve loops
            if self.loop_valid:
                cmds.add(SOL)
            return frozenset(cmds)

        if self.state == self.AFTER_EXT:
            if self._ext_limit:
                return frozenset({EOS})
            return frozenset({SOL, EOS})

        return frozenset({EOS})   # DONE

    def step(self, cmd: int):
        """Advance state given the emitted command."""
        if cmd == SOL:
            self.n_curves_in_loop = 0
            self.has_circle       = False
            self.loop_valid       = False
            self.state            = self.IN_LOOP_EMPTY

        elif cmd in (LINE, ARC):
            self.n_curves_in_loop += 1
            self.state             = self.IN_LOOP_ACTIVE
            if self.n_curves_in_loop >= 3:
                self.loop_valid        = True
                self.feature_has_valid = True

        elif cmd == CIRCLE:
            self.n_curves_in_loop += 1
            self.has_circle        = True
            self.loop_valid        = True
            self.feature_has_valid = True
            self.state             = self.IN_LOOP_ACTIVE

        elif cmd == EXT:
            self.n_ext            += 1
            self.n_curves_in_loop  = 0
            self.has_circle        = False
            self.loop_valid        = False
            self.feature_has_valid = False
            self.state             = self.AFTER_EXT

        elif cmd == EOS:
            self.state = self.DONE

    def logit_mask(self, device: torch.device) -> torch.Tensor:
        """
        Return additive mask for command logits.
        Shape: [N_COMMANDS]. Value 0 = allowed, -inf = blocked.
        """
        mask = torch.full((N_COMMANDS,), float('-inf'), device=device)
        for cmd in self.allowed():
            if 0 <= cmd < N_COMMANDS:
                mask[cmd] = 0.0
        return mask


# ══════════════════════════════════════════════════════════════════════════
# Soft-target KL loss  (Drawing2CAD)
# ══════════════════════════════════════════════════════════════════════════

def soft_target_kl(pred_logits: torch.Tensor,
                   targets: torch.Tensor,
                   valid_mask: torch.Tensor,
                   tolerance: int = 3,
                   alpha: float = 2.0) -> torch.Tensor:
    """
    KL divergence with Gaussian soft targets over the 256-bin arg space.

    pred_logits : [N, ARGS_DIM+1]  raw logits  (257 bins: slot 0 = PAD)
    targets     : [N]              ground-truth bin index 0-255
    valid_mask  : [N]              bool, True where arg is not PAD
    tolerance   : Gaussian sigma in bins
    alpha       : Gaussian temperature

    Instead of one-hot CE, target distribution is a truncated Gaussian
    centered at the GT bin. This avoids over-penalising near-correct args
    (e.g., a coordinate off by 1 bin should get ~0 loss).
    """
    if valid_mask.sum() == 0:
        return pred_logits.sum() * 0.0   # differentiable zero

    D    = pred_logits.shape[-1]         # 257
    bins = torch.arange(D, dtype=torch.float32, device=pred_logits.device)

    # targets in 0-255; shift +1 to align with our 1-256 bin slots (slot 0 = PAD)
    target_f = (targets.float() + 1).unsqueeze(-1)           # [N, 1]
    gauss    = torch.exp(-alpha * (bins - target_f) ** 2 / (2 * tolerance ** 2))
    gauss[:, 0] = 0.0                                         # zero out PAD slot
    soft_tgt = gauss / (gauss.sum(-1, keepdim=True) + 1e-8)   # [N, D]

    log_pred = F.log_softmax(pred_logits, dim=-1)
    kl       = (soft_tgt * (torch.log(soft_tgt + 1e-8) - log_pred)).sum(-1)  # [N]

    return (kl * valid_mask.float()).sum() / valid_mask.float().sum()


# ══════════════════════════════════════════════════════════════════════════
# Transformer block
# ══════════════════════════════════════════════════════════════════════════

class ARBlock(nn.Module):
    """
    Pre-LN Transformer decoder block.
    use_cross_attn=False → blocks 0-1 (self-attn only, Text2CAD pattern)
    use_cross_attn=True  → blocks 2-7 (self-attn + cross-attn to z* memory)
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 dropout: float, use_cross_attn: bool):
        super().__init__()
        self.use_cross_attn = use_cross_attn

        self.norm1     = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                               batch_first=True)
        self.drop1     = nn.Dropout(dropout)

        if use_cross_attn:
            self.norm2      = nn.LayerNorm(d_model)
            self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                                    batch_first=True)
            self.drop2      = nn.Dropout(dropout)

        self.norm3 = nn.LayerNorm(d_model)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.drop3 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, memory: torch.Tensor,
                causal_mask: torch.Tensor) -> torch.Tensor:
        # Self-attention (causal)
        h = self.norm1(x)
        h, _ = self.self_attn(h, h, h, attn_mask=causal_mask,
                               is_causal=False)   # mask passed explicitly
        x = x + self.drop1(h)

        # Cross-attention to z* memory (no causal mask needed)
        if self.use_cross_attn:
            h = self.norm2(x)
            h, _ = self.cross_attn(h, memory, memory)
            x = x + self.drop2(h)

        # FFN
        h = self.norm3(x)
        x = x + self.drop3(self.ffn(h))
        return x


# ══════════════════════════════════════════════════════════════════════════
# Main decoder
# ══════════════════════════════════════════════════════════════════════════

class CADDecoderARV2(nn.Module):
    """
    8-block AR Transformer decoder for DeepCAD token sequences.

    Hyperparameters (defaults match Text2CAD decoder scale):
        latent_d   = 512    JEPA z* dimension
        d_model    = 256    CAD-token hidden dim
        n_heads    = 8
        n_layers   = 8      first 2 SA-only, last 6 SA+CA
        d_ff       = 1024
        n_mem      = 8      number of z* memory tokens for cross-attention
        d_arg_emb  = 32     per-arg embedding dim (16×32=512 → proj to d_model)
        dropout    = 0.1
    """

    def __init__(self, latent_d: int = 512, d_model: int = 256,
                 n_heads: int = 8, n_layers: int = 8, d_ff: int = 1024,
                 n_mem: int = 8, d_arg_emb: int = 32, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_mem   = n_mem

        # ── z* → n_mem memory tokens ──────────────────────────────────────
        # Using n_mem>1 prevents degenerate cross-attention (single-vector K/V)
        self.z_proj = nn.Sequential(
            nn.Linear(latent_d, n_mem * d_model),
            nn.LayerNorm(n_mem * d_model),
        )

        # ── Input embeddings ───────────────────────────────────────────────
        # Commands: indices 0-5 (real) + 6 (BOS)
        self.cmd_embed = nn.Embedding(N_COMMANDS + 1, d_model)

        # Args: values -1 (PAD) to 255.  Shift +1 → 0 (PAD) to 256.
        # Shared table, padded at index 0.
        self.arg_embed = nn.Embedding(ARGS_DIM + 1, d_arg_emb, padding_idx=0)
        self.arg_proj  = nn.Linear(N_ARGS * d_arg_emb, d_model, bias=False)

        # Learned absolute positional encoding
        self.pos_embed  = nn.Embedding(MAX_LEN + 2, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.input_drop = nn.Dropout(dropout)

        # ── Transformer blocks ─────────────────────────────────────────────
        self.blocks = nn.ModuleList([
            ARBlock(d_model, n_heads, d_ff, dropout,
                    use_cross_attn=(i >= 2))   # 0,1 → SA only; 2-7 → SA+CA
            for i in range(n_layers)
        ])
        self.out_norm = nn.LayerNorm(d_model)

        # ── Output heads ───────────────────────────────────────────────────
        self.cmd_head = nn.Linear(d_model, N_COMMANDS)

        # Command-guided args (Drawing2CAD): command softmax → d_model,
        # added to decoder hidden state before args head
        self.cmd_guidance = nn.Linear(N_COMMANDS, d_model)
        # 16 arg positions × 257 bins (256 values + PAD slot 0)
        self.args_head = nn.Linear(d_model, N_ARGS * (ARGS_DIM + 1))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
                if m.padding_idx is not None:
                    m.weight.data[m.padding_idx].zero_()

    # ── Internal helpers ───────────────────────────────────────────────────

    def _embed(self, cmds: torch.Tensor, args: torch.Tensor) -> torch.Tensor:
        """
        cmds : [B, T]      int, 0-6
        args : [B, T, 16]  int, -1 to 255
        → [B, T, d_model]
        """
        B, T = cmds.shape
        pos      = torch.arange(T, device=cmds.device).unsqueeze(0)
        cmd_emb  = self.cmd_embed(cmds)                              # [B,T,d]
        args_s   = (args + 1).clamp(min=0)                          # PAD→0, vals→1-256
        arg_emb  = self.arg_embed(args_s)                            # [B,T,16,d_arg]
        arg_emb  = self.arg_proj(arg_emb.view(B, T, -1))            # [B,T,d]
        x = self.input_drop(self.input_norm(cmd_emb + arg_emb + self.pos_embed(pos)))
        return x

    @staticmethod
    def _causal_mask(T: int, device: torch.device) -> torch.Tensor:
        """Boolean upper-triangular mask: True = position is masked out."""
        return torch.triu(torch.ones(T, T, dtype=torch.bool, device=device),
                          diagonal=1)

    # ── Forward pass (teacher forcing) ─────────────────────────────────────

    def forward(self, z_star: torch.Tensor,
                in_cmds: torch.Tensor,
                in_args: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        z_star  : [B, latent_d]
        in_cmds : [B, T]        shift-right sequence starting with BOS
        in_args : [B, T, 16]    corresponding args (PAD row prepended)

        Returns
        -------
        cmd_logits  : [B, T, N_COMMANDS]
        args_logits : [B, T, N_ARGS, ARGS_DIM+1]
        """
        B, T = in_cmds.shape
        mem  = self.z_proj(z_star).view(B, self.n_mem, self.d_model)  # [B, n_mem, d]
        x    = self._embed(in_cmds, in_args)                           # [B, T, d]
        mask = self._causal_mask(T, x.device)                          # [T, T] bool

        for blk in self.blocks:
            x = blk(x, mem, mask)
        x = self.out_norm(x)                                           # [B, T, d]

        cmd_logits = self.cmd_head(x)                                  # [B, T, 6]

        # Command-guided args head
        cmd_soft   = torch.softmax(cmd_logits.detach(), dim=-1)        # [B, T, 6]
        guided     = x + self.cmd_guidance(cmd_soft)                   # [B, T, d]
        args_raw   = self.args_head(guided)                            # [B, T, 16*257]
        args_logits = args_raw.view(B, T, N_ARGS, ARGS_DIM + 1)       # [B, T, 16, 257]

        return cmd_logits, args_logits

    # ── Loss ───────────────────────────────────────────────────────────────

    def compute_loss(self, z_star: torch.Tensor,
                     tgt_cmds: torch.Tensor,
                     tgt_args: torch.Tensor,
                     kl_tolerance: int = 3,
                     kl_alpha: float = 2.0) -> Dict[str, torch.Tensor]:
        """
        z_star   : [B, latent_d]
        tgt_cmds : [B, T]       ground-truth commands  (0-5, EOS for padding)
        tgt_args : [B, T, 16]   ground-truth args      (-1 for PAD/unused)

        Returns dict with keys: loss, cmd_loss, args_loss
        """
        B, T = tgt_cmds.shape
        device = tgt_cmds.device

        # Build teacher-forcing input: shift right, prepend BOS / PAD row
        bos_c   = torch.full((B, 1),       BOS_IDX, dtype=torch.long, device=device)
        bos_a   = torch.full((B, 1, N_ARGS), PAD_VAL, dtype=torch.long, device=device)
        in_cmds = torch.cat([bos_c, tgt_cmds[:, :-1]], dim=1)   # [B, T]
        in_args = torch.cat([bos_a, tgt_args[:, :-1]], dim=1)   # [B, T, 16]

        cmd_logits, args_logits = self.forward(z_star, in_cmds, in_args)

        # Command loss: all-position CE (Text2CAD / DeepCAD convention)
        cmd_loss = F.cross_entropy(
            cmd_logits.reshape(-1, N_COMMANDS),
            tgt_cmds.reshape(-1),
        )

        # Args loss: soft-target KL on non-PAD positions
        al_flat  = args_logits.reshape(B * T * N_ARGS, ARGS_DIM + 1)
        ta_flat  = tgt_args.reshape(B * T * N_ARGS)           # -1 to 255
        valid    = (ta_flat != PAD_VAL)                        # [B*T*N_ARGS] bool
        args_loss = soft_target_kl(al_flat, ta_flat.clamp(min=0), valid,
                                   tolerance=kl_tolerance, alpha=kl_alpha)

        return {
            'loss':      cmd_loss + args_loss,
            'cmd_loss':  cmd_loss.detach(),
            'args_loss': args_loss.detach(),
        }

    # ── Inference ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(self, z_star: torch.Tensor,
                 n_ext_targets: Optional[List[int]] = None,
                 max_len: int = MAX_LEN,
                 temperature: float = 0.8,
                 greedy: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Grammar-constrained autoregressive generation.

        z_star       : [B, latent_d]
        n_ext_targets: list of B ints from n_ext_head; None = no forced stop
        temperature  : sampling temperature (ignored when greedy=True)
        greedy       : argmax decoding

        Returns
        -------
        cmds : [B, max_len]       int64
        args : [B, max_len, 16]   int64  (PAD_VAL = -1 for unused slots)
        """
        B, device = z_star.shape[0], z_star.device

        grammars = [
            CADGrammar(n_ext_target=(n_ext_targets[b] if n_ext_targets else None))
            for b in range(B)
        ]

        # Running sequence (starts with BOS)
        seq_cmds = torch.full((B, 1), BOS_IDX,   dtype=torch.long, device=device)
        seq_args = torch.full((B, 1, N_ARGS), PAD_VAL, dtype=torch.long, device=device)

        out_cmds: List[torch.Tensor] = []
        out_args: List[torch.Tensor] = []
        done = [False] * B

        for _ in range(max_len):
            cmd_logits, args_logits = self.forward(z_star, seq_cmds, seq_args)
            last_cmd  = cmd_logits[:, -1, :]    # [B, 6]
            last_args = args_logits[:, -1, :, :] # [B, 16, 257]

            step_cmds = torch.empty(B, dtype=torch.long, device=device)
            step_args = torch.full((B, N_ARGS), PAD_VAL, dtype=torch.long, device=device)

            for b in range(B):
                if done[b]:
                    step_cmds[b] = EOS
                    continue

                # Grammar mask
                logits = last_cmd[b] + grammars[b].logit_mask(device)

                if greedy:
                    cmd = int(logits.argmax().item())
                else:
                    probs = F.softmax(logits / max(temperature, 1e-5), dim=-1)
                    cmd   = int(torch.multinomial(probs, 1).item())

                step_cmds[b] = cmd

                # Sample args for commands that use them
                if cmd not in (SOL, EOS):
                    for a in range(N_ARGS):
                        al = last_args[b, a]   # [257]
                        if greedy:
                            v = int(al.argmax().item())
                        else:
                            v = int(torch.multinomial(
                                F.softmax(al / max(temperature, 1e-5), dim=-1), 1
                            ).item())
                        step_args[b, a] = v - 1   # shift back: 0→-1(PAD), 1→0, 256→255

                grammars[b].step(cmd)
                if cmd == EOS:
                    done[b] = True

            out_cmds.append(step_cmds)
            out_args.append(step_args)

            seq_cmds = torch.cat([seq_cmds, step_cmds.unsqueeze(1)], dim=1)
            seq_args = torch.cat([seq_args, step_args.unsqueeze(1)], dim=1)

            if all(done):
                break

        # Stack and pad / trim to max_len
        gen_len   = len(out_cmds)
        cmds_out  = torch.stack(out_cmds, dim=1)                     # [B, gen_len]
        args_out  = torch.stack(out_args, dim=1)                     # [B, gen_len, 16]

        if gen_len < max_len:
            pad = max_len - gen_len
            cmds_out = F.pad(cmds_out, (0, pad),       value=EOS)
            args_out = F.pad(args_out, (0, 0, 0, pad), value=PAD_VAL)
        else:
            cmds_out = cmds_out[:, :max_len]
            args_out = args_out[:, :max_len]

        return cmds_out, args_out

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)