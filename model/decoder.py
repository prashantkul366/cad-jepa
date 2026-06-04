"""
model/decoder.py
Stage 3 — CAD Sequence Decoder

Autoregressive transformer decoder conditioned on a latent z*.
Maps z* [B, 512] → CAD token sequences → CadQuery → .STEP

Architecture:
    z* → z_proj → [B, 1, d_model]        (cross-attention memory)
    BOS + tgt_embed[:-1] → 6-layer causal decoder → output
    → command_head [B, L, 6]
    → args_head    [B, L, 16, 257]

CRITICAL design notes:
    - Trained with z_target from CACHED Stage 1 latents (NOT Stage 2 bridge output)
      This decouples Stage 2 and Stage 3 and prevents error accumulation.
    - At inference: chain text → bridge → z* → decoder.generate()
    - BOS is a learnable embedding (not tied to any command index)
    - args_dim+1 = 257 classes: index 0 = PAD (arg=-1), indices 1-256 = values 0-255
    - Command loss: ignore_index=EOS_IDX (padding positions not penalised)
    - Args   loss: ignore_index=0        (PAD args not penalised)

Param count (defaults):
    z_proj + embeds  :   ~1M
    6-layer decoder  :  ~25M
    output heads     :   ~2M
    Total            :  ~28M
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# CADSequenceDecoder
# ──────────────────────────────────────────────────────────────────────────────

class CADSequenceDecoder(nn.Module):
    """
    Conditioned autoregressive decoder for CAD token sequences.

    Args:
        latent_d   : Dimension of input latent z* — must match Stage 1 d_model (512)
        d_model    : Internal decoder width (512)
        n_heads    : Attention heads (8)
        n_layers   : Decoder layers (6)
        d_ff       : FFN hidden width (2048)
        dropout    : Dropout probability (0.1)
        n_commands : Command vocabulary size — matches len(ALL_COMMANDS) (6)
        args_dim   : Quantization bins (256) — args values are 0-255
        n_args     : Parameters per operation (16)
        eos_idx    : EOS command index (3) — used in generate() stopping criterion
        max_len    : Maximum sequence length (60) — MAX_TOTAL_LEN from macro.py

    Usage (training):
        decoder = CADSequenceDecoder().cuda()
        cmd_logits, args_logits = decoder(z_star, tgt_commands, tgt_args)
        loss = decoder_loss(cmd_logits, args_logits, tgt_commands, tgt_args)

    Usage (inference):
        cmds, args = decoder.generate(z_star, beam_k=5)
    """

    def __init__(
        self,
        latent_d   : int   = 512,
        d_model    : int   = 512,
        n_heads    : int   = 8,
        n_layers   : int   = 6,
        d_ff       : int   = 2048,
        dropout    : float = 0.1,
        n_commands : int   = 6,
        args_dim   : int   = 256,
        n_args     : int   = 16,
        eos_idx    : int   = 3,
        max_len    : int   = 60,
    ):
        super().__init__()

        # Store for use in forward / generate
        self.n_commands = n_commands
        self.args_dim   = args_dim
        self.n_args     = n_args
        self.eos_idx    = eos_idx
        self.max_len    = max_len

        # ── Latent conditioning ───────────────────────────────────────────────
        self.z_proj = nn.Linear(latent_d, d_model)

        # ── BOS: learnable start-of-sequence embedding ────────────────────────
        self.bos_embed = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # ── Token embeddings (same scheme as CADJEPAEncoder) ──────────────────
        self.command_embed = nn.Embedding(n_commands, d_model)
        # args: args_dim+1 classes (0=PAD from arg=-1, 1-256 = values 0-255)
        self.arg_embed = nn.Embedding(args_dim + 1, 64, padding_idx=0)
        # Project concatenated arg embeddings to d_model
        self.embed_proj = nn.Linear(64 * n_args, d_model)

        # ── Causal transformer decoder (Pre-LN) ───────────────────────────────
        decoder_layer = nn.TransformerDecoderLayer(
            d_model         = d_model,
            nhead           = n_heads,
            dim_feedforward = d_ff,
            dropout         = dropout,
            batch_first     = True,
            norm_first      = True,    # Pre-LN — more stable
        )
        self.decoder     = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.output_norm = nn.LayerNorm(d_model)

        # ── Output heads ──────────────────────────────────────────────────────
        self.command_head = nn.Linear(d_model, n_commands)
        self.args_head    = nn.Linear(d_model, n_args * (args_dim + 1))

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier uniform for linear layers, normal for embeddings."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding) and m.padding_idx is not None:
                nn.init.normal_(m.weight, std=0.02)
                nn.init.zeros_(m.weight[m.padding_idx])

    # ── Embedding helpers ─────────────────────────────────────────────────────

    def _embed_tokens(
        self,
        commands : torch.Tensor,   # [B, L]  long
        args     : torch.Tensor,   # [B, L, n_args]  long, -1 for PAD
    ) -> torch.Tensor:             # [B, L, d_model]
        """Embed command type + args into d_model space."""
        B, L = commands.shape

        cmd_emb = self.command_embed(commands)              # [B, L, d_model]

        # Shift args: -1 → 0 (PAD, zeroed by padding_idx), 0-255 → 1-256
        arg_idx = (args + 1).clamp(min=0)                   # [B, L, n_args]
        arg_emb = self.arg_embed(arg_idx)                   # [B, L, n_args, 64]
        arg_emb = arg_emb.reshape(B, L, -1)                 # [B, L, n_args*64]
        arg_emb = self.embed_proj(arg_emb)                  # [B, L, d_model]

        return cmd_emb + arg_emb                            # [B, L, d_model]

    def _build_dec_input(
        self,
        B        : int,
        commands : torch.Tensor | None,   # [B, L] or None
        args     : torch.Tensor | None,   # [B, L, n_args] or None
    ) -> torch.Tensor:                    # [B, L+1, d_model]  (BOS prepended)
        """
        Build decoder input by prepending the learnable BOS embedding.
        When commands=None, returns just BOS (length 1).
        """
        bos = self.bos_embed.expand(B, 1, -1)       # [B, 1, d_model]
        if commands is None or commands.size(1) == 0:
            return bos
        token_emb = self._embed_tokens(commands, args)  # [B, L, d_model]
        return torch.cat([bos, token_emb], dim=1)        # [B, L+1, d_model]

    # ── Forward (teacher forcing — training) ──────────────────────────────────

    def forward(
        self,
        z_star       : torch.Tensor,   # [B, latent_d]
        tgt_commands : torch.Tensor,   # [B, L]         — ground truth command indices
        tgt_args     : torch.Tensor,   # [B, L, n_args] — ground truth args (-1 for PAD)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Teacher forcing forward pass.

        Decoder input  : BOS + tgt_commands[:-1]   (length L)
        Decoder targets: tgt_commands               (length L)

        Returns:
            cmd_logits  : [B, L, n_commands]         — command class logits
            args_logits : [B, L, n_args, args_dim+1] — per-arg class logits
        """
        B, L = tgt_commands.shape

        # Memory: z* as single-token cross-attention source
        memory = self.z_proj(z_star).unsqueeze(1)       # [B, 1, d_model]

        # Decoder input: BOS prepended, last ground-truth token dropped
        dec_input = self._build_dec_input(
            B,
            tgt_commands[:, :-1],    # [B, L-1]
            tgt_args[:, :-1],        # [B, L-1, n_args]
        )                            # → [B, L, d_model]

        # Causal mask [L, L]: upper triangle = -inf, diagonal+lower = 0
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            L, device=z_star.device
        )

        # Decode
        out = self.decoder(
            dec_input, memory,
            tgt_mask  = causal_mask,
        )                                               # [B, L, d_model]
        out = self.output_norm(out)                     # [B, L, d_model]

        cmd_logits  = self.command_head(out)            # [B, L, n_commands]
        args_logits = self.args_head(out).reshape(
            B, L, self.n_args, self.args_dim + 1
        )                                               # [B, L, n_args, args_dim+1]

        return cmd_logits, args_logits

    # ── Beam search (inference) ───────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        z_star      : torch.Tensor,   # [1, latent_d]
        max_len     : int   = None,
        beam_k      : int   = 5,
        temperature : float = 0.8,
        len_penalty : float = 0.6,    # length normalization exponent (0 = no norm)
        return_all  : bool  = False,  # if True, return all beam_k beams
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Beam search over command types; greedy decoding for args.

        Args:
            z_star     : [1, latent_d] — single latent vector
            max_len    : maximum tokens to generate (default: self.max_len)
            beam_k     : beam width
            temperature: softmax temperature for command distribution
            len_penalty: length normalisation exponent α in score / len^α
                         0 = no normalization, 0.6 = Google beam search default
            return_all : if True, return list of all beam_k sequences

        Returns (default):
            commands: [max_len]      int tensor — best beam command sequence
            args    : [max_len, n_args] int tensor — best beam args sequence

        Returns (return_all=True):
            list of (commands, args) tuples, best-first

        Inference pipeline:
            z* = bridge.encode_text(["a bracket with a hole"], device)
            cmds, args = decoder.generate(z*)
            # fallback: try beam 1, 2, 3 if beam 0 produces invalid solid
            all_beams = decoder.generate(z*, return_all=True)
            for cmds, args in all_beams:
                if cadquery_executes(cmds, args):
                    break
        """
        assert z_star.size(0) == 1, "generate() requires B=1; loop externally for batch"

        if max_len is None:
            max_len = self.max_len

        device = z_star.device
        EOS    = self.eos_idx

        # Precompute memory (shared across all beams)
        memory = self.z_proj(z_star).unsqueeze(1)   # [1, 1, d_model]

        # ── Beam state ────────────────────────────────────────────────────────
        # Each beam: dict with cumulative log-prob, predicted cmd/arg lists, done flag
        beams = [{'lp': 0.0, 'cmds': [], 'args': [], 'done': False}]

        for step in range(max_len):
            if all(b['done'] for b in beams):
                break

            active = [b for b in beams if not b['done']]
            frozen = [b for b in beams if b['done']]
            n_act  = len(active)

            # ── Build decoder input for all active beams ──────────────────────
            mem_exp = memory.expand(n_act, -1, -1)   # [n_act, 1, d_model]

            if step == 0:
                dec_input = self.bos_embed.expand(n_act, 1, -1)  # [n_act, 1, d_model]
            else:
                cmds_t = torch.tensor(
                    [b['cmds'] for b in active],
                    dtype=torch.long, device=device,
                )   # [n_act, step]
                args_t = torch.tensor(
                    [b['args'] for b in active],
                    dtype=torch.long, device=device,
                )   # [n_act, step, n_args]
                dec_input = self._build_dec_input(n_act, cmds_t, args_t)  # [n_act, step+1, d_model]

            # ── Run decoder, get logits at last position ───────────────────────
            seq_len     = dec_input.size(1)
            causal_mask = nn.Transformer.generate_square_subsequent_mask(
                seq_len, device=device
            )
            out = self.decoder(dec_input, mem_exp, tgt_mask=causal_mask)
            out = self.output_norm(out)

            # Logits at last position only
            cmd_logits  = self.command_head(out[:, -1, :])                          # [n_act, n_cmds]
            args_logits = self.args_head(out[:, -1, :]).reshape(
                n_act, self.n_args, self.args_dim + 1
            )                                                                         # [n_act, n_args, 257]

            # ── Expand each active beam by top-k commands ─────────────────────
            candidates = []

            for i, beam in enumerate(active):
                log_probs = F.log_softmax(cmd_logits[i] / temperature, dim=-1)
                top_lps, top_cmds = torch.topk(log_probs, min(beam_k, self.n_commands))

                # Greedy args: argmax per arg, shift back from embedding space
                arg_pred = args_logits[i].argmax(dim=-1) - 1   # [n_args], -1=PAD, 0-255=values

                for lp, cmd in zip(top_lps.tolist(), top_cmds.tolist()):
                    done = (cmd == EOS) or (step == max_len - 1)
                    candidates.append({
                        'lp'  : beam['lp'] + lp,
                        'cmds': beam['cmds'] + [cmd],
                        'args': beam['args'] + [arg_pred.tolist()],
                        'done': done,
                    })

            # Merge with already-done beams and prune to top-k
            all_cands = candidates + frozen
            all_cands.sort(
                key=lambda b: b['lp'] / max(len(b['cmds']), 1) ** len_penalty,
                reverse=True,
            )
            beams = all_cands[:beam_k]

        # ── Pad all beams to max_len ───────────────────────────────────────────
        for beam in beams:
            n = len(beam['cmds'])
            if n < max_len:
                beam['cmds'] += [EOS] * (max_len - n)
                beam['args'] += [[-1] * self.n_args] * (max_len - n)

        def to_tensors(beam):
            return (
                torch.tensor(beam['cmds'][:max_len], dtype=torch.long, device=device),
                torch.tensor(beam['args'][:max_len], dtype=torch.long, device=device),
            )

        if return_all:
            return [to_tensors(b) for b in beams]

        return to_tensors(beams[0])

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def param_summary(self) -> None:
        total = sum(p.numel() for p in self.parameters())
        print("=" * 50)
        print("CADSequenceDecoder — parameter summary")
        print("=" * 50)
        for name, m in [
            ('z_proj + bos',   [self.z_proj, self.bos_embed]),
            ('command_embed',  [self.command_embed]),
            ('arg_embed',      [self.arg_embed]),
            ('embed_proj',     [self.embed_proj]),
            ('decoder (6L)',   [self.decoder, self.output_norm]),
            ('command_head',   [self.command_head]),
            ('args_head',      [self.args_head]),
        ]:
            n = sum(
                p.numel()
                for item in m
                for p in (item.parameters() if isinstance(item, nn.Module)
                          else [item])
            )
            print(f"  {name:<18}: {n:>10,}")
        print("-" * 50)
        print(f"  {'Total':<18}: {total:>10,}")
        print("=" * 50)


# ──────────────────────────────────────────────────────────────────────────────
# Loss
# ──────────────────────────────────────────────────────────────────────────────

def decoder_loss(
    cmd_logits   : torch.Tensor,   # [B, L, n_commands]
    args_logits  : torch.Tensor,   # [B, L, n_args, args_dim+1]
    tgt_commands : torch.Tensor,   # [B, L]           long
    tgt_args     : torch.Tensor,   # [B, L, n_args]   long, -1 for PAD
    eos_idx      : int   = 3,
    label_smooth : float = 0.1,
) -> tuple[torch.Tensor, dict]:
    """
    Combined command + args cross-entropy loss.

    Command loss : ignore EOS-padded positions (ignore_index=EOS_IDX)
    Args loss    : ignore PAD args (ignore_index=0 after +1 shift)

    Returns:
        loss    : scalar
        metrics : dict with individual components for logging
    """
    B, L = tgt_commands.shape

    # ── Command loss ──────────────────────────────────────────────────────────
    # cmd_loss = F.cross_entropy(
    #     cmd_logits.reshape(-1, cmd_logits.size(-1)),
    #     tgt_commands.reshape(-1),
    #     ignore_index    = eos_idx,
    #     label_smoothing = label_smooth,
    # )
    # is_eos   = (tgt_commands == eos_idx)               # [B, L]
    # cum_eos  = is_eos.long().cumsum(dim=1)             # [B, L]
    # cmd_mask = (cum_eos <= 1)                          # [B, L] True = compute loss

    # ce_all   = F.cross_entropy(
    #     cmd_logits.reshape(-1, cmd_logits.size(-1)),
    #     tgt_commands.reshape(-1),
    #     reduction       = 'none',
    #     label_smoothing = label_smooth,
    # )                                                  # [B*L]
    # denom    = cmd_mask.float().sum().clamp(min=1)
    # cmd_loss = (ce_all * cmd_mask.reshape(-1).float()).sum() / denom

    is_eos      = (tgt_commands == eos_idx)            # [B, L]
    cum_eos     = is_eos.long().cumsum(dim=1)          # [B, L]
    cmd_mask    = (cum_eos <= 1)                       # real tokens + first EOS
    first_eos   = is_eos & (cum_eos == 1)             # exactly first EOS position

    # Upweight first-EOS 5x: 1 EOS position vs ~14 real positions → balanced signal
    pos_w    = torch.ones_like(tgt_commands, dtype=torch.float)
    pos_w[first_eos] = 5.0

    ce_all   = F.cross_entropy(
        cmd_logits.reshape(-1, cmd_logits.size(-1)),
        tgt_commands.reshape(-1),
        reduction       = 'none',
        label_smoothing = label_smooth,
    )                                                  # [B*L]
    denom    = (cmd_mask.float() * pos_w).sum().clamp(min=1)
    cmd_loss = (ce_all * cmd_mask.reshape(-1).float() * pos_w.reshape(-1)).sum() / denom

    # ── Args loss ─────────────────────────────────────────────────────────────
    # Shift args: -1 → 0 (PAD, ignored), 0-255 → 1-256
    args_target = (tgt_args + 1).clamp(min=0)     # [B, L, n_args], PAD=0, vals=1-256

    args_loss = F.cross_entropy(
        args_logits.reshape(-1, args_logits.size(-1)),
        args_target.reshape(-1),
        ignore_index    = 0,                       # ignore PAD class
        label_smoothing = label_smooth,
    )

    loss = cmd_loss + args_loss

    # ── Accuracy (non-EOS, non-PAD positions) ─────────────────────────────────
    with torch.no_grad():
        cmd_mask    = (tgt_commands != eos_idx)
        cmd_acc     = (cmd_logits.argmax(-1)[cmd_mask] == tgt_commands[cmd_mask]).float().mean()

        args_mask   = (tgt_args != -1)
        args_pred   = args_logits.argmax(-1) - 1                        # shift back
        args_acc    = (args_pred[args_mask] == tgt_args[args_mask]).float().mean()

    metrics = {
        'loss_total' : loss.item(),
        'loss_cmd'   : cmd_loss.item(),
        'loss_args'  : args_loss.item(),
        'acc_cmd'    : cmd_acc.item(),
        'acc_args'   : args_acc.item(),
    }

    return loss, metrics


# ──────────────────────────────────────────────────────────────────────────────
# Inference helper
# ──────────────────────────────────────────────────────────────────────────────

def load_decoder_for_inference(
    ckpt_path : str,
    device    : torch.device,
) -> CADSequenceDecoder:
    """
    Load Stage 3 decoder checkpoint for inference pipeline.

    Usage:
        decoder = load_decoder_for_inference('decoder/best.pt', device)
        cmds, args = decoder.generate(z_star)
    """
    ckpt    = torch.load(ckpt_path, map_location=device)
    cfg     = ckpt.get('cfg', {})
    decoder = CADSequenceDecoder(
        latent_d   = cfg.get('latent_d',   512),
        d_model    = cfg.get('d_model',    512),
        n_heads    = cfg.get('n_heads',    8),
        n_layers   = cfg.get('n_layers',   6),
        d_ff       = cfg.get('d_ff',       2048),
        n_commands = cfg.get('n_commands', 6),
        args_dim   = cfg.get('args_dim',   256),
        n_args     = cfg.get('n_args',     16),
        eos_idx    = cfg.get('eos_idx',    3),
        max_len    = cfg.get('max_len',    60),
    )
    decoder.load_state_dict(ckpt['decoder'])
    decoder.to(device).eval()
    print(f"[Decoder] Loaded: epoch={ckpt['epoch']}  "
          f"cmd_acc={ckpt.get('val_cmd_acc', 0):.3f}  "
          f"args_acc={ckpt.get('val_args_acc', 0):.3f}")
    return decoder


# ──────────────────────────────────────────────────────────────────────────────
# Smoke test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    decoder = CADSequenceDecoder().to(device)
    decoder.param_summary()

    B, L = 4, 60
    z_star       = torch.randn(B, 512, device=device)
    tgt_commands = torch.randint(0, 6, (B, L), device=device)
    tgt_args     = torch.randint(-1, 256, (B, L, 16), device=device)

    print("\n[1/4] Teacher forcing forward pass...")
    cmd_logits, args_logits = decoder(z_star, tgt_commands, tgt_args)
    print(f"  cmd_logits  : {cmd_logits.shape}")   # [4, 60, 6]
    print(f"  args_logits : {args_logits.shape}")  # [4, 60, 16, 257]
    assert cmd_logits.shape  == (B, L, 6)
    assert args_logits.shape == (B, L, 16, 257)

    print("\n[2/4] Loss computation...")
    loss, metrics = decoder_loss(cmd_logits, args_logits, tgt_commands, tgt_args)
    loss.backward()
    print(f"  loss_total  : {metrics['loss_total']:.4f}")
    print(f"  loss_cmd    : {metrics['loss_cmd']:.4f}")
    print(f"  loss_args   : {metrics['loss_args']:.4f}")
    print(f"  acc_cmd     : {metrics['acc_cmd']:.4f}")
    print(f"  acc_args    : {metrics['acc_args']:.4f}")
    print(f"  backward    : OK")

    print("\n[3/4] Beam search generate (B=1, beam_k=3)...")
    decoder.eval()
    z1 = torch.randn(1, 512, device=device)
    with torch.no_grad():
        cmds, args = decoder.generate(z1, max_len=60, beam_k=3, temperature=0.8)
    print(f"  cmds shape  : {cmds.shape}")    # [60]
    print(f"  args shape  : {args.shape}")    # [60, 16]
    print(f"  first 8 cmds: {cmds[:8].tolist()}")
    assert cmds.shape == (60,)
    assert args.shape == (60, 16)

    print("\n[4/4] return_all=True (all beams)...")
    all_beams = decoder.generate(z1, beam_k=3, return_all=True)
    print(f"  n_beams     : {len(all_beams)}")
    for i, (c, a) in enumerate(all_beams):
        print(f"  beam {i}: cmds={c[:5].tolist()}  args[0]={a[0, :4].tolist()}")

    print("\nAll checks passed.")