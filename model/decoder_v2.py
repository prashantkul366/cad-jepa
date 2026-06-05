"""
model/decoder_v2.py
CAD Sequence Decoder — Version 2

Key changes from V1:
  1. Multi-token memory: z* → [B, n_mem, d_model] (was [B, 1, d_model])
  2. Additive z* injection: z_add(z*) broadcast-added to every decoder input position
  3. Structural weighting in loss: upweights premature SOL in non-circle loops
  4. Label smoothing 0.05 (was 0.1)

These three changes together force the decoder to:
  (a) actually use z* at inference (not just during teacher forcing)
  (b) not collapse to single-curve loops
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Grammar constants ─────────────────────────────────────────────────────────
_CURVE_SET = {0, 1, 2}

_GRAMMAR_ALLOWED = {
    'need_sol'      : {4},
    'in_loop_empty' : {0, 1, 2},
    'in_loop_active': {0, 1, 2, 4, 5},
    'after_ext'     : {3, 4},
    'done'          : {3},
}

def _apply_grammar_mask(logits, state, n_curves_in_loop=0, n_curves_in_feature=0,
                        loop_has_circle=False, max_curves=4,
                        min_curves_noncircle=3, neg_inf=-1e9):
    allowed = set(_GRAMMAR_ALLOWED.get(state, set(range(6))))
    if state == 'in_loop_active':
        if n_curves_in_feature >= max_curves:
            allowed = {5}                          # feature full → force EXT
        elif loop_has_circle:
            pass                                    # circle closes itself, SOL/EXT ok
        elif n_curves_in_loop < min_curves_noncircle:
            allowed -= {4, 5}                       # too few curves → block SOL and EXT
    mask = torch.full_like(logits, neg_inf)
    for idx in allowed:
        mask[idx] = 0.0
    return logits + mask

def _grammar_transition(cmd, state, n_ext, n_curves_in_loop=0,
                        n_curves_in_feature=0, loop_has_circle=False,
                        max_ext=10, target_n_ext=None):
    if cmd == 4:   # SOL
        return 'in_loop_empty', n_ext, 0, n_curves_in_feature, False
    elif cmd == 2:  # CIRCLE — self-closing
        return 'in_loop_active', n_ext, n_curves_in_loop+1, n_curves_in_feature+1, True
    elif cmd in (0, 1):  # LINE, ARC
        return 'in_loop_active', n_ext, n_curves_in_loop+1, n_curves_in_feature+1, loop_has_circle
    elif cmd == 5:  # EXT
        n_ext += 1
        if target_n_ext is not None and n_ext >= target_n_ext:
            return 'done', n_ext, 0, 0, False
        return ('done' if n_ext >= max_ext else 'after_ext'), n_ext, 0, 0, False
    elif cmd == 3:  # EOS
        return 'done', n_ext, 0, 0, False
    return state, n_ext, n_curves_in_loop, n_curves_in_feature, loop_has_circle


# ── Structural weight helper ──────────────────────────────────────────────────

def _structural_weights(tgt_commands: torch.Tensor,
                        upweight: float = 5.0) -> torch.Tensor:
    """
    Returns per-position weights [B, L] for the command loss.

    Weight = upweight at any SOL position where the previous loop contained
    only LINE/ARC curves AND fewer than 2 of them (no CIRCLE).
    These are structurally degenerate transitions that cause the mode collapse.

    CIRCLE single-curve loops are valid and are NOT penalised.
    This loop runs on CPU (small seq lengths, negligible overhead).
    """
    B, L = tgt_commands.shape
    w = torch.ones(B, L, dtype=torch.float, device=tgt_commands.device)

    cmds_cpu = tgt_commands.cpu().tolist()
    for b in range(B):
        n_line_arc   = 0
        has_circle   = False
        first_sol    = True
        for t in range(L):
            cmd = cmds_cpu[b][t]
            if cmd == 3:   # EOS — stop
                break
            elif cmd == 4:   # SOL
                if not first_sol and n_line_arc < 2 and not has_circle:
                    w[b, t] = upweight   # premature SOL: upweight
                n_line_arc = 0
                has_circle = False
                first_sol  = False
            elif cmd == 2:   # CIRCLE
                has_circle = True
            elif cmd in (0, 1):   # LINE, ARC
                n_line_arc += 1
    return w


# ── Decoder ───────────────────────────────────────────────────────────────────

class CADSequenceDecoder(nn.Module):
    """
    V2 autoregressive decoder: z* [B,512] → CAD token sequences.

    Key design:
      z_to_mem  : z* → n_mem separate memory tokens for cross-attention
      z_add     : z* → d_model bias, added at every decoder input position
                  (ensures z* is present in gradient everywhere, not just via attention)
    """

    def __init__(
        self,
        latent_d   : int   = 512,
        d_model    : int   = 512,
        n_heads    : int   = 8,
        n_layers   : int   = 6,
        d_ff       : int   = 2048,
        n_mem      : int   = 8,     # multi-token memory size
        dropout    : float = 0.1,
        n_commands : int   = 6,
        args_dim   : int   = 256,
        n_args     : int   = 16,
        eos_idx    : int   = 3,
        max_len    : int   = 60,
    ):
        super().__init__()
        self.n_commands = n_commands
        self.args_dim   = args_dim
        self.n_args     = n_args
        self.eos_idx    = eos_idx
        self.max_len    = max_len
        self.n_mem      = n_mem

        # ── z* conditioning ───────────────────────────────────────────────────
        # Multi-token memory: decompose z* into n_mem query slots
        self.z_to_mem = nn.Linear(latent_d, n_mem * d_model)
        # Additive injection: broadcast z* bias to every decoder input position
        self.z_add    = nn.Linear(latent_d, d_model)

        # ── Token embeddings ──────────────────────────────────────────────────
        self.bos_embed     = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.command_embed = nn.Embedding(n_commands, d_model)
        self.arg_embed     = nn.Embedding(args_dim + 1, 64, padding_idx=0)
        self.embed_proj    = nn.Linear(64 * n_args, d_model)

        # ── Transformer decoder ───────────────────────────────────────────────
        layer = nn.TransformerDecoderLayer(
            d_model         = d_model,
            nhead           = n_heads,
            dim_feedforward = d_ff,
            dropout         = dropout,
            batch_first     = True,
            norm_first      = True,
        )
        self.decoder     = nn.TransformerDecoder(layer, num_layers=n_layers)
        self.output_norm = nn.LayerNorm(d_model)

        # ── Output heads ──────────────────────────────────────────────────────
        self.command_head = nn.Linear(d_model, n_commands)
        self.args_head    = nn.Linear(d_model, n_args * (args_dim + 1))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding) and m.padding_idx is not None:
                nn.init.normal_(m.weight, std=0.02)
                nn.init.zeros_(m.weight[m.padding_idx])
        # Initialise z_add small so it starts as a weak perturbation
        nn.init.normal_(self.z_add.weight, std=0.01)
        nn.init.zeros_(self.z_add.bias)

    def _embed_tokens(self, commands, args):
        """[B, L] + [B, L, n_args] → [B, L, d_model]"""
        B, L   = commands.shape
        c_emb  = self.command_embed(commands)
        a_idx  = (args + 1).clamp(min=0)
        a_emb  = self.arg_embed(a_idx).reshape(B, L, -1)
        return c_emb + self.embed_proj(a_emb)

    def _build_memory(self, z_star):
        """z* [B, latent_d] → memory [B, n_mem, d_model]"""
        B = z_star.size(0)
        return self.z_to_mem(z_star).reshape(B, self.n_mem, -1)

    def _build_dec_input(self, B, z_star, commands=None, args=None):
        """
        Build decoder input with z* additive bias.
        Returns [B, L+1, d_model] (BOS prepended, z_bias added to all positions).
        """
        bos = self.bos_embed.expand(B, 1, -1)
        if commands is None or commands.size(1) == 0:
            seq = bos
        else:
            seq = torch.cat([bos, self._embed_tokens(commands, args)], dim=1)

        # Additive z* bias broadcast over all positions — key V2 change
        z_bias = self.z_add(z_star).unsqueeze(1)   # [B, 1, d_model]
        return seq + z_bias

    # ── Forward (teacher forcing) ──────────────────────────────────────────────

    def forward(self, z_star, tgt_commands, tgt_args):
        B, L   = tgt_commands.shape
        memory = self._build_memory(z_star)                # [B, n_mem, d_model]
        dec_in = self._build_dec_input(
            B, z_star,
            tgt_commands[:, :-1],
            tgt_args[:, :-1],
        )                                                   # [B, L, d_model]
        causal = nn.Transformer.generate_square_subsequent_mask(L, device=z_star.device)
        out    = self.output_norm(self.decoder(dec_in, memory, tgt_mask=causal))
        cmd_l  = self.command_head(out)                    # [B, L, n_commands]
        arg_l  = self.args_head(out).reshape(B, L, self.n_args, self.args_dim + 1)
        return cmd_l, arg_l

    # ── Beam search (inference) ───────────────────────────────────────────────

    @torch.no_grad()
    def generate(self, z_star, max_len=None, beam_k=5, temperature=0.8,
                 len_penalty=0.6, return_all=False, n_ext_head=None):
        assert z_star.size(0) == 1
        if max_len is None:
            max_len = self.max_len
        device = z_star.device
        EOS    = self.eos_idx

        target_n_ext = None
        if n_ext_head is not None:
            target_n_ext = max(1, round(n_ext_head(z_star).item()))

        memory = self._build_memory(z_star)   # [1, n_mem, d_model]

        beams = [{
            'lp': 0.0, 'cmds': [], 'args': [], 'done': False,
            'state': 'need_sol', 'n_ext': 0,
            'n_curves_in_loop': 0, 'n_curves_in_feature': 0,
            'loop_has_circle': False,
        }]

        for step in range(max_len):
            if all(b['done'] for b in beams):
                break
            active = [b for b in beams if not b['done']]
            frozen = [b for b in beams if b['done']]
            n_act  = len(active)

            mem_exp = memory.expand(n_act, -1, -1)

            if step == 0:
                dec_in = self._build_dec_input(n_act, z_star.expand(n_act, -1))
            else:
                cmds_t = torch.tensor([b['cmds'] for b in active], dtype=torch.long, device=device)
                args_t = torch.tensor([b['args'] for b in active], dtype=torch.long, device=device)
                dec_in = self._build_dec_input(n_act, z_star.expand(n_act, -1), cmds_t, args_t)

            seq_len = dec_in.size(1)
            causal  = nn.Transformer.generate_square_subsequent_mask(seq_len, device=device)
            out     = self.output_norm(self.decoder(dec_in, mem_exp, tgt_mask=causal))
            cmd_l   = self.command_head(out[:, -1, :])
            arg_l   = self.args_head(out[:, -1, :]).reshape(n_act, self.n_args, self.args_dim + 1)

            candidates = []
            for i, beam in enumerate(active):
                masked = _apply_grammar_mask(
                    cmd_l[i], beam['state'],
                    beam['n_curves_in_loop'], beam['n_curves_in_feature'],
                    beam['loop_has_circle'],
                )
                lps, top_cmds = torch.topk(
                    F.log_softmax(masked / temperature, dim=-1),
                    min(beam_k, self.n_commands)
                )
                arg_pred = arg_l[i].argmax(dim=-1) - 1

                for lp, cmd in zip(lps.tolist(), top_cmds.tolist()):
                    ns, ne, ncl, ncf, nhc = _grammar_transition(
                        cmd, beam['state'], beam['n_ext'],
                        beam['n_curves_in_loop'], beam['n_curves_in_feature'],
                        beam['loop_has_circle'], target_n_ext=target_n_ext,
                    )
                    candidates.append({
                        'lp': beam['lp'] + lp,
                        'cmds': beam['cmds'] + [cmd],
                        'args': beam['args'] + [arg_pred.tolist()],
                        'done': (ns == 'done') or (step == max_len - 1),
                        'state': ns, 'n_ext': ne,
                        'n_curves_in_loop': ncl, 'n_curves_in_feature': ncf,
                        'loop_has_circle': nhc,
                    })

            all_cands = sorted(
                candidates + frozen,
                key=lambda b: b['lp'] / max(len(b['cmds']), 1) ** len_penalty,
                reverse=True,
            )
            beams = all_cands[:beam_k]

        for beam in beams:
            n = len(beam['cmds'])
            if n < max_len:
                beam['cmds'] += [EOS] * (max_len - n)
                beam['args'] += [[-1] * self.n_args] * (max_len - n)

        def to_tensors(b):
            return (
                torch.tensor(b['cmds'][:max_len], dtype=torch.long, device=device),
                torch.tensor(b['args'][:max_len], dtype=torch.long, device=device),
            )

        return [to_tensors(b) for b in beams] if return_all else to_tensors(beams[0])

    def param_summary(self):
        n = sum(p.numel() for p in self.parameters())
        print(f"CADSequenceDecoder V2 — {n:,} parameters")
        for name, m in [
            ('z_to_mem',    self.z_to_mem),
            ('z_add',       self.z_add),
            ('bos+embeds',  [self.bos_embed, self.command_embed, self.arg_embed, self.embed_proj]),
            ('decoder(6L)', self.decoder),
            ('heads',       [self.command_head, self.args_head]),
        ]:
            if isinstance(m, list):
                cnt = sum(p.numel() for item in m
                          for p in (item.parameters() if isinstance(item, nn.Module) else [item]))
            else:
                cnt = sum(p.numel() for p in m.parameters())
            print(f"  {name:<18}: {cnt:>10,}")


# ── Loss ──────────────────────────────────────────────────────────────────────

def decoder_loss(cmd_logits, args_logits, tgt_commands, tgt_args,
                 eos_idx=3, label_smooth=0.05):
    """
    Structurally-weighted CE loss.
    
    Command loss uses _structural_weights to upweight (×5) any SOL position
    that follows a degenerate loop (< 2 LINE/ARC curves, no CIRCLE).
    This directly penalises the mode-collapse pattern.
    
    Label smoothing reduced to 0.05 (from V1's 0.1) for tighter commitment.
    """
    B, L = tgt_commands.shape

    # Structural weights — CPU loop, negligible overhead (~1ms per batch)
    sw = _structural_weights(tgt_commands).to(cmd_logits.device)   # [B, L]

    ce_cmd = F.cross_entropy(
        cmd_logits.reshape(-1, cmd_logits.size(-1)),
        tgt_commands.reshape(-1),
        reduction='none',
        label_smoothing=label_smooth,
    )                                                               # [B*L]
    cmd_loss = (ce_cmd * sw.reshape(-1)).sum() / sw.reshape(-1).sum()

    args_target = (tgt_args + 1).clamp(min=0)
    args_loss   = F.cross_entropy(
        args_logits.reshape(-1, args_logits.size(-1)),
        args_target.reshape(-1),
        ignore_index=0,
        label_smoothing=label_smooth,
    )

    loss = cmd_loss + args_loss

    with torch.no_grad():
        non_eos  = (tgt_commands != eos_idx)
        cmd_acc  = (cmd_logits.argmax(-1)[non_eos] == tgt_commands[non_eos]).float().mean()
        non_pad  = (tgt_args != -1)
        args_acc = ((args_logits.argmax(-1) - 1)[non_pad] == tgt_args[non_pad]).float().mean()

    return loss, {
        'loss_total': loss.item(), 'loss_cmd': cmd_loss.item(),
        'loss_args': args_loss.item(), 'acc_cmd': cmd_acc.item(), 'acc_args': args_acc.item(),
    }