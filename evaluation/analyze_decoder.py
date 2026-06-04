"""
evaluation/analyze_decoder.py

Comprehensive validation of the Stage 3 CADSequenceDecoder.
Seven modules — no plots, all printed metrics + saved JSON.

Modules:
    A. teacher_forcing      verify cmd_acc / args_acc match training log
    B. autoregressive       generate from GT z, check structural validity
    C. command_distribution compare predicted vs GT command type distributions
    D. args_quality         per-command-type args accuracy + L1 error
    E. z_conditioning       decoder must vary output with z, not generate blind
    F. distribution_shift   check bridge z* lands in correct region for decoder
    G. chain_1_2_3          full text → bridge → correction → decoder pipeline

Usage:
    python evaluation/analyze_decoder.py
    python evaluation/analyze_decoder.py --n-samples 500 --beam-k 1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config.configJEPA import ConfigJEPA
from dataset.cad_dataset import CADDataset
from model.decoder import CADSequenceDecoder
from model.text_bridge import TextToLatentBridge
from dataset.text_latent_dataset import TextLatentDataset, _collate_text_latent

# ── Command index constants ───────────────────────────────────────────────────
LINE_IDX   = 0
ARC_IDX    = 1
CIRCLE_IDX = 2
EOS_IDX    = 3
SOL_IDX    = 4
EXT_IDX    = 5
CMD_NAMES  = {0:'LINE', 1:'ARC', 2:'CIRCLE', 3:'EOS', 4:'SOL', 5:'EXT'}


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class AnalysisConfig:
    decoder_ckpt : str = '/content/drive/MyDrive/cad-jepa-checkpoints/decoder/best.pt'
    bridge_ckpt  : str = '/content/drive/MyDrive/cad-jepa-checkpoints/bridge/best.pt'
    correction_pt: str = '/content/drive/MyDrive/cad-jepa-checkpoints/bridge/correction.pt'
    cache_val    : str = '/content/drive/MyDrive/cad-jepa-data/latent_cache_validation.npy'
    cache_train  : str = '/content/drive/MyDrive/cad-jepa-data/latent_cache_train.npy'
    annotations  : str = '/content/drive/MyDrive/cad-jepa-data/text2cad_annotations.json'
    split_path   : str = '/content/train_val_test_split.json'
    data_root    : str = '/content'
    output_dir   : str = '/content/drive/MyDrive/cad-jepa-analysis'
    n_samples    : int = 300     # val samples to analyse
    beam_k       : int = 1       # 1=greedy for speed, 5 for quality
    batch_size   : int = 64
    seed         : int = 42


# ──────────────────────────────────────────────────────────────────────────────
# Analyzer
# ──────────────────────────────────────────────────────────────────────────────

class DecoderAnalyzer:

    def __init__(self, cfg: AnalysisConfig):
        self.cfg    = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(cfg.seed)

        print(f"Device: {self.device}")

        # ── Decoder ──────────────────────────────────────────────────────────
        print(f"\nLoading decoder: {cfg.decoder_ckpt}")
        dec_ckpt      = torch.load(cfg.decoder_ckpt, map_location=self.device)
        self.dec_epoch = dec_ckpt['epoch']
        dec_cfg        = dec_ckpt.get('cfg', {})
        self.decoder   = CADSequenceDecoder(
            latent_d   = dec_cfg.get('latent_d',   512),
            d_model    = dec_cfg.get('d_model',    512),
            n_heads    = dec_cfg.get('n_heads',    8),
            n_layers   = dec_cfg.get('n_layers',   6),
            d_ff       = dec_cfg.get('d_ff',       2048),
            n_commands = dec_cfg.get('n_commands', 6),
            args_dim   = dec_cfg.get('args_dim',   256),
            n_args     = dec_cfg.get('n_args',     16),
            eos_idx    = dec_cfg.get('eos_idx',    3),
            max_len    = dec_cfg.get('max_len',    60),
        ).to(self.device)
        self.decoder.load_state_dict(dec_ckpt['decoder'])
        self.decoder.eval()
        print(f"  epoch={self.dec_epoch}  "
              f"val_cmd_acc={dec_ckpt.get('val_cmd_acc',0):.4f}  "
              f"val_args_acc={dec_ckpt.get('val_args_acc',0):.4f}")

        # ── Bridge + correction ───────────────────────────────────────────────
        print(f"\nLoading bridge: {cfg.bridge_ckpt}")
        br_ckpt      = torch.load(cfg.bridge_ckpt, map_location=self.device)
        self.bridge  = TextToLatentBridge().to(self.device)
        self.bridge.load_state_dict(br_ckpt['bridge'])
        self.bridge.eval()
        print(f"  epoch={br_ckpt['epoch']}  val_cos={br_ckpt['val_cos_sim']:.4f}")

        if os.path.exists(cfg.correction_pt):
            corr = torch.load(cfg.correction_pt, map_location='cpu')
            self.norm_scale  = corr['norm_scale']
            self.feat_scale  = corr['feat_scale'].to(self.device)
            print(f"  correction loaded: norm_scale={self.norm_scale:.4f}")
        else:
            self.norm_scale  = 1.0
            self.feat_scale  = None
            print("  correction.pt not found — running without correction")

        # ── Latent cache ──────────────────────────────────────────────────────
        print(f"\nLoading val latent cache...")
        self.val_cache = np.load(cfg.cache_val, allow_pickle=True).item()
        print(f"  {len(self.val_cache):,} UIDs")

        # ── CAD Dataset (for gt commands/args) ────────────────────────────────
        cad_cfg = SimpleNamespace(
            data_root=cfg.data_root, augment=False,
            max_n_loops=6, max_n_curves=6,
            max_total_len=60, batch_size=cfg.batch_size, num_workers=4,
        )
        ds = CADDataset('validation', cad_cfg)

        loader = DataLoader(ds, batch_size=cfg.batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)

        all_cmds, all_args, all_ids = [], [], []
        n_target = cfg.n_samples if cfg.n_samples > 0 else len(ds)
        n_loaded = 0
        for batch in loader:
            take = min(batch['command'].size(0), n_target - n_loaded)
            all_cmds.append(batch['command'][:take])
            all_args.append(batch['args'][:take])
            all_ids.extend(batch['id'][:take])
            n_loaded += take
            if n_loaded >= n_target:
                break

        self.commands = torch.cat(all_cmds).to(self.device)  # [N, 60]
        self.args     = torch.cat(all_args).to(self.device)  # [N, 60, 16]
        self.uids     = all_ids
        self.N        = self.commands.size(0)

        # Build z_gt tensor from cache
        z_list = []
        for uid in self.uids:
            z_list.append(torch.from_numpy(self.val_cache[uid].copy()).float())
        self.z_gt = torch.stack(z_list).to(self.device)      # [N, 512]

        print(f"  Commands loaded: {self.N:,} val samples")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _struct_features(self, commands: torch.Tensor) -> dict:
        """Extract n_ext, seq_len, prim_ratio from [B, 60] command tensor."""
        n_ext   = (commands == EXT_IDX).sum(1).float().cpu()
        n_line  = (commands == LINE_IDX).sum(1).float().cpu()
        n_arc   = (commands == ARC_IDX).sum(1).float().cpu()
        n_circ  = (commands == CIRCLE_IDX).sum(1).float().cpu()
        seq_len = (commands != EOS_IDX).sum(1).float().cpu()
        prim_r  = (n_arc + n_circ) / (n_line + n_arc + n_circ + 1e-6)
        has_sol = (commands[:, 0] == SOL_IDX).cpu()
        has_eos = (commands == EOS_IDX).any(1).cpu()
        has_ext = (commands == EXT_IDX).any(1).cpu()
        struct_valid = has_sol & has_eos & has_ext
        return {
            'n_ext'        : n_ext,
            'seq_len'      : seq_len,
            'prim_ratio'   : prim_r,
            'struct_valid' : struct_valid.float(),
        }

    def _apply_bridge_correction(self, z: torch.Tensor) -> torch.Tensor:
        z = z * self.norm_scale
        if self.feat_scale is not None:
            z = z * self.feat_scale.unsqueeze(0)
        return z

    def _generate_batch(self, z_batch: torch.Tensor) -> torch.Tensor:
        """
        Autoregressively decode a batch of z vectors.
        Returns [B, 60] predicted commands.
        Uses beam_k=1 (greedy) for speed in analysis.
        """
        all_cmds = []
        for i in range(z_batch.size(0)):
            z1 = z_batch[i:i+1]   # [1, 512]
            with torch.no_grad():
                cmds, _ = self.decoder.generate(
                    z1, beam_k=self.cfg.beam_k, temperature=0.8)
            all_cmds.append(cmds.unsqueeze(0))
        return torch.cat(all_cmds, dim=0)   # [B, 60]

    # ── Module A: Teacher Forcing Accuracy ───────────────────────────────────

    def module_A_teacher_forcing(self) -> dict:
        """
        Re-compute teacher forcing accuracy on val set.
        Must match training log (cmd_acc >0.85, args_acc >0.90 at epoch 100).
        """
        cmd_correct = cmd_total = args_correct = args_total = 0

        for i in range(0, self.N, self.cfg.batch_size):
            c = self.commands[i:i+self.cfg.batch_size]
            a = self.args[i:i+self.cfg.batch_size]
            z = self.z_gt[i:i+self.cfg.batch_size]

            with torch.no_grad():
                cmd_logits, args_logits = self.decoder(z, c, a)

            # Command accuracy (non-EOS positions)
            cmd_mask = (c != EOS_IDX)
            pred_cmd = cmd_logits.argmax(-1)
            cmd_correct += (pred_cmd[cmd_mask] == c[cmd_mask]).sum().item()
            cmd_total   += cmd_mask.sum().item()

            # Args accuracy (non-PAD positions)
            args_mask = (a != -1)
            pred_args = args_logits.argmax(-1) - 1    # shift back
            args_correct += (pred_args[args_mask] == a[args_mask]).sum().item()
            args_total   += args_mask.sum().item()

        cmd_acc  = cmd_correct  / max(cmd_total, 1)
        args_acc = args_correct / max(args_total, 1)

        r = {'cmd_acc': round(cmd_acc, 4), 'args_acc': round(args_acc, 4)}
        cs = '✓' if cmd_acc  > 0.85 else '⚠'
        as_ = '✓' if args_acc > 0.90 else '~ rising'

        print("\n── Module A  Teacher Forcing Accuracy ───────────────────────────")
        print(f"  cmd_acc  : {cmd_acc:.4f}  [{cs}]  target >0.85")
        print(f"  args_acc : {args_acc:.4f}  [{as_}]  target >0.90")
        print(f"  (must match training log — if differs, wrong checkpoint loaded)")
        return r

    # ── Module B: Autoregressive Generation from GT z ─────────────────────────

    def module_B_autoregressive_gt_z(self, n: int = 200) -> dict:
        """
        Generate sequences from GT latent vectors (no text involved).
        Tests decoder in isolation. Upper bound of the full pipeline.

        Key checks:
            struct_valid_rate : fraction starting SOL, containing EXT, containing EOS
            n_ext_match       : fraction where n_ext matches GT exactly
            seq_len_error     : mean |predicted_seq_len - gt_seq_len|
            eos_rate          : fraction of sequences that terminate with EOS
        """
        n = min(n, self.N)
        print(f"\n── Module B  Autoregressive from GT z  (n={n}, beam_k={self.cfg.beam_k}) ─")
        print("  Generating... (may take a few minutes for beam_k>1)")

        gt_feats   = self._struct_features(self.commands[:n])
        pred_cmds_list = []

        t0 = time.time()
        for i in range(0, n, self.cfg.batch_size):
            z_b = self.z_gt[i:i+self.cfg.batch_size]
            pred_cmds_list.append(self._generate_batch(z_b).cpu())
        pred_cmds = torch.cat(pred_cmds_list, dim=0)   # [n, 60]
        elapsed   = time.time() - t0

        pred_feats = self._struct_features(pred_cmds)

        struct_rate = pred_feats['struct_valid'].mean().item()
        # n_ext_match = (pred_feats['n_ext'] == gt_feats['n_ext']).float().mean().item()
        # Align sizes — batch generation may overshoot n by up to (batch_size - 1)
        n_actual   = min(len(pred_feats['n_ext']), len(gt_feats['n_ext']))
        pred_feats = {k: v[:n_actual] for k, v in pred_feats.items()}
        gt_feats   = {k: v[:n_actual] for k, v in gt_feats.items()}
        n_ext_match = (pred_feats['n_ext'] == gt_feats['n_ext']).float().mean().item()
        
        n_ext_close = ((pred_feats['n_ext'] - gt_feats['n_ext']).abs() <= 1).float().mean().item()
        seq_len_err = (pred_feats['seq_len'] - gt_feats['seq_len']).abs().mean().item()
        eos_rate    = pred_feats['struct_valid'].mean().item()   # eos is in struct_valid
        prim_r_err  = (pred_feats['prim_ratio'] - gt_feats['prim_ratio']).abs().mean().item()

        # How often does generation terminate early (seq_len < 60)?
        term_early  = (pred_feats['seq_len'] < 60).float().mean().item()

        r = {
            'struct_valid_rate': round(struct_rate, 4),
            'n_ext_exact_match': round(n_ext_match, 4),
            'n_ext_within_1'   : round(n_ext_close, 4),
            'seq_len_mae'      : round(seq_len_err, 2),
            'prim_ratio_mae'   : round(prim_r_err, 4),
            'early_termination': round(term_early, 4),
            'elapsed_s'        : round(elapsed, 1),
        }

        sv = '✓' if struct_rate > 0.70 else '⚠'
        ne = '✓' if n_ext_match > 0.50 else '⚠'

        print(f"  Structural valid  : {struct_rate:.4f}  [{sv}]  target >0.70")
        print(f"  n_ext exact match : {n_ext_match:.4f}  [{ne}]  target >0.50")
        print(f"  n_ext within ±1   : {n_ext_close:.4f}        target >0.75")
        print(f"  seq_len MAE       : {seq_len_err:.2f} tokens")
        print(f"  prim_ratio MAE    : {prim_r_err:.4f}")
        print(f"  Early termination : {term_early:.4f}  (fraction that output EOS before pos 59)")
        print(f"  Time              : {elapsed:.1f}s  ({elapsed/n*1000:.1f} ms/sample)")

        if struct_rate < 0.50:
            print("  ⚠ Low struct rate: decoder may not be generating SOL/EXT/EOS correctly")
            print("    Check: is the decoder trained enough? (epoch 100 expected)")
        if term_early < 0.30:
            print("  ⚠ Most sequences run to length 60 without EOS termination.")
            print("    Cause: EOS was excluded from command loss (ignore_index=3).")
            print("    Fix: apply the EOS mask correction in decoder_loss before Stage 4 eval.")

        return r

    # ── Module C: Command Distribution ────────────────────────────────────────

    def module_C_command_distribution(self, n: int = 200) -> dict:
        """
        Compare predicted (autoregressive from GT z) vs GT command distributions.
        Well-trained decoder should produce similar frequency of each command type.
        Large divergence in EXT frequency → decoder generates wrong model complexity.
        """
        n = min(n, self.N)

        # GT distribution
        gt_cmds   = self.commands[:n].cpu()
        gt_valid  = gt_cmds[gt_cmds != EOS_IDX]

        # Predicted distribution (reuse from Module B if cached, else recompute)
        pred_cmds_list = []
        for i in range(0, n, self.cfg.batch_size):
            pred_cmds_list.append(self._generate_batch(self.z_gt[i:i+self.cfg.batch_size]).cpu())
        pred_cmds  = torch.cat(pred_cmds_list)
        pred_valid = pred_cmds[pred_cmds != EOS_IDX]

        gt_counts   = {CMD_NAMES[k]: (gt_valid == k).sum().item()   for k in range(6) if k != 3}
        pred_counts = {CMD_NAMES[k]: (pred_valid == k).sum().item() for k in range(6) if k != 3}

        gt_total   = sum(gt_counts.values())
        pred_total = sum(pred_counts.values())

        gt_pct   = {k: v/gt_total*100   for k, v in gt_counts.items()}
        pred_pct = {k: v/pred_total*100 for k, v in pred_counts.items()}

        # Jensen-Shannon divergence as single scalar
        gt_p   = np.array([gt_pct.get(n, 0)   for n in ['LINE','ARC','CIRCLE','SOL','EXT']]) / 100
        pred_p = np.array([pred_pct.get(n, 0) for n in ['LINE','ARC','CIRCLE','SOL','EXT']]) / 100
        m      = (gt_p + pred_p) / 2 + 1e-10
        jsd    = 0.5 * np.sum(gt_p * np.log(gt_p/m + 1e-10) + pred_p * np.log(pred_p/m + 1e-10))

        r = {
            'gt_pct'  : {k: round(v, 2) for k, v in gt_pct.items()},
            'pred_pct': {k: round(v, 2) for k, v in pred_pct.items()},
            'jsd'     : round(float(jsd), 4),
        }

        print("\n── Module C  Command Distribution ──────────────────────────────")
        print(f"  {'Command':<10} {'GT %':>8} {'Pred %':>8} {'Diff':>8}")
        print(f"  {'-'*38}")
        for cmd in ['LINE','ARC','CIRCLE','SOL','EXT']:
            g = gt_pct.get(cmd, 0)
            p = pred_pct.get(cmd, 0)
            print(f"  {cmd:<10} {g:>8.2f} {p:>8.2f} {p-g:>+8.2f}")
        print(f"  {'-'*38}")
        status = '✓' if jsd < 0.05 else ('~ ok' if jsd < 0.15 else '⚠ high')
        print(f"  Jensen-Shannon div: {jsd:.4f}  [{status}]  target <0.05")
        return r

    # ── Module D: Per-Type Args Accuracy ──────────────────────────────────────

    def module_D_args_quality(self) -> dict:
        """
        Break down args reconstruction accuracy by command type.
        Each command type uses different arg slots:
            LINE   : 4 coords (x1,y1,x2,y2)
            ARC    : 5 params (center_x, center_y, radius, start_angle, end_angle)
            CIRCLE : 3 params (center_x, center_y, radius)
            EXT    : 9 params (extrude parameters)
        """
        type_stats = {t: {'correct': 0, 'total': 0, 'l1_sum': 0.0}
                      for t in CMD_NAMES.values() if t != 'EOS'}

        for i in range(0, self.N, self.cfg.batch_size):
            c = self.commands[i:i+self.cfg.batch_size]
            a = self.args[i:i+self.cfg.batch_size]
            z = self.z_gt[i:i+self.cfg.batch_size]

            with torch.no_grad():
                _, args_logits = self.decoder(z, c, a)   # [B, 60, 16, 257]

            pred_args = args_logits.argmax(-1) - 1       # [B, 60, 16]

            for cmd_idx, cmd_name in CMD_NAMES.items():
                if cmd_name == 'EOS':
                    continue
                pos_mask  = (c == cmd_idx)                # [B, 60]
                args_mask = (a != -1)                     # [B, 60, 16]
                combined  = pos_mask.unsqueeze(-1) & args_mask  # [B, 60, 16]

                if combined.sum() == 0:
                    continue

                gt_vals   = a[combined]
                pred_vals = pred_args[combined]

                type_stats[cmd_name]['correct'] += (pred_vals == gt_vals).sum().item()
                type_stats[cmd_name]['total']   += combined.sum().item()
                type_stats[cmd_name]['l1_sum']  += (pred_vals - gt_vals).abs().float().sum().item()

        r = {}
        print("\n── Module D  Per-Type Args Accuracy ─────────────────────────────")
        print(f"  {'Type':<10} {'acc':>8} {'MAE (bins)':>12} {'samples':>10}")
        print(f"  {'-'*44}")

        for cmd_name in ['LINE','ARC','CIRCLE','SOL','EXT']:
            stats = type_stats[cmd_name]
            if stats['total'] == 0:
                print(f"  {cmd_name:<10} {'—':>8} {'—':>12} {'0':>10}")
                continue
            acc  = stats['correct'] / stats['total']
            mae  = stats['l1_sum']  / stats['total']
            r[cmd_name] = {'acc': round(acc, 4), 'mae': round(mae, 3)}
            flag = '✓' if acc > 0.88 else '⚠'
            print(f"  {cmd_name:<10} {acc:>8.4f} {mae:>12.3f} {stats['total']:>10,}  {flag}")

        print(f"\n  EXT args accuracy is most critical for geometric correctness.")
        return r

    # ── Module E: Z-Conditioning Test ─────────────────────────────────────────

    def module_E_z_conditioning(self, n: int = 100) -> dict:
        """
        Validates that the decoder actually uses z to condition generation.
        Test: does z_gt_i → decode produce sequences more similar to GT_i
              than z_gt_shuffled → decode?

        If the decoder ignores z (unconditional), both will score equally.
        Large gap = decoder is properly z-conditioned.

        Also tests: does random z → decode produce worse structural match?
        """
        n = min(n, self.N)

        gt_cmds = self.commands[:n].cpu()

        # 1. Decode from GT z
        pred_from_gt = []
        for i in range(0, n, self.cfg.batch_size):
            pred_from_gt.append(self._generate_batch(self.z_gt[i:i+self.cfg.batch_size]).cpu())
        pred_from_gt = torch.cat(pred_from_gt)

        # 2. Decode from shuffled z (wrong z for each sample)
        perm         = torch.randperm(n)
        z_shuffled   = self.z_gt[:n][perm]
        pred_from_shuf = []
        for i in range(0, n, self.cfg.batch_size):
            pred_from_shuf.append(self._generate_batch(z_shuffled[i:i+self.cfg.batch_size]).cpu())
        pred_from_shuf = torch.cat(pred_from_shuf)

        # 3. Decode from random z
        z_random = torch.randn_like(self.z_gt[:n]) * self.z_gt[:n].std() + self.z_gt[:n].mean()
        pred_from_rand = []
        for i in range(0, n, self.cfg.batch_size):
            pred_from_rand.append(self._generate_batch(z_random[i:i+self.cfg.batch_size].to(self.device)).cpu())
        pred_from_rand = torch.cat(pred_from_rand)

        def _n_ext_match(pred, gt):
            p_ext = (pred == EXT_IDX).sum(1).float()
            g_ext = (gt   == EXT_IDX).sum(1).float()
            return (p_ext == g_ext).float().mean().item()

        def _struct_valid_rate(pred):
            sf = self._struct_features(pred)
            return sf['struct_valid'].mean().item()

        n_ext_gt   = _n_ext_match(pred_from_gt,   gt_cmds)
        n_ext_shuf = _n_ext_match(pred_from_shuf, gt_cmds)
        n_ext_rand = _n_ext_match(pred_from_rand, gt_cmds)

        sv_gt   = _struct_valid_rate(pred_from_gt)
        sv_shuf = _struct_valid_rate(pred_from_shuf)
        sv_rand = _struct_valid_rate(pred_from_rand)

        r = {
            'n_ext_match_gt'    : round(n_ext_gt, 4),
            'n_ext_match_shuf'  : round(n_ext_shuf, 4),
            'n_ext_match_rand'  : round(n_ext_rand, 4),
            'struct_valid_gt'   : round(sv_gt, 4),
            'struct_valid_shuf' : round(sv_shuf, 4),
            'struct_valid_rand' : round(sv_rand, 4),
            'conditioning_lift' : round(n_ext_gt - n_ext_rand, 4),
        }

        cond_ok = n_ext_gt > n_ext_shuf > n_ext_rand
        status  = '✓ HEALTHY' if cond_ok else '⚠ check'

        print("\n── Module E  Z-Conditioning Test ────────────────────────────────")
        print(f"  {'Condition':<22} {'n_ext match':>12} {'struct valid':>14}")
        print(f"  {'-'*52}")
        print(f"  {'GT z (upper bound)':<22} {n_ext_gt:>12.4f} {sv_gt:>14.4f}")
        print(f"  {'Shuffled z':<22} {n_ext_shuf:>12.4f} {sv_shuf:>14.4f}")
        print(f"  {'Random z (lower)':<22} {n_ext_rand:>12.4f} {sv_rand:>14.4f}")
        print(f"  {'-'*52}")
        print(f"  Conditioning lift  : +{r['conditioning_lift']:.4f}  (GT vs random)")
        print(f"  Ordering GT > shuf > rand: {cond_ok}  [{status}]")

        if not cond_ok:
            print("  ⚠ Decoder may be ignoring z. Check z_proj layer is not zeroed.")
        return r

    # ── Module F: Distribution Shift (Stage 2→3 interface) ───────────────────

    def module_F_distribution_shift(self, n: int = 200) -> dict:
        """
        Checks that bridge z* (with correction) lands in the same distribution
        as the cached z used to train the decoder. If not, decoder gets OOD input.

        Critical: norm ratio should be 0.90–1.10 after correction.
        """
        val_ds = TextLatentDataset(
            cache_path       = self.cfg.cache_val,
            annotations_path = self.cfg.annotations,
            split_path       = self.cfg.split_path,
            phase='validation', levels=['intermediate'],
        )
        loader = DataLoader(val_ds, batch_size=256, shuffle=False,
                            collate_fn=_collate_text_latent)

        z_pred_list, z_gt_list = [], []
        n_collected = 0
        with torch.no_grad():
            for batch in loader:
                z_p = self.bridge(self.bridge.tokenize(batch['text'], self.device)).cpu()
                z_p = self._apply_bridge_correction(z_p.to(self.device)).cpu()
                z_pred_list.append(z_p)
                z_gt_list.append(batch['z_target'])
                n_collected += z_p.size(0)
                if n_collected >= n:
                    break

        z_pred = torch.cat(z_pred_list)[:n]
        z_gt   = torch.cat(z_gt_list)[:n]

        norm_ratio = z_pred.norm(dim=1).mean() / z_gt.norm(dim=1).mean()
        std_ratio  = z_pred.std(dim=0).mean()  / z_gt.std(dim=0).mean()
        cos_sim    = F.cosine_similarity(z_pred, z_gt, dim=1).mean().item()

        # Pairwise diversity ratio (collapse check)
        idx  = torch.randperm(min(200, n))
        zp_n = F.normalize(z_pred[idx], dim=1)
        zg_n = F.normalize(z_gt[idx], dim=1)
        mask = ~torch.eye(len(idx), dtype=torch.bool)
        div_ratio = (zp_n @ zp_n.T)[mask].mean() / ((zg_n @ zg_n.T)[mask].mean() + 1e-6)

        r = {
            'norm_ratio'   : round(norm_ratio.item(), 4),
            'std_ratio'    : round(std_ratio.item(), 4),
            'cos_sim'      : round(cos_sim, 4),
            'div_ratio'    : round(div_ratio.item(), 4),
        }

        nr_ok  = '✓' if 0.90 < norm_ratio < 1.10 else '⚠'
        sr_ok  = '✓' if 0.70 < std_ratio  < 1.10 else '⚠'
        dr_ok  = '✓' if div_ratio < 1.30          else '⚠'

        print("\n── Module F  Stage 2→3 Distribution Shift ───────────────────────")
        print(f"  {'Metric':<22} {'After correction':>18} {'Target':>12}")
        print(f"  {'-'*54}")
        print(f"  {'norm ratio':<22} {norm_ratio:>18.4f} {'0.90–1.10':>12}  {nr_ok}")
        print(f"  {'std ratio':<22} {std_ratio:>18.4f} {'>0.70':>12}  {sr_ok}")
        print(f"  {'val_cos_sim':<22} {cos_sim:>18.4f} {'>0.82':>12}")
        print(f"  {'diversity ratio':<22} {div_ratio:>18.4f} {'<1.30':>12}  {dr_ok}")

        if not (0.90 < norm_ratio < 1.10):
            print(f"\n  ⚠ Norm ratio {norm_ratio:.3f} outside target — recompute correction.pt")
        return r

    # ── Module G: Full Stage 1+2+3 Chain ─────────────────────────────────────

    def module_G_chain_123(self, n: int = 200) -> dict:
        """
        End-to-end: text → bridge → correction → decoder.generate() → structure

        Three conditions compared on same n val samples:
            GT z    → decode  (upper bound: perfect z)
            Bridge z → decode (what the full system achieves)
            Random z → decode (lower bound: no conditioning)

        Structural match vs GT is the proxy for CD without running CadQuery.
        """
        n = min(n, self.N)
        print(f"\n── Module G  Stage 1+2+3 Chain  (n={n}) ─────────────────────────")

        # Load text annotations for our val UIDs
        import json
        with open(self.cfg.annotations) as f:
            annot = json.load(f)

        texts = []
        valid_idx = []
        for i, uid in enumerate(self.uids[:n]):
            t = annot.get(uid, {}).get('intermediate', None)
            if t:
                texts.append(t)
                valid_idx.append(i)

        valid_idx = valid_idx[:n]
        texts     = texts[:n]
        nv        = len(valid_idx)
        print(f"  Valid text-annotated samples: {nv}")

        # GT commands for valid subset
        gt_cmds_sub = self.commands[[i for i in valid_idx]].cpu()
        gt_z_sub    = self.z_gt[[i for i in valid_idx]]

        # ── Condition 1: GT z → decode ────────────────────────────────────────
        pred_gt = []
        for i in range(0, nv, self.cfg.batch_size):
            pred_gt.append(self._generate_batch(gt_z_sub[i:i+self.cfg.batch_size]).cpu())
        pred_gt = torch.cat(pred_gt)

        # ── Condition 2: Bridge z → decode ────────────────────────────────────
        z_bridge_list = []
        bs = 256
        with torch.no_grad():
            for i in range(0, nv, bs):
                t_batch = texts[i:i+bs]
                tok     = self.bridge.tokenize(t_batch, self.device)
                z_p     = self.bridge(tok)
                z_p     = self._apply_bridge_correction(z_p)
                z_bridge_list.append(z_p)
        z_bridge = torch.cat(z_bridge_list)

        pred_bridge = []
        for i in range(0, nv, self.cfg.batch_size):
            pred_bridge.append(self._generate_batch(z_bridge[i:i+self.cfg.batch_size]).cpu())
        pred_bridge = torch.cat(pred_bridge)

        # ── Condition 3: Random z → decode ────────────────────────────────────
        z_rand      = torch.randn_like(gt_z_sub) * gt_z_sub.std() + gt_z_sub.mean()
        pred_rand   = []
        for i in range(0, nv, self.cfg.batch_size):
            pred_rand.append(self._generate_batch(z_rand[i:i+self.cfg.batch_size].to(self.device)).cpu())
        pred_rand = torch.cat(pred_rand)

        # ── Compare ───────────────────────────────────────────────────────────
        def _metrics(pred, gt):
            p_sf = self._struct_features(pred)
            g_sf = self._struct_features(gt)
            n_ext_match = (p_sf['n_ext'] == g_sf['n_ext']).float().mean().item()
            n_ext_close = ((p_sf['n_ext'] - g_sf['n_ext']).abs() <= 1).float().mean().item()
            struct_rate = p_sf['struct_valid'].mean().item()
            prim_err    = (p_sf['prim_ratio'] - g_sf['prim_ratio']).abs().mean().item()
            return {'n_ext_exact': n_ext_match, 'n_ext_within1': n_ext_close,
                    'struct_valid': struct_rate, 'prim_ratio_err': prim_err}

        m_gt     = _metrics(pred_gt,     gt_cmds_sub)
        m_bridge = _metrics(pred_bridge, gt_cmds_sub)
        m_rand   = _metrics(pred_rand,   gt_cmds_sub)

        r = {'gt': m_gt, 'bridge': m_bridge, 'random': m_rand}

        # Pipeline efficiency: what % of GT z quality does bridge recover?
        rand_base = m_rand['n_ext_exact']
        gt_ceil   = m_gt['n_ext_exact']
        bridge_v  = m_bridge['n_ext_exact']
        efficiency = (bridge_v - rand_base) / (gt_ceil - rand_base + 1e-6)

        print(f"\n  {'Metric':<20} {'GT z (ceil)':>12} {'Bridge z':>12} {'Random z (floor)':>18}")
        print(f"  {'-'*66}")
        for key in ['n_ext_exact', 'n_ext_within1', 'struct_valid', 'prim_ratio_err']:
            print(f"  {key:<20} {m_gt[key]:>12.4f} {m_bridge[key]:>12.4f} {m_rand[key]:>18.4f}")
        print(f"  {'-'*66}")
        print(f"  Pipeline efficiency: {efficiency:.3f}  "
              f"(bridge recovers {efficiency*100:.1f}% of the gap between random and GT z)")

        if efficiency > 0.70:
            print("  → Stage 1+2+3 chain is working well ✓")
        elif efficiency > 0.40:
            print("  → Chain is functional, room for improvement in Stage 2")
        else:
            print("  ⚠ Bridge is not adding much over random — check correction.pt")

        r['pipeline_efficiency'] = round(efficiency, 4)
        return r

    # ── Run all ───────────────────────────────────────────────────────────────

    def run_all(self) -> dict:
        print("\n" + "=" * 68)
        print(f"  CAD-JEPA Decoder Analysis  —  epoch_{self.dec_epoch:04d}.pt")
        print("=" * 68)

        results = {'decoder_epoch': self.dec_epoch, 'n_samples': self.N}

        results['A_teacher_forcing']    = self.module_A_teacher_forcing()
        results['B_autoregressive']     = self.module_B_autoregressive_gt_z(n=min(200, self.N))
        results['C_cmd_distribution']   = self.module_C_command_distribution(n=min(200, self.N))
        results['D_args_quality']       = self.module_D_args_quality()
        results['E_z_conditioning']     = self.module_E_z_conditioning(n=min(100, self.N))
        results['F_dist_shift']         = self.module_F_distribution_shift(n=200)
        results['G_chain_123']          = self.module_G_chain_123(n=min(200, self.N))

        self._print_summary(results)

        os.makedirs(self.cfg.output_dir, exist_ok=True)
        out = os.path.join(self.cfg.output_dir,
                           f'decoder_analysis_epoch_{self.dec_epoch:04d}.json')
        with open(out, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved → {out}")
        return results

    def _print_summary(self, results):
        A = results.get('A_teacher_forcing', {})
        B = results.get('B_autoregressive', {})
        E = results.get('E_z_conditioning', {})
        F = results.get('F_dist_shift', {})
        G = results.get('G_chain_123', {})

        def ok(val, thresh): return '✓' if val >= thresh else '⚠'

        cmd_acc   = A.get('cmd_acc', 0)
        args_acc  = A.get('args_acc', 0)
        sv_rate   = B.get('struct_valid_rate', 0)
        cond_lift = E.get('conditioning_lift', 0)
        norm_r    = F.get('norm_ratio', 0)
        eff       = G.get('pipeline_efficiency', 0)

        print("\n" + "=" * 68)
        print("  SUMMARY")
        print("=" * 68)
        print(f"  [Stage 3 standalone]")
        print(f"    cmd_acc (TF)       : {cmd_acc:.4f}  {ok(cmd_acc, 0.85)}   target >0.85")
        print(f"    args_acc (TF)      : {args_acc:.4f}  {ok(args_acc, 0.90)}   target >0.90")
        print(f"    struct_valid (AR)  : {sv_rate:.4f}  {ok(sv_rate, 0.70)}   target >0.70")
        print(f"    z-conditioning lift: {cond_lift:.4f}  {ok(cond_lift, 0.10)}   target >0.10")
        print(f"\n  [Stage 2→3 interface]")
        print(f"    norm ratio         : {norm_r:.4f}  {ok(abs(1-norm_r)<0.10, 1)}   target 0.90–1.10")
        print(f"\n  [Full chain 1+2+3]")
        print(f"    pipeline efficiency: {eff:.4f}  {ok(eff, 0.50)}   target >0.50")

        n_ok = sum([cmd_acc >= 0.85, args_acc >= 0.90, sv_rate >= 0.70,
                    cond_lift >= 0.10, 0.90 < norm_r < 1.10, eff >= 0.50])
        print(f"\n  Overall: {n_ok}/6 checks passing")

        if n_ok >= 5:
            print("  → All systems healthy. Proceed to label efficiency + evaluation.")
        elif n_ok >= 3:
            print("  → Mostly healthy. Check individual failing modules.")
        else:
            print("  → Multiple checks failing. Investigate before proceeding.")
        print("=" * 68)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--decoder-ckpt', default=AnalysisConfig.decoder_ckpt)
    parser.add_argument('--bridge-ckpt',  default=AnalysisConfig.bridge_ckpt)
    parser.add_argument('--n-samples',    type=int,   default=300)
    parser.add_argument('--beam-k',       type=int,   default=1)
    parser.add_argument('--output-dir',   default=AnalysisConfig.output_dir)
    a = parser.parse_args()

    cfg = AnalysisConfig(
        decoder_ckpt = a.decoder_ckpt,
        bridge_ckpt  = a.bridge_ckpt,
        n_samples    = a.n_samples,
        beam_k       = a.beam_k,
        output_dir   = a.output_dir,
    )
    analyzer = DecoderAnalyzer(cfg)
    analyzer.run_all()


if __name__ == '__main__':
    main()