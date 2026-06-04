"""
evaluation/analyze_encoder.py

Comprehensive diagnostic analysis of the Stage 1 CAD-JEPA encoder.
Six modules, no plots — all output is printed metrics + saved JSON.

Modules:
    1. latent_stats         basic distribution of z vectors
    2. effective_rank       verify rank matches training log (target: 0.70-0.75)
    3. masked_consistency   encoder stability when 50% of tokens are masked
    4. token_separation     do LINE / ARC / EXT form distinct token-level clusters?
    5. knn_coherence        nearest neighbors in latent space = similar CAD structure?
    6. jepa_quality         predictor cosine sim vs EMA target (direct JEPA objective)

Why NOT linear probes:
    Standard linear probes on n_extrusions / primitive type are trivially solved
    by random encoders (~99%) because mean-pooling preserves token frequency directly.
    The 6 modules above test things a random encoder cannot pass.

Usage:
    # Full analysis on val set subset (fast, ~5 min on A100)
    python evaluation/analyze_encoder.py

    # Full val set (all 8946 samples)
    python evaluation/analyze_encoder.py --n-samples -1

    # Specific checkpoint
    python evaluation/analyze_encoder.py --ckpt /path/to/epoch_0300.pt
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
from model.jepa_encoder import CADJEPAEncoder
from model.predictor import CADPredictor


# ── Constants (from cadlib/macro.py) ──────────────────────────────────────────
LINE_IDX   = 0
ARC_IDX    = 1
CIRCLE_IDX = 2
EOS_IDX    = 3
SOL_IDX    = 4
EXT_IDX    = 5

CMD_NAMES  = {
    LINE_IDX: 'LINE', ARC_IDX: 'ARC', CIRCLE_IDX: 'CIRCLE',
    SOL_IDX: 'SOL', EXT_IDX: 'EXT',
}


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class AnalysisConfig:
    ckpt_path    : str   = '/content/drive/MyDrive/cad-jepa-checkpoints/pretrain/epoch_0300.pt'
    data_root    : str   = '/content'
    output_dir   : str   = '/content/drive/MyDrive/cad-jepa-analysis'
    n_samples    : int   = 2000    # val samples to use. -1 = all 8946
    k_neighbors  : int   = 10
    n_mask_runs  : int   = 5       # repeated mask draws for consistency test
    mask_ratio   : float = 0.50
    batch_size   : int   = 256
    seed         : int   = 42


# ──────────────────────────────────────────────────────────────────────────────
# Analyzer
# ──────────────────────────────────────────────────────────────────────────────

class EncoderAnalyzer:

    def __init__(self, cfg: AnalysisConfig):
        self.cfg    = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(cfg.seed)

        # ── Load checkpoint ───────────────────────────────────────────────────
        if not os.path.exists(cfg.ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {cfg.ckpt_path}")

        print(f"Loading checkpoint: {cfg.ckpt_path}")
        ckpt = torch.load(cfg.ckpt_path, map_location=self.device)
        self.epoch = ckpt['epoch']

        # Build encoder config
        enc_cfg           = ConfigJEPA()
        enc_cfg.data_root = cfg.data_root
        enc_cfg.augment   = False
        self.enc_cfg      = enc_cfg

        # Context encoder
        self.enc = CADJEPAEncoder(enc_cfg).to(self.device)
        self.enc.load_state_dict(ckpt['encoder'])
        self.enc.eval()
        for p in self.enc.parameters():
            p.requires_grad_(False)

        # EMA target encoder (same architecture, updated by EMA during training)
        self.ema_enc = CADJEPAEncoder(enc_cfg).to(self.device)
        self.ema_enc.load_state_dict(ckpt['ema']['target'])
        self.ema_enc.eval()
        for p in self.ema_enc.parameters():
            p.requires_grad_(False)

        # Predictor (for module 6)
        self.predictor = CADPredictor(enc_cfg).to(self.device)
        self.predictor.load_state_dict(ckpt['predictor'])
        self.predictor.eval()
        for p in self.predictor.parameters():
            p.requires_grad_(False)

        n_enc = sum(p.numel() for p in self.enc.parameters())
        tau   = ckpt['ema'].get('tau', 0.0)
        print(f"  Epoch : {self.epoch}")
        print(f"  Params: {n_enc:,}")
        print(f"  EMA τ : {tau:.4f}")
        print(f"  Device: {self.device}")

    # ── Data loading ──────────────────────────────────────────────────────────

    def _load_val_samples(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (commands [N,60], args [N,60,16]) on device."""
        ds     = CADDataset('validation', self.enc_cfg)
        loader = DataLoader(ds, batch_size=self.cfg.batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)

        n_target = self.cfg.n_samples if self.cfg.n_samples > 0 else len(ds)
        all_cmds, all_args = [], []
        n_loaded = 0

        for batch in loader:
            take = min(batch['command'].size(0), n_target - n_loaded)
            all_cmds.append(batch['command'][:take])
            all_args.append(batch['args'][:take])
            n_loaded += take
            if n_loaded >= n_target:
                break

        commands = torch.cat(all_cmds).to(self.device)  # [N, 60]
        args     = torch.cat(all_args).to(self.device)  # [N, 60, 16]
        print(f"Loaded {commands.size(0):,} validation samples")
        return commands, args

    # ── Encoding helpers ──────────────────────────────────────────────────────

    def _encode_full(self, commands: torch.Tensor,
                     args: torch.Tensor, bs: int = 256) -> torch.Tensor:
        """No mask, mean pool → [N, 512]  (CPU)."""
        out = []
        for i in range(0, len(commands), bs):
            with torch.no_grad():
                z = self.enc.encode_mean(commands[i:i+bs], args[i:i+bs])
            out.append(z.cpu())
        return torch.cat(out)

    def _encode_masked(self, commands: torch.Tensor,
                       args: torch.Tensor,
                       mask: torch.Tensor) -> torch.Tensor:
        """
        Encode with jepa_mask applied.
        mask: [B, 60] bool — True = hidden from context encoder.
        Mean-pool over visible (not masked, not EOS) positions → [B, 512]  (CPU).
        """
        with torch.no_grad():
            h = self.enc(commands, args, jepa_mask=mask)   # [B, 60, 512]

        eos_m   = (commands == EOS_IDX)              # [B, 60]
        visible = ~mask & ~eos_m                      # [B, 60]

        h_vis = h * visible.unsqueeze(-1).float()
        n_vis = visible.sum(dim=1, keepdim=True).float().clamp(min=1.0)
        return (h_vis.sum(dim=1) / n_vis).cpu()      # [B, 512]

    def _random_mask(self, commands: torch.Tensor, ratio: float = 0.5) -> torch.Tensor:
        """Sample random mask on non-EOS positions. True = hidden."""
        valid = (commands != EOS_IDX)                       # [B, L]
        mask  = (torch.rand_like(commands.float()) < ratio) & valid
        # Guarantee at least one token visible per sample
        fully_masked = (mask.int() == valid.int()).all(dim=1)
        for i in fully_masked.nonzero(as_tuple=True)[0]:
            first = valid[i].nonzero(as_tuple=True)[0][0]
            mask[i, first] = False
        return mask                                          # [B, L]

    def _struct_features(self, commands: torch.Tensor) -> torch.Tensor:
        """
        Extract structural features from command sequences.
        Returns [N, 3]:
            col 0 = n_ext       (number of extrusion operations)
            col 1 = seq_len     (number of non-EOS tokens)
            col 2 = prim_ratio  ((n_arc + n_circ) / total curves — curved vs linear)
        """
        n_ext  = (commands == EXT_IDX).sum(1).float()
        n_line = (commands == LINE_IDX).sum(1).float()
        n_arc  = (commands == ARC_IDX).sum(1).float()
        n_circ = (commands == CIRCLE_IDX).sum(1).float()
        seq_len = (commands != EOS_IDX).sum(1).float()
        prim_ratio = (n_arc + n_circ) / (n_line + n_arc + n_circ + 1e-6)
        return torch.stack([n_ext, seq_len, prim_ratio], dim=1).cpu()  # [N, 3]

    # ─────────────────────────────────────────────────────────────────────────
    # Module 1 — Latent Statistics
    # ─────────────────────────────────────────────────────────────────────────

    def module_latent_stats(self, latents: torch.Tensor) -> dict:
        """
        Basic health checks on z vectors.

        Expected (well-trained encoder with output_norm):
            z_norm ≈ sqrt(512) ≈ 22.6  (each feature ~ N(0,1) → norm = sqrt(d))
            feat_std_mean ≈ 1.0         (output LayerNorm normalises to unit std)
            dead_dims = 0               (no inactive dimensions)
            z_mean_norm ≈ 0             (representations centred)
        """
        norms   = latents.norm(dim=1)                    # [N]
        f_std   = latents.std(dim=0)                     # [512]
        dead    = (f_std < 0.1).sum().item()
        z_mean  = latents.mean(0).norm().item()
        q5, q95 = norms.quantile(0.05).item(), norms.quantile(0.95).item()

        r = {
            'n_samples'     : latents.size(0),
            'z_norm_mean'   : norms.mean().item(),
            'z_norm_std'    : norms.std().item(),
            'z_norm_q5_q95' : (round(q5, 2), round(q95, 2)),
            'feat_std_mean' : f_std.mean().item(),
            'feat_std_min'  : f_std.min().item(),
            'dead_dims'     : dead,
            'z_mean_norm'   : z_mean,
        }

        print("\n── Module 1  Latent Statistics ──────────────────────────────")
        print(f"  Samples          : {r['n_samples']:,}")
        print(f"  z norm mean ± std: {r['z_norm_mean']:.2f} ± {r['z_norm_std']:.2f}"
              f"  (expected ≈{22.6:.1f})")
        print(f"  z norm [q5, q95] : {r['z_norm_q5_q95']}")
        print(f"  Feature std mean : {r['feat_std_mean']:.4f}  (expected ≈1.0)")
        print(f"  Feature std min  : {r['feat_std_min']:.4f}")
        print(f"  Dead dims        : {dead} / {latents.size(1)}")
        centered = '✓' if z_mean < 2.0 else '✗'
        print(f"  z mean-vec norm  : {z_mean:.3f}  centered: {centered}")
        return r

    # ─────────────────────────────────────────────────────────────────────────
    # Module 2 — Effective Rank
    # ─────────────────────────────────────────────────────────────────────────

    def module_effective_rank(self, latents: torch.Tensor) -> dict:
        """
        Measure effective dimensionality of the representation space.

        rank_frac = R_eff / d_model  (Roy & Vetterli 2007 spectral entropy formula)

        Training log showed 0.70–0.75 throughout.
        rank_frac < 0.50 indicates representation collapse.

        Also computes participation ratio (Gao et al. 2017) and the percentage of
        variance explained by the top-5 singular values — a high top-5 percentage
        indicates a low-rank or collapsed representation.
        """
        z = (latents - latents.mean(0)).float()     # centre
        _, S, _ = torch.linalg.svd(z, full_matrices=False)

        s  = S ** 2
        p  = s / s.sum()
        H  = -(p * torch.log(p + 1e-10)).sum()
        R_eff     = H.exp().item()
        rank_frac = R_eff / latents.size(1)

        # Participation ratio (alternative collapse measure)
        pr_frac   = ((s.sum() ** 2) / (s ** 2).sum()).item() / latents.size(1)

        # Concentration in top-5 singular values
        top5_pct  = (s[:5].sum() / s.sum()).item() * 100

        r = {
            'effective_rank' : round(R_eff, 2),
            'rank_frac'      : round(rank_frac, 4),
            'pr_frac'        : round(pr_frac, 4),
            'top5_pct'       : round(top5_pct, 2),
            'd_model'        : latents.size(1),
        }

        status = 'HEALTHY ✓' if rank_frac >= 0.70 else ('MARGINAL' if rank_frac >= 0.50 else 'COLLAPSED ✗')
        in_range = '✓ matches' if 0.68 < rank_frac < 0.78 else '✗ differs'

        print("\n── Module 2  Effective Rank ──────────────────────────────────")
        print(f"  Effective rank   : {R_eff:.1f} / {latents.size(1)}"
              f"  →  rank_frac = {rank_frac:.4f}  [{status}]")
        print(f"  Participation r  : {pr_frac:.4f}")
        print(f"  Top-5 sv pct     : {top5_pct:.1f}%  "
              f"(low = good, collapse shows as >50%)")
        print(f"  Training log was : 0.70 – 0.75  →  {in_range}")
        return r

    # ─────────────────────────────────────────────────────────────────────────
    # Module 3 — Masked Consistency
    # ─────────────────────────────────────────────────────────────────────────

    def module_masked_consistency(self, commands: torch.Tensor,
                                  args: torch.Tensor,
                                  n: int = 500) -> dict:
        """
        Tests whether the encoder produces stable z vectors when 50% of tokens
        are randomly masked — the core JEPA invariance property.

        A random encoder would score ≈ 0.0 (representations are arbitrary).
        A good JEPA encoder should score > 0.80 (representations are consistent
        regardless of which tokens are visible).

        Three measurements:
            full_vs_masked : cos(z_unmasked, z_masked)   — context sees full vs partial
            cross_masked   : cos(z_mask_i, z_mask_j)     — two independent masked views
            baseline       : cos(z_i, z_j)  i≠j          — different CAD models
        """
        cmds = commands[:n]
        a    = args[:n]

        # Unmasked (gold) representation
        z_full = self._encode_full(cmds, a, bs=128)               # [N, 512]
        z_full_n = F.normalize(z_full, dim=1)

        # n_mask_runs independent masked views
        z_masked_list = []
        for _ in range(self.cfg.n_mask_runs):
            chunks = []
            for i in range(0, n, 128):
                mask = self._random_mask(cmds[i:i+128], self.cfg.mask_ratio)
                z_m  = self._encode_masked(cmds[i:i+128], a[i:i+128], mask)
                chunks.append(z_m)
            z_masked_list.append(torch.cat(chunks))               # [N, 512]

        # full vs masked (per run, then averaged)
        fvm_scores = []
        for z_m in z_masked_list:
            cos = (z_full_n * F.normalize(z_m, dim=1)).sum(dim=1) # [N]
            fvm_scores.append(cos)
        fvm = torch.stack(fvm_scores).reshape(-1)

        # cross-masked (all pairs of runs)
        cross = []
        for i in range(len(z_masked_list)):
            for j in range(i + 1, len(z_masked_list)):
                c = (F.normalize(z_masked_list[i], dim=1)
                     * F.normalize(z_masked_list[j], dim=1)).sum(1)
                cross.append(c)
        cross = torch.cat(cross)

        # Random baseline: cos sim between different models
        perm = torch.randperm(n)
        base = (z_full_n * F.normalize(z_full[perm], dim=1)).sum(1)

        r = {
            'full_vs_masked_mean'  : fvm.mean().item(),
            'full_vs_masked_std'   : fvm.std().item(),
            'cross_masked_mean'    : cross.mean().item(),
            'cross_masked_std'     : cross.std().item(),
            'baseline_mean'        : base.mean().item(),
            'baseline_std'         : base.std().item(),
            'lift_over_baseline'   : (fvm.mean() - base.mean()).item(),
        }

        status = 'HEALTHY ✓' if r['full_vs_masked_mean'] > 0.80 else '⚠ low'
        print("\n── Module 3  Masked Consistency  "
              f"(mask={int(self.cfg.mask_ratio*100)}%, runs={self.cfg.n_mask_runs}) ──")
        print(f"  Full vs masked    : {r['full_vs_masked_mean']:.4f} ± {r['full_vs_masked_std']:.4f}"
              f"  [{status}]  target >0.80")
        print(f"  Cross-masked      : {r['cross_masked_mean']:.4f} ± {r['cross_masked_std']:.4f}")
        print(f"  Random baseline   : {r['baseline_mean']:.4f} ± {r['baseline_std']:.4f}")
        print(f"  Lift over random  : +{r['lift_over_baseline']:.4f}")
        return r

    # ─────────────────────────────────────────────────────────────────────────
    # Module 4 — Token-Type Separation
    # ─────────────────────────────────────────────────────────────────────────

    def module_token_separation(self, commands: torch.Tensor,
                                args: torch.Tensor,
                                n: int = 1000) -> dict:
        """
        Extracts per-token representations (no mean pooling) and measures whether
        different command types occupy distinct regions of the 512-d space.

        Separation ratio = mean inter-cluster distance / mean intra-cluster spread.
        > 2.0 : command types are clearly distinct  ✓
        < 1.0 : all token types look alike          ✗  (collapse)

        Note: a random encoder has sep_ratio ≈ 1.0 by isotropy.
        """
        all_h, all_types = [], []

        for i in range(0, n, 128):
            c = commands[i:i+128]
            a = args[i:i+128]
            with torch.no_grad():
                h = self.enc(c, a, jepa_mask=None)   # [B, 60, 512]

            eos_m = (c == EOS_IDX)
            for b in range(c.size(0)):
                vpos = (~eos_m[b]).nonzero(as_tuple=True)[0]
                all_h.append(h[b, vpos].cpu())
                all_types.append(c[b, vpos].cpu())

        all_h     = torch.cat(all_h)       # [T_total, 512]
        all_types = torch.cat(all_types)   # [T_total]

        centroids   = {}
        intra_dists = {}
        counts      = {}

        for idx, name in CMD_NAMES.items():
            sel = (all_types == idx)
            cnt = sel.sum().item()
            if cnt < 20:
                continue
            h_t      = all_h[sel].float()
            centroid = h_t.mean(0)
            intra    = (h_t - centroid).norm(dim=1).mean().item()
            centroids[name]   = centroid
            intra_dists[name] = intra
            counts[name]      = cnt

        inter_vals, inter_pairs = [], {}
        names = list(centroids.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                n1, n2 = names[i], names[j]
                d = (centroids[n1] - centroids[n2]).norm().item()
                inter_vals.append(d)
                inter_pairs[f"{n1}↔{n2}"] = round(d, 3)

        mean_inter = float(np.mean(inter_vals)) if inter_vals else 0.0
        mean_intra = float(np.mean(list(intra_dists.values())))
        sep_ratio  = mean_inter / (mean_intra + 1e-6)

        r = {
            'sep_ratio'   : round(sep_ratio, 3),
            'mean_inter'  : round(mean_inter, 3),
            'mean_intra'  : round(mean_intra, 3),
            'inter_pairs' : inter_pairs,
            'intra_dists' : {k: round(v, 3) for k, v in intra_dists.items()},
            'token_counts': counts,
        }

        status = 'HEALTHY ✓' if sep_ratio > 2.0 else '⚠ low'
        print("\n── Module 4  Token Type Separation ──────────────────────────")
        print(f"  {'Type':<10} {'count':>8}  {'intra-dist':>10}")
        for name in names:
            print(f"  {name:<10} {counts.get(name, 0):>8,}  "
                  f"{intra_dists.get(name, 0):>10.3f}")
        print(f"\n  Inter-cluster distances:")
        for pair, d in inter_pairs.items():
            print(f"    {pair:<18} : {d:.3f}")
        print(f"\n  Mean inter-dist : {mean_inter:.3f}")
        print(f"  Mean intra-dist : {mean_intra:.3f}")
        print(f"  Separation ratio: {sep_ratio:.3f}  [{status}]  target >2.0")
        return r

    # ─────────────────────────────────────────────────────────────────────────
    # Module 5 — k-NN Structural Coherence
    # ─────────────────────────────────────────────────────────────────────────

    def module_knn_coherence(self, latents: torch.Tensor,
                             commands: torch.Tensor) -> dict:
        """
        Tests whether the k nearest neighbours of each sample in latent space
        have similar CAD structure to the query.

        Three structural features compared:
            n_ext      : number of extrusion operations
            seq_len    : total non-EOS tokens
            prim_ratio : fraction of curved (arc/circle) vs linear (line) primitives

        Coherence per feature:
            = 1 - mean_nn_diff / mean_random_diff
            = 0.0 → NN just as different as random (latent space is unstructured)
            = 1.0 → NN always have identical structural feature (perfectly structured)
        """
        k      = self.cfg.k_neighbors
        N      = latents.size(0)
        struct = self._struct_features(commands[:N])          # [N, 3] on CPU
        z_norm = F.normalize(latents.float(), dim=1)          # [N, 512] on CPU

        # Build cosine NN index in chunks to avoid OOM
        nn_idx = []
        chunk  = 500
        for i in range(0, N, chunk):
            z_q  = z_norm[i:i+chunk]             # [c, 512]
            sim  = z_q @ z_norm.T                 # [c, N]
            # Exclude diagonal (self) — approximate for chunk-based computation
            for ii in range(z_q.size(0)):
                sim[ii, i + ii] = -1.0
            nn_idx.append(sim.topk(k, dim=1).indices.cpu())  # [c, k]
        nn_idx = torch.cat(nn_idx, dim=0)         # [N, k]

        rand_idx = torch.randint(0, N, (N, k))    # random baseline

        feat_names   = ['n_ext', 'seq_len', 'prim_ratio']
        coherences   = {}
        nn_diffs_out = {}
        rnd_diffs_out= {}

        for fi, fname in enumerate(feat_names):
            fq = struct[:, fi].unsqueeze(1).expand(-1, k)    # [N, k]

            # NN diff
            f_nn  = struct[nn_idx.reshape(-1), fi].reshape(N, k)
            nn_d  = (fq - f_nn).abs().mean().item()

            # Random diff
            f_rnd = struct[rand_idx.reshape(-1), fi].reshape(N, k)
            rnd_d = (fq - f_rnd).abs().mean().item()

            coh = 1.0 - nn_d / (rnd_d + 1e-6)
            coherences[fname]    = coh
            nn_diffs_out[fname]  = nn_d
            rnd_diffs_out[fname] = rnd_d

        mean_coh = float(np.mean(list(coherences.values())))

        r = {
            'k'              : k,
            'coherences'     : {k: round(v, 4) for k, v in coherences.items()},
            'nn_diffs'       : {k: round(v, 4) for k, v in nn_diffs_out.items()},
            'rand_diffs'     : {k: round(v, 4) for k, v in rnd_diffs_out.items()},
            'mean_coherence' : round(mean_coh, 4),
        }

        status = 'HEALTHY ✓' if mean_coh > 0.50 else '⚠ low'
        print(f"\n── Module 5  k-NN Structural Coherence  (k={k}) ────────────")
        print(f"  {'Feature':<14} {'NN diff':>10} {'Rand diff':>10} {'Coherence':>10}")
        print(f"  {'-'*48}")
        for fname in feat_names:
            print(f"  {fname:<14} {nn_diffs_out[fname]:>10.4f}"
                  f" {rnd_diffs_out[fname]:>10.4f}"
                  f" {coherences[fname]:>10.4f}")
        print(f"  {'-'*48}")
        print(f"  Mean coherence: {mean_coh:.4f}  [{status}]  target >0.50")
        return r

    # ─────────────────────────────────────────────────────────────────────────
    # Module 6 — JEPA Prediction Quality
    # ─────────────────────────────────────────────────────────────────────────

    def module_jepa_quality(self, commands: torch.Tensor,
                            args: torch.Tensor,
                            n: int = 500) -> dict:
        """
        Directly evaluates the JEPA objective at inference time:
        how well does the predictor reconstruct masked token representations?

        predictor_cos: cosine similarity between predictor output and EMA target
                       at masked positions. Directly corresponds to the training loss.
                       Training loss ~0.006 at epoch 300 → cos_sim expected ~0.85+

        Two baselines:
            context_mean_cos: predict the mean of visible context features for
                              all masked positions (trivial context-based baseline)
            zero_cos:         predict zero → cos_sim = 0 by definition

        A good predictor must beat context_mean_cos by a clear margin, otherwise
        the predictor hasn't learned to extrapolate beyond the visible context.
        """
        cmds = commands[:n]
        a    = args[:n]

        pred_cos_list  = []
        base_cos_list  = []
        n_masked_list  = []

        for i in range(0, n, 64):
            c_b = cmds[i:i+64]
            a_b = a[i:i+64]
            B   = c_b.size(0)

            mask = self._random_mask(c_b, ratio=self.cfg.mask_ratio)

            with torch.no_grad():
                ctx_h = self.enc(c_b, a_b, jepa_mask=mask)       # [B, 60, 512]
                tgt_h = self.ema_enc(c_b, a_b, jepa_mask=None)   # [B, 60, 512]

            valid  = (c_b != EOS_IDX)
            masked = mask & valid                                  # [B, 60]

            for b in range(B):
                m_pos = masked[b].nonzero(as_tuple=True)[0]       # indices of masked tokens
                if len(m_pos) == 0:
                    continue

                t    = tgt_h[b, m_pos].float()                    # [n_mask, 512]
                t_n  = F.normalize(t, dim=1)

                # Predictor output for this sample (B=1)
                with torch.no_grad():
                    pred = self.predictor(
                        ctx_h[b:b+1],             # [1, 60, 512]
                        m_pos.unsqueeze(0),        # [1, n_mask]
                    )[0].float()                   # [n_mask, 512]

                pred_n = F.normalize(pred, dim=1)
                cos_p  = (pred_n * t_n).sum(dim=1).mean().item()
                pred_cos_list.append(cos_p)

                # Baseline: broadcast mean of visible context reps to all masked positions
                v_pos    = (~mask[b] & valid[b]).nonzero(as_tuple=True)[0]
                if len(v_pos) > 0:
                    ctx_mean = ctx_h[b, v_pos].float().mean(0, keepdim=True)
                    ctx_mean = F.normalize(ctx_mean.expand(len(m_pos), -1), dim=1)
                    cos_b    = (ctx_mean * t_n).sum(dim=1).mean().item()
                    base_cos_list.append(cos_b)

                n_masked_list.append(len(m_pos))

        pred_cos = np.array(pred_cos_list)
        base_cos = np.array(base_cos_list) if base_cos_list else np.array([0.0])

        r = {
            'predictor_cos_mean' : round(float(pred_cos.mean()), 4),
            'predictor_cos_std'  : round(float(pred_cos.std()),  4),
            'context_mean_cos'   : round(float(base_cos.mean()), 4),
            'lift_over_baseline' : round(float(pred_cos.mean() - base_cos.mean()), 4),
            'mean_n_masked'      : round(float(np.mean(n_masked_list)), 1),
        }

        status = 'HEALTHY ✓' if r['predictor_cos_mean'] > 0.70 else '⚠ low'
        print("\n── Module 6  JEPA Prediction Quality ────────────────────────")
        print(f"  Predictor cos sim  : {r['predictor_cos_mean']:.4f} ± {r['predictor_cos_std']:.4f}"
              f"  [{status}]  target >0.70")
        print(f"  Context-mean base  : {r['context_mean_cos']:.4f}")
        print(f"  Lift over baseline : +{r['lift_over_baseline']:.4f}")
        print(f"  Avg masked tokens  : {r['mean_n_masked']:.1f} per sample")
        return r

    # ─────────────────────────────────────────────────────────────────────────
    # Run all
    # ─────────────────────────────────────────────────────────────────────────

    def run_all(self) -> dict:
        print("\n" + "=" * 68)
        print(f"  CAD-JEPA Encoder Analysis  —  epoch_{self.epoch:04d}.pt")
        print("=" * 68)

        commands, args = self._load_val_samples()
        N = commands.size(0)

        print(f"\nEncoding {N:,} samples (full, no mask)...")
        t0      = time.time()
        latents = self._encode_full(commands, args)
        print(f"Done in {time.time()-t0:.1f}s   latents: {list(latents.shape)}")

        results = {'epoch': self.epoch, 'n_samples': N}

        results['latent_stats']        = self.module_latent_stats(latents)
        results['effective_rank']      = self.module_effective_rank(latents)
        results['masked_consistency']  = self.module_masked_consistency(
            commands, args, n=min(500, N))
        results['token_separation']    = self.module_token_separation(
            commands, args, n=min(1000, N))
        results['knn_coherence']       = self.module_knn_coherence(latents, commands)
        results['jepa_quality']        = self.module_jepa_quality(
            commands, args, n=min(500, N))

        self._print_summary(results)

        # Save JSON
        os.makedirs(self.cfg.output_dir, exist_ok=True)
        out = os.path.join(self.cfg.output_dir, f'analysis_epoch_{self.epoch:04d}.json')
        with open(out, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved → {out}")

        return results

    def _print_summary(self, results: dict) -> None:
        def _get(key, subkey, default=0.0):
            return results.get(key, {}).get(subkey, default)

        rf  = _get('effective_rank',     'rank_frac')
        mc  = _get('masked_consistency', 'full_vs_masked_mean')
        ts  = _get('token_separation',   'sep_ratio')
        kn  = _get('knn_coherence',      'mean_coherence')
        jq  = _get('jepa_quality',       'predictor_cos_mean')

        def status(val, thresh):
            return 'HEALTHY ✓' if val >= thresh else '⚠  LOW  '

        n_healthy = sum([rf >= 0.70, mc >= 0.80, ts >= 2.0, kn >= 0.50, jq >= 0.70])

        print("\n" + "=" * 68)
        print("  SUMMARY")
        print("=" * 68)
        print(f"  Effective rank     : {rf:.3f}   {status(rf, 0.70)}   target >0.70")
        print(f"  Masked consistency : {mc:.3f}   {status(mc, 0.80)}   target >0.80")
        print(f"  Token separation   : {ts:.2f}   {status(ts, 2.00)}   target >2.0")
        print(f"  kNN coherence      : {kn:.3f}   {status(kn, 0.50)}   target >0.50")
        print(f"  JEPA prediction    : {jq:.3f}   {status(jq, 0.70)}   target >0.70")
        print(f"\n  Overall: {n_healthy}/5 healthy")

        if n_healthy == 5:
            print("  → All checks pass. Proceed to build_latent_cache.py → Stage 2.")
        elif n_healthy >= 3:
            print("  → Encoder is usable. Acceptable to proceed to Stage 2.")
            print("  → Check failing modules individually before finalising paper numbers.")
        else:
            print("  → Encoder may be underfit. Investigate failing modules.")
        print("=" * 68)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='CAD-JEPA Encoder Analysis')
    parser.add_argument('--ckpt',        default=AnalysisConfig.ckpt_path,
                        help='Path to Stage 1 checkpoint')
    parser.add_argument('--data-root',   default=AnalysisConfig.data_root)
    parser.add_argument('--output-dir',  default=AnalysisConfig.output_dir)
    parser.add_argument('--n-samples',   type=int, default=2000,
                        help='Val samples to use. -1 = all 8946')
    parser.add_argument('--n-mask-runs', type=int, default=5)
    parser.add_argument('--k',           type=int, default=10)
    parser.add_argument('--batch-size',  type=int, default=256)
    a = parser.parse_args()

    cfg = AnalysisConfig(
        ckpt_path   = a.ckpt,
        data_root   = a.data_root,
        output_dir  = a.output_dir,
        n_samples   = a.n_samples if a.n_samples > 0 else 9000,
        n_mask_runs = a.n_mask_runs,
        k_neighbors = a.k,
        batch_size  = a.batch_size,
    )

    analyzer = EncoderAnalyzer(cfg)
    analyzer.run_all()


if __name__ == '__main__':
    main()