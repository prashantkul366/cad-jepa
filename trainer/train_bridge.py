"""
trainer/train_bridge.py
Stage 2 — Text-to-Latent Bridge Training

Trains TextToLatentBridge to map text descriptions → CAD-JEPA latent space.
The JEPA encoder is NEVER loaded here — z_targets come from the pre-built cache.

Pipeline:
    text → bridge(tokenize + CLIP + MLP) → z_pred
    loss = 0.9 * MSE(z_pred, z_target) + 0.1 * (1 - cos_sim(z_pred, z_target))

Key invariants:
    - Encoder is FROZEN and not present in this trainer
    - z_targets are loaded from disk cache (latent_cache_{train,val}.npy)
    - BF16 autocast (no GradScaler — BF16 doesn't need it)
    - Two AdamW param groups: projector (1e-4) and CLIP finetune (1e-5)
    - Best checkpoint tracked by val cosine similarity
    - Label efficiency experiment: trains independently at each fraction

Target metrics:
    val_cos_sim > 0.82  → bridge well trained
    val_cos_sim < 0.75  → undertrained; increase epochs or unfreeze more CLIP layers

Usage (Colab):
    from trainer.train_bridge import BridgeTrainer, ConfigBridge
    cfg = ConfigBridge()
    trainer = BridgeTrainer(cfg)
    trainer.run()

Label efficiency:
    from trainer.train_bridge import run_label_efficiency, ConfigBridge
    run_label_efficiency(ConfigBridge(), fractions=[0.10, 0.25, 0.50, 0.75, 1.00])
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset.text_latent_dataset import (
    TextLatentDataset,
    _collate_text_latent,
    ALL_LEVELS,
)
from model.text_bridge import TextToLatentBridge, bridge_loss
from utils.schedulers import WarmupCosineSchedule


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ConfigBridge:
    # ── Paths ─────────────────────────────────────────────────────────────────
    cache_train      : str = '/content/drive/MyDrive/cad-jepa-data/latent_cache_train.npy'
    cache_val        : str = '/content/drive/MyDrive/cad-jepa-data/latent_cache_val.npy'
    annotations_path : str = '/content/drive/MyDrive/cad-jepa-data/text2cad_annotations.json'
    split_path       : str = '/content/train_val_test_split.json'
    ckpt_dir         : str = '/content/drive/MyDrive/cad-jepa-checkpoints/bridge'

    # ── CLIP + bridge architecture ────────────────────────────────────────────
    clip_model   : str = 'openai/clip-vit-base-patch32'
    jepa_d       : int = 512     # must match Stage 1 d_model
    proj_hidden  : int = 1024
    dropout      : float = 0.1
    n_freeze     : int = 10      # freeze first 10/12 CLIP transformer layers

    # ── Training ──────────────────────────────────────────────────────────────
    epochs       : int   = 50
    batch_size   : int   = 512
    lr_proj      : float = 1e-4   # MLP projector learning rate
    lr_clip      : float = 1e-5   # CLIP finetune learning rate (10× lower)
    weight_decay : float = 0.01
    grad_clip    : float = 1.0
    warmup_epochs: int   = 5

    # ── Loss weights ──────────────────────────────────────────────────────────
    mse_w : float = 0.9
    cos_w : float = 0.1

    # ── Data ──────────────────────────────────────────────────────────────────
    levels          : list = field(default_factory=lambda: ALL_LEVELS)
    label_fraction  : float = 1.0
    num_workers     : int   = 4
    seed            : int   = 42

    # ── Checkpointing ─────────────────────────────────────────────────────────
    save_every      : int   = 10   # save checkpoint every N epochs
    save_best       : bool  = True # always save best val_cos_sim checkpoint


# ──────────────────────────────────────────────────────────────────────────────
# BridgeTrainer
# ──────────────────────────────────────────────────────────────────────────────

class BridgeTrainer:
    """
    Stage 2 trainer.

    Args:
        cfg           : ConfigBridge
        label_fraction: Override cfg.label_fraction (used by label efficiency runner)
        run_tag       : String appended to checkpoint filenames (e.g. 'frac0.25')
    """

    def __init__(
        self,
        cfg            : ConfigBridge,
        label_fraction : Optional[float] = None,
        run_tag        : str             = '',
    ):
        self.cfg     = cfg
        self.run_tag = run_tag
        self.device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        effective_fraction = label_fraction if label_fraction is not None else cfg.label_fraction

        # ── Data ──────────────────────────────────────────────────────────────
        self.train_loader, self.val_loader = self._build_loaders(effective_fraction)

        # ── Model ─────────────────────────────────────────────────────────────
        self.bridge = TextToLatentBridge(
            clip_model  = cfg.clip_model,
            jepa_d      = cfg.jepa_d,
            proj_hidden = cfg.proj_hidden,
            dropout     = cfg.dropout,
            n_freeze    = cfg.n_freeze,
        ).to(self.device)

        self.bridge.verify_freeze()
        self.bridge.param_summary()

        # ── Optimizer ─────────────────────────────────────────────────────────
        self.optimizer = torch.optim.AdamW(
            self.bridge.param_groups(
                lr_proj      = cfg.lr_proj,
                lr_clip      = cfg.lr_clip,
                weight_decay = cfg.weight_decay,
            )
        )

        # ── Scheduler: warmup then cosine decay ───────────────────────────────
        total_steps   = cfg.epochs * len(self.train_loader)
        warmup_steps  = cfg.warmup_epochs * len(self.train_loader)
        self.scheduler = WarmupCosineSchedule(
            self.optimizer,
            warmup_steps = warmup_steps,
            t_total      = total_steps,
        )

        # ── State ─────────────────────────────────────────────────────────────
        self.best_cos_sim    = 0.0
        self.best_epoch      = 0
        self.start_epoch     = 1

        # ── Checkpoint dir ────────────────────────────────────────────────────
        os.makedirs(cfg.ckpt_dir, exist_ok=True)

        self._print_header(effective_fraction)

    # ── Setup ─────────────────────────────────────────────────────────────────

    def _build_loaders(self, label_fraction: float):
        train_ds = TextLatentDataset(
            cache_path       = self.cfg.cache_train,
            annotations_path = self.cfg.annotations_path,
            split_path       = self.cfg.split_path,
            phase            = 'train',
            label_fraction   = label_fraction,
            levels           = self.cfg.levels,
            seed             = self.cfg.seed,
        )
        val_ds = TextLatentDataset(
            cache_path       = self.cfg.cache_val,
            annotations_path = self.cfg.annotations_path,
            split_path       = self.cfg.split_path,
            phase            = 'validation',
            label_fraction   = 1.0,    # always validate on full val set
            levels           = self.cfg.levels,
            seed             = self.cfg.seed,
        )

        train_loader = DataLoader(
            train_ds,
            batch_size  = self.cfg.batch_size,
            shuffle     = True,
            num_workers = self.cfg.num_workers,
            pin_memory  = True,
            drop_last   = True,
            collate_fn  = _collate_text_latent,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size  = self.cfg.batch_size * 2,
            shuffle     = False,
            num_workers = self.cfg.num_workers,
            pin_memory  = True,
            drop_last   = False,
            collate_fn  = _collate_text_latent,
        )
        return train_loader, val_loader

    def _print_header(self, fraction: float) -> None:
        cfg = self.cfg
        print("=" * 68)
        print("Stage 2 — Text-to-Latent Bridge Training")
        print("=" * 68)
        print(f"Device         : {self.device}")
        print(f"CLIP model     : {cfg.clip_model}")
        print(f"Label fraction : {fraction:.2f}")
        print(f"Train samples  : {len(self.train_loader.dataset):,}")
        print(f"Val   samples  : {len(self.val_loader.dataset):,}")
        print(f"Batches/epoch  : {len(self.train_loader)}")
        print(f"Epochs         : {cfg.epochs}")
        print(f"LR (proj/clip) : {cfg.lr_proj} / {cfg.lr_clip}")
        print(f"Warmup epochs  : {cfg.warmup_epochs}")
        print(f"Loss weights   : MSE={cfg.mse_w}  cos={cfg.cos_w}")
        print(f"Checkpoint dir : {cfg.ckpt_dir}")
        print("=" * 68)

    # ── Training epoch ────────────────────────────────────────────────────────

    def train_epoch(self, epoch: int) -> dict:
        """One full training epoch. Returns aggregated metrics."""
        self.bridge.train()
        cfg = self.cfg

        total_loss = total_mse = total_cos = 0.0
        n_batches  = 0
        t0         = time.time()

        for batch in self.train_loader:
            texts    = batch['text']                          # list[str], len B
            z_target = batch['z_target'].to(self.device)     # [B, 512]

            # Tokenise on CPU then move to device
            tokens = self.bridge.tokenize(texts, self.device)

            # Forward (BF16 — no GradScaler needed)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                z_pred        = self.bridge(tokens)           # [B, 512]
                loss, metrics = bridge_loss(
                    z_pred, z_target.to(torch.bfloat16),
                    mse_w=cfg.mse_w, cos_w=cfg.cos_w,
                )

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()

            # Gradient clipping — collect all trainable params
            trainable = [p for p in self.bridge.parameters() if p.requires_grad]
            torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip)

            self.optimizer.step()
            self.scheduler.step()

            total_loss += metrics['loss_total']
            total_mse  += metrics['loss_mse']
            total_cos  += metrics['loss_cos']
            n_batches  += 1

        elapsed = time.time() - t0
        return {
            'loss'    : total_loss / n_batches,
            'loss_mse': total_mse  / n_batches,
            'loss_cos': total_cos  / n_batches,
            'lr_proj' : self.optimizer.param_groups[0]['lr'],
            'lr_clip' : self.optimizer.param_groups[1]['lr'],
            'time_s'  : elapsed,
        }

    # ── Validation ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def validate(self) -> dict:
        """
        Compute mean cosine similarity and MSE on the full validation set.
        Returns metrics dict.

        Target: val_cos_sim > 0.82
        Alarm:  val_cos_sim < 0.75 → bridge undertrained
        """
        self.bridge.eval()

        cos_sims   = []
        mse_vals   = []
        import torch.nn.functional as F

        for batch in self.val_loader:
            texts    = batch['text']
            z_target = batch['z_target'].to(self.device)

            tokens = self.bridge.tokenize(texts, self.device)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                z_pred = self.bridge(tokens).float()

            cos = F.cosine_similarity(z_pred, z_target, dim=-1)  # [B]
            mse = F.mse_loss(z_pred, z_target, reduction='none').mean(dim=1)  # [B]

            cos_sims.append(cos.cpu())
            mse_vals.append(mse.cpu())

        cos_sims = torch.cat(cos_sims)
        mse_vals = torch.cat(mse_vals)

        return {
            'val_cos_sim' : cos_sims.mean().item(),
            'val_cos_std' : cos_sims.std().item(),
            'val_mse'     : mse_vals.mean().item(),
        }

    # ── Checkpointing ─────────────────────────────────────────────────────────

    def _ckpt_path(self, tag: str) -> str:
        prefix = f"{self.run_tag}_" if self.run_tag else ""
        return os.path.join(self.cfg.ckpt_dir, f"{prefix}{tag}.pt")

    def save_checkpoint(self, epoch: int, val_metrics: dict) -> None:
        payload = {
            'epoch'       : epoch,
            'val_cos_sim' : val_metrics['val_cos_sim'],
            'val_mse'     : val_metrics['val_mse'],
            'bridge'      : self.bridge.state_dict(),
            'optimizer'   : self.optimizer.state_dict(),
            'scheduler'   : self.scheduler.state_dict(),
            'cfg'         : self.cfg.__dict__,
        }
        torch.save(payload, self._ckpt_path(f"epoch_{epoch:04d}"))

    def save_best(self, epoch: int, val_metrics: dict) -> None:
        payload = {
            'epoch'       : epoch,
            'val_cos_sim' : val_metrics['val_cos_sim'],
            'val_mse'     : val_metrics['val_mse'],
            'bridge'      : self.bridge.state_dict(),
            'cfg'         : self.cfg.__dict__,
        }
        torch.save(payload, self._ckpt_path("best"))
        print(f"  ✓ New best saved  (val_cos_sim={val_metrics['val_cos_sim']:.4f}  epoch={epoch})")

    def load_checkpoint(self, path: str) -> None:
        """Resume training from checkpoint."""
        ckpt = torch.load(path, map_location=self.device)
        self.bridge.load_state_dict(ckpt['bridge'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.scheduler.load_state_dict(ckpt['scheduler'])
        self.start_epoch  = ckpt['epoch'] + 1
        self.best_cos_sim = ckpt.get('val_cos_sim', 0.0)
        print(f"Resumed from epoch {ckpt['epoch']}  val_cos_sim={self.best_cos_sim:.4f}")

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self) -> dict:
        """
        Full training loop.
        Returns final val metrics dict for use by label efficiency runner.
        """
        cfg         = self.cfg
        val_metrics = {}

        print(f"\n{'='*68}")
        print(f"Training starts at epoch {self.start_epoch}/{cfg.epochs}")
        print(f"{'='*68}")

        for epoch in range(self.start_epoch, cfg.epochs + 1):

            # ── Train ──────────────────────────────────────────────────────────
            train_metrics = self.train_epoch(epoch)

            # ── Validate (every epoch — val set is small, ~27k samples) ────────
            val_metrics = self.validate()

            # ── Log ────────────────────────────────────────────────────────────
            cos_sim   = val_metrics['val_cos_sim']
            alarm_tag = _health_tag(cos_sim)
            print(
                f"Epoch {epoch:>4d}/{cfg.epochs} | "
                f"loss={train_metrics['loss']:.4f} "
                f"(mse={train_metrics['loss_mse']:.4f} "
                f"cos={train_metrics['loss_cos']:.4f}) | "
                f"val_cos={cos_sim:.4f}±{val_metrics['val_cos_std']:.4f} | "
                f"lr={train_metrics['lr_proj']:.2e} | "
                f"{alarm_tag}"
            )

            # ── Best checkpoint ────────────────────────────────────────────────
            if cfg.save_best and cos_sim > self.best_cos_sim:
                self.best_cos_sim = cos_sim
                self.best_epoch   = epoch
                self.save_best(epoch, val_metrics)

            # ── Periodic checkpoint ────────────────────────────────────────────
            if epoch % cfg.save_every == 0:
                self.save_checkpoint(epoch, val_metrics)
                print(f"  Saved: epoch_{epoch:04d}.pt")

        # ── Final summary ──────────────────────────────────────────────────────
        print(f"\n{'='*68}")
        print(f"Training complete.")
        print(f"Best val_cos_sim : {self.best_cos_sim:.4f}  (epoch {self.best_epoch})")
        _print_health_assessment(self.best_cos_sim, cfg)
        print(f"{'='*68}")

        return {
            'best_cos_sim' : self.best_cos_sim,
            'best_epoch'   : self.best_epoch,
            'final_metrics': val_metrics,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Label efficiency experiment
# ──────────────────────────────────────────────────────────────────────────────

def run_label_efficiency(
    cfg       : ConfigBridge,
    fractions : list[float] = None,
    seed      : int         = 42,
) -> dict[float, dict]:
    """
    Train a separate bridge at each label fraction and collect results.
    Used to generate the headline "4× annotation efficiency" result.

    Args:
        cfg      : ConfigBridge (base config; fraction is overridden per run)
        fractions: Label fractions to evaluate  (default: [0.10, 0.25, 0.50, 0.75, 1.00])
        seed     : Random seed (same across all fractions for nested subsets)

    Returns:
        results: {fraction: {'best_cos_sim': float, 'best_epoch': int, ...}}

    Each fraction trains a fresh bridge independently.
    Checkpoints saved to cfg.ckpt_dir/frac{fraction}/best.pt

    Expected output (from paper):
        fraction=0.25 → val_cos_sim ≈ 0.80  (matches Text2CAD @ 100% labels downstream)
        fraction=1.00 → val_cos_sim ≈ 0.86
    """
    if fractions is None:
        fractions = [0.10, 0.25, 0.50, 0.75, 1.00]

    results = {}

    print("\n" + "=" * 68)
    print("Label Efficiency Experiment")
    print(f"Fractions: {fractions}")
    print("=" * 68 + "\n")

    for frac in fractions:
        tag = f"frac{frac:.2f}".replace('.', 'p')   # e.g. 'frac0p25'

        # Per-fraction checkpoint subdirectory
        frac_cfg          = ConfigBridge(**cfg.__dict__)
        frac_cfg.ckpt_dir = os.path.join(cfg.ckpt_dir, tag)
        frac_cfg.seed     = seed

        print(f"\n{'─'*68}")
        print(f"Fraction {frac:.2f}  →  {frac_cfg.ckpt_dir}")
        print(f"{'─'*68}")

        trainer = BridgeTrainer(frac_cfg, label_fraction=frac, run_tag=tag)
        result  = trainer.run()
        results[frac] = result

    # ── Summary table ──────────────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("Label Efficiency — Summary")
    print(f"{'Fraction':>10}  {'UIDs':>10}  {'val_cos_sim':>12}  {'best_epoch':>12}")
    print("-" * 50)
    for frac, res in sorted(results.items()):
        print(f"{frac:>10.2f}  {'—':>10}  {res['best_cos_sim']:>12.4f}  {res['best_epoch']:>12d}")
    print("=" * 68)

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Inference helper  (used by Stage 3 pipeline)
# ──────────────────────────────────────────────────────────────────────────────

def load_bridge_for_inference(
    ckpt_path : str,
    device    : torch.device,
) -> TextToLatentBridge:
    """
    Load best bridge checkpoint for Stage 3 inference pipeline.

    Usage:
        bridge = load_bridge_for_inference('bridge/best.pt', device)
        z_star = bridge.encode_text(["a cylinder with a hole"], device)
    """
    ckpt   = torch.load(ckpt_path, map_location=device)
    cfg    = ckpt.get('cfg', {})

    bridge = TextToLatentBridge(
        clip_model  = cfg.get('clip_model',  'openai/clip-vit-base-patch32'),
        jepa_d      = cfg.get('jepa_d',      512),
        proj_hidden = cfg.get('proj_hidden', 1024),
        dropout     = cfg.get('dropout',     0.1),
        n_freeze    = cfg.get('n_freeze',    10),
    )
    bridge.load_state_dict(ckpt['bridge'])
    bridge.to(device).eval()

    print(f"[Bridge] Loaded: epoch={ckpt['epoch']}  val_cos_sim={ckpt['val_cos_sim']:.4f}")
    return bridge


# ──────────────────────────────────────────────────────────────────────────────
# Health assessment helpers
# ──────────────────────────────────────────────────────────────────────────────

def _health_tag(cos_sim: float) -> str:
    if cos_sim >= 0.82:
        return "✓ good"
    if cos_sim >= 0.75:
        return "~ ok"
    return "⚠ undertrained"


def _print_health_assessment(best_cos_sim: float, cfg: ConfigBridge) -> None:
    print()
    if best_cos_sim >= 0.82:
        print("  Status  : GOOD — bridge is well trained.")
        print("  Action  : Proceed to Stage 3 (train decoder).")
    elif best_cos_sim >= 0.75:
        print("  Status  : MARGINAL — bridge may be undertrained.")
        print("  Action  : Try increasing epochs to 75 or 100,")
        print("            or reduce n_freeze from 10 to 8.")
    else:
        print("  Status  : UNDERTRAINED — val_cos_sim too low.")
        print("  Actions to try (in order):")
        print("    1. Increase epochs (try 100)")
        print("    2. Reduce n_freeze from 10 to 8 (unfreeze 4 CLIP layers)")
        print("    3. Reduce lr_clip to 5e-6 (more conservative CLIP finetune)")
        print("    4. Check that latent cache was built with eval() + no_grad()")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='CAD-JEPA Stage 2 Bridge Training')
    parser.add_argument('--label-efficiency', action='store_true',
                        help='Run full label efficiency experiment across all fractions')
    parser.add_argument('--fraction', type=float, default=1.0,
                        help='Label fraction for single run (default: 1.0)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override ConfigBridge.epochs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    args = parser.parse_args()

    cfg = ConfigBridge()

    if args.epochs is not None:
        cfg.epochs = args.epochs

    if args.label_efficiency:
        run_label_efficiency(cfg)
    else:
        trainer = BridgeTrainer(cfg, label_fraction=args.fraction)
        if args.resume:
            trainer.load_checkpoint(args.resume)
        trainer.run()