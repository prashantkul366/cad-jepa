"""
trainer/train_decoder.py
Stage 3 — CAD Sequence Decoder Training

Trains CADSequenceDecoder to reconstruct CAD token sequences from latent z*.

CRITICAL: z_targets come from the Stage 1 latent cache — NOT the Stage 2 bridge.
This decouples Stage 2 and Stage 3 and prevents error accumulation.
At inference only, the full chain is: text → bridge → z* → decoder.

Pipeline:
    z_target (from cache) → decoder(teacher forcing) → cmd_logits, args_logits
    loss = CE(cmd_logits, tgt_commands) + CE(args_logits, tgt_args)

Training targets (well-trained decoder):
    val_cmd_acc  > 0.85
    val_args_acc > 0.90

Usage (Colab):
    from trainer.train_decoder import DecoderTrainer, ConfigDecoder
    trainer = DecoderTrainer(ConfigDecoder())
    trainer.run()
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from dataset.cad_dataset import CADDataset
from model.decoder import CADSequenceDecoder, decoder_loss
from utils.schedulers import WarmupCosineSchedule
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ConfigDecoder:
    # ── Paths ─────────────────────────────────────────────────────────────────
    cache_train : str = '/content/drive/MyDrive/cad-jepa-data/latent_cache_train.npy'
    cache_val   : str = '/content/drive/MyDrive/cad-jepa-data/latent_cache_val.npy'
    data_root   : str = '/content'
    # ckpt_dir    : str = '/content/drive/MyDrive/cad-jepa-checkpoints/decoder'
    ckpt_dir        : str   = '/content/drive/MyDrive/cad-jepa-checkpoints/decoder_v2'

    # ── Decoder architecture (must match model/decoder.py defaults) ───────────
    latent_d    : int   = 512
    d_model     : int   = 512
    n_heads     : int   = 8
    n_layers    : int   = 6
    n_mem       : int   = 8      # NEW — multi-token memory     # same
    d_ff        : int   = 2048
    dropout     : float = 0.1
    n_commands  : int   = 6
    args_dim    : int   = 256
    n_args      : int   = 16
    eos_idx     : int   = 3      # EOS_IDX from cadlib/macro.py
    max_len     : int   = 60     # MAX_TOTAL_LEN from cadlib/macro.py

    # ── Training ──────────────────────────────────────────────────────────────
    epochs          : int   = 100
    batch_size      : int   = 128      # smaller than Stage 2 — sequences are long
    lr              : float = 1e-4
    weight_decay    : float = 0.01
    grad_clip       : float = 1.0
    warmup_epochs   : int   = 5
    # label_smoothing : float = 0.1
    label_smoothing : float = 0.05   # was 0.1

    # ── Data ──────────────────────────────────────────────────────────────────
    num_workers   : int  = 4
    augment       : bool = False   # no augmentation for decoder training

    # ── Checkpointing ─────────────────────────────────────────────────────────
    save_every    : int  = 20
    save_best     : bool = True    # save best val_cmd_acc checkpoint


# ──────────────────────────────────────────────────────────────────────────────
# Dataset — CAD sequences paired with their cached latent vectors
# ──────────────────────────────────────────────────────────────────────────────

class CADLatentDataset(Dataset):
    """
    Wraps CADDataset and augments each sample with its cached JEPA latent.

    Returns per sample:
        z_target : [512]       float — frozen encoder latent for this model
        command  : [60]        long  — command type sequence (EOS-padded)
        args     : [60, 16]    long  — args sequence (-1 for PAD)
        id       : str         — UID e.g. '0067/00675619'

    The cache is a dict {uid: np.float32[512]} loaded from .npy.
    Every UID in CADDataset should be in the cache (built from same split).
    If a UID is missing, __getitem__ raises KeyError with a clear message.
    """

    def __init__(self, cad_dataset: CADDataset, cache: dict):
        self.ds    = cad_dataset
        self.cache = cache

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> dict:
        sample = self.ds[idx]
        uid    = sample['id']

        if uid not in self.cache:
            raise KeyError(
                f"UID '{uid}' not found in latent cache.\n"
                "Re-run scripts/build_latent_cache.py to rebuild the cache."
            )

        z = torch.from_numpy(self.cache[uid].copy()).float()   # [512]

        return {
            'z_target': z,
            'command' : sample['command'],   # [60]
            'args'    : sample['args'],      # [60, 16]
            'id'      : uid,
        }


def _load_cache(path: str) -> dict:
    """Load .npy latent cache dict. Raises FileNotFoundError if missing."""
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Latent cache not found: {path}\n"
            "Run scripts/build_latent_cache.py first (needs Stage 1 epoch_0300.pt)."
        )
    return np.load(path, allow_pickle=True).item()


def _make_cad_cfg(cfg: ConfigDecoder) -> SimpleNamespace:
    """Build a config object that CADDataset expects."""
    return SimpleNamespace(
        data_root     = cfg.data_root,
        augment       = cfg.augment,
        max_n_loops   = 6,
        max_n_curves  = 6,
        max_total_len = cfg.max_len,
        batch_size    = cfg.batch_size,
        num_workers   = cfg.num_workers,
    )


# ──────────────────────────────────────────────────────────────────────────────
# DecoderTrainer
# ──────────────────────────────────────────────────────────────────────────────

class DecoderTrainer:
    """
    Stage 3 trainer: maps cached latent z* → CAD token sequences.

    Args:
        cfg    : ConfigDecoder
        run_tag: appended to checkpoint filenames (e.g. for ablation runs)
    """

    def __init__(self, cfg: ConfigDecoder, run_tag: str = ''):
        self.cfg     = cfg
        self.run_tag = run_tag
        self.device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # ── Data ──────────────────────────────────────────────────────────────
        self.train_loader, self.val_loader = self._build_loaders()

        # ── Model ─────────────────────────────────────────────────────────────
        self.decoder = CADSequenceDecoder(
            latent_d    = cfg.latent_d,
            d_model     = cfg.d_model,
            n_heads     = cfg.n_heads,
            n_layers    = cfg.n_layers,
            d_ff        = cfg.d_ff,
            dropout     = cfg.dropout,
            n_commands  = cfg.n_commands,
            args_dim    = cfg.args_dim,
            n_args      = cfg.n_args,
            eos_idx     = cfg.eos_idx,
            max_len     = cfg.max_len,
            n_mem       = cfg.n_mem,
        ).to(self.device)

        self.decoder.param_summary()

        # ── Optimizer — single group (unlike Stage 2, no frozen sub-model) ────
        self.optimizer = torch.optim.AdamW(
            self.decoder.parameters(),
            lr           = cfg.lr,
            weight_decay = cfg.weight_decay,
        )

        # ── Scheduler: warmup then cosine decay ───────────────────────────────
        total_steps  = cfg.epochs * len(self.train_loader)
        warmup_steps = cfg.warmup_epochs * len(self.train_loader)
        # self.scheduler = WarmupCosineSchedule(
        #     self.optimizer,
        #     warmup_steps = warmup_steps,
        #     t_total      = total_steps,
        # )

        
        _warmup = LinearLR(self.optimizer, start_factor=0.1,
                        total_iters=warmup_steps)
        _cosine = CosineAnnealingLR(self.optimizer,
                                    T_max=total_steps - warmup_steps,
                                    eta_min=1e-6)
        self.scheduler = SequentialLR(self.optimizer,
                                    schedulers=[_warmup, _cosine],
                                    milestones=[warmup_steps])

        # ── State ─────────────────────────────────────────────────────────────
        self.best_cmd_acc  = 0.0
        self.best_epoch    = 0
        self.start_epoch   = 1

        os.makedirs(cfg.ckpt_dir, exist_ok=True)
        self._print_header()

    # ── Setup ─────────────────────────────────────────────────────────────────

    def _build_loaders(self) -> tuple[DataLoader, DataLoader]:
        cfg     = self.cfg
        cad_cfg = _make_cad_cfg(cfg)

        print("Loading latent caches ...")
        cache_train = _load_cache(cfg.cache_train)
        cache_val   = _load_cache(cfg.cache_val)
        print(f"  train cache: {len(cache_train):,} UIDs")
        print(f"  val   cache: {len(cache_val):,}   UIDs")

        train_ds = CADLatentDataset(CADDataset('train', cad_cfg), cache_train)
        val_ds   = CADLatentDataset(CADDataset('validation', cad_cfg), cache_val)

        train_loader = DataLoader(
            train_ds,
            batch_size  = cfg.batch_size,
            shuffle     = True,
            num_workers = cfg.num_workers,
            pin_memory  = True,
            drop_last   = True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size  = cfg.batch_size * 2,
            shuffle     = False,
            num_workers = cfg.num_workers,
            pin_memory  = True,
            drop_last   = False,
        )
        return train_loader, val_loader

    def _print_header(self) -> None:
        cfg = self.cfg
        n   = sum(p.numel() for p in self.decoder.parameters())
        print("=" * 68)
        print("Stage 3 — CAD Sequence Decoder Training")
        print("=" * 68)
        print(f"Device          : {self.device}")
        print(f"Decoder params  : {n:,}")
        print(f"Train samples   : {len(self.train_loader.dataset):,}")
        print(f"Val   samples   : {len(self.val_loader.dataset):,}")
        print(f"Batches/epoch   : {len(self.train_loader)}")
        print(f"Epochs          : {cfg.epochs}")
        print(f"Batch size      : {cfg.batch_size}")
        print(f"LR              : {cfg.lr}  (warmup {cfg.warmup_epochs} epochs)")
        print(f"Label smoothing : {cfg.label_smoothing}")
        print(f"Checkpoint dir  : {cfg.ckpt_dir}")
        print("=" * 68)

    # ── Training epoch ────────────────────────────────────────────────────────

    def train_epoch(self, epoch: int) -> dict:
        """One full training epoch with teacher forcing."""
        self.decoder.train()
        cfg = self.cfg

        total_loss = total_cmd = total_args = 0.0
        total_cmd_acc = total_args_acc = 0.0
        n_batches = 0
        t0 = time.time()

        for batch in self.train_loader:
            z_target = batch['z_target'].to(self.device)    # [B, 512]
            commands = batch['command'].to(self.device)     # [B, 60]
            args     = batch['args'].to(self.device)        # [B, 60, 16]

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                cmd_logits, args_logits = self.decoder(z_target, commands, args)
                loss, metrics = decoder_loss(
                    cmd_logits, args_logits,
                    commands, args,
                    eos_idx      = cfg.eos_idx,
                    label_smooth = cfg.label_smoothing,
                )

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.decoder.parameters(), cfg.grad_clip
            )
            self.optimizer.step()
            self.scheduler.step()

            total_loss     += metrics['loss_total']
            total_cmd      += metrics['loss_cmd']
            total_args     += metrics['loss_args']
            total_cmd_acc  += metrics['acc_cmd']
            total_args_acc += metrics['acc_args']
            n_batches      += 1

        elapsed = time.time() - t0
        return {
            'loss'      : total_loss     / n_batches,
            'loss_cmd'  : total_cmd      / n_batches,
            'loss_args' : total_args     / n_batches,
            'cmd_acc'   : total_cmd_acc  / n_batches,
            'args_acc'  : total_args_acc / n_batches,
            'lr'        : self.optimizer.param_groups[0]['lr'],
            'time_s'    : elapsed,
        }

    # ── Validation ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def validate(self) -> dict:
        """
        Compute val loss, cmd_acc, args_acc, and structural validity rate.

        structural_valid_rate: fraction of teacher-forced outputs where
            predicted command sequence has at least one EXT token,
            starts with SOL, and EOS appears within MAX_TOTAL_LEN.
        This is a cheap proxy for IR before full CadQuery evaluation.
        """
        self.decoder.eval()
        cfg = self.cfg

        total_loss = total_cmd = total_args = 0.0
        total_cmd_acc = total_args_acc = 0.0
        n_struct_valid = n_total = 0
        n_batches = 0

        for batch in self.val_loader:
            z_target = batch['z_target'].to(self.device)
            commands = batch['command'].to(self.device)
            args     = batch['args'].to(self.device)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                cmd_logits, args_logits = self.decoder(z_target, commands, args)
                loss, metrics = decoder_loss(
                    cmd_logits, args_logits,
                    commands, args,
                    eos_idx      = cfg.eos_idx,
                    label_smooth = cfg.label_smoothing,
                )

            total_loss     += metrics['loss_total']
            total_cmd      += metrics['loss_cmd']
            total_args     += metrics['loss_args']
            total_cmd_acc  += metrics['acc_cmd']
            total_args_acc += metrics['acc_args']
            n_batches      += 1

            # Structural validity: check predicted cmd sequence (greedy)
            pred_cmds = cmd_logits.float().argmax(dim=-1)   # [B, L]
            n_struct_valid += _count_struct_valid(pred_cmds, cfg.eos_idx).item()
            n_total        += pred_cmds.size(0)

        return {
            'val_loss'      : total_loss     / n_batches,
            'val_loss_cmd'  : total_cmd      / n_batches,
            'val_loss_args' : total_args     / n_batches,
            'val_cmd_acc'   : total_cmd_acc  / n_batches,
            'val_args_acc'  : total_args_acc / n_batches,
            'val_struct_ok' : n_struct_valid / max(n_total, 1),
        }

    # ── Checkpointing ─────────────────────────────────────────────────────────

    def _ckpt_path(self, tag: str) -> str:
        prefix = f"{self.run_tag}_" if self.run_tag else ""
        return os.path.join(self.cfg.ckpt_dir, f"{prefix}{tag}.pt")

    def save_checkpoint(self, epoch: int, val_metrics: dict) -> None:
        torch.save({
            'epoch'        : epoch,
            'val_cmd_acc'  : val_metrics['val_cmd_acc'],
            'val_args_acc' : val_metrics['val_args_acc'],
            'val_loss'     : val_metrics['val_loss'],
            'decoder'      : self.decoder.state_dict(),
            'optimizer'    : self.optimizer.state_dict(),
            'scheduler'    : self.scheduler.state_dict(),
            'cfg'          : self.cfg.__dict__,
        }, self._ckpt_path(f"epoch_{epoch:04d}"))

    # def save_best(self, epoch: int, val_metrics: dict) -> None:
    #     torch.save({
    #         'epoch'        : epoch,
    #         'val_cmd_acc'  : val_metrics['val_cmd_acc'],
    #         'val_args_acc' : val_metrics['val_args_acc'],
    #         'val_loss'     : val_metrics['val_loss'],
    #         'decoder'      : self.decoder.state_dict(),
    #         'cfg'          : self.cfg.__dict__,
    #     }, self._ckpt_path("best"))
    #     print(f"  ✓ New best saved  "
    #           f"(cmd_acc={val_metrics['val_cmd_acc']:.4f}  "
    #           f"args_acc={val_metrics['val_args_acc']:.4f}  "
    #           f"epoch={epoch})")

    def save_best(self, epoch: int, val_metrics: dict) -> None:
        torch.save({
            'epoch'        : epoch,
            'val_cmd_acc'  : val_metrics['val_cmd_acc'],
            'val_args_acc' : val_metrics['val_args_acc'],
            'val_loss'     : val_metrics['val_loss'],
            'decoder'      : self.decoder.state_dict(),
            'optimizer'    : self.optimizer.state_dict(),
            'scheduler'    : self.scheduler.state_dict(),
            'cfg'          : self.cfg.__dict__,
        }, self._ckpt_path("best"))
        print(f"  ✓ New best saved  "
              f"(cmd_acc={val_metrics['val_cmd_acc']:.4f}  "
              f"args_acc={val_metrics['val_args_acc']:.4f}  "
              f"epoch={epoch})")

    def load_checkpoint(self, path: str) -> None:
        """Resume training from a checkpoint."""
        ckpt = torch.load(path, map_location=self.device)
        self.decoder.load_state_dict(ckpt['decoder'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.scheduler.load_state_dict(ckpt['scheduler'])
        self.start_epoch   = ckpt['epoch'] + 1
        self.best_cmd_acc  = ckpt.get('val_cmd_acc', 0.0)
        print(f"Resumed from epoch {ckpt['epoch']}  "
              f"val_cmd_acc={self.best_cmd_acc:.4f}")

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self) -> dict:
        """Full training loop. Returns final val metrics."""
        cfg         = self.cfg
        val_metrics = {}

        print(f"\n{'='*68}")
        print(f"Training starts at epoch {self.start_epoch}/{cfg.epochs}")
        print(f"{'='*68}")

        for epoch in range(self.start_epoch, cfg.epochs + 1):

            # ── Train ──────────────────────────────────────────────────────────
            tr = self.train_epoch(epoch)

            # ── Validate ────────────────────────────────────────────────────────
            val_metrics = self.validate()

            # ── Log ─────────────────────────────────────────────────────────────
            print(
                f"Epoch {epoch:>4d}/{cfg.epochs} | "
                f"loss={tr['loss']:.4f} "
                f"(cmd={tr['loss_cmd']:.4f} args={tr['loss_args']:.4f}) | "
                f"train cmd_acc={tr['cmd_acc']:.4f} args_acc={tr['args_acc']:.4f} | "
                f"val cmd_acc={val_metrics['val_cmd_acc']:.4f} "
                f"args_acc={val_metrics['val_args_acc']:.4f} "
                f"struct={val_metrics['val_struct_ok']:.3f} | "
                f"lr={tr['lr']:.2e}"
            )

            # ── Best checkpoint ────────────────────────────────────────────────
            if cfg.save_best and val_metrics['val_cmd_acc'] > self.best_cmd_acc:
                self.best_cmd_acc = val_metrics['val_cmd_acc']
                self.best_epoch   = epoch
                self.save_best(epoch, val_metrics)

            # ── Periodic checkpoint ────────────────────────────────────────────
            if epoch % cfg.save_every == 0:
                self.save_checkpoint(epoch, val_metrics)
                print(f"  Saved: epoch_{epoch:04d}.pt")

        # ── Final summary ──────────────────────────────────────────────────────
        print(f"\n{'='*68}")
        print(f"Training complete.")
        print(f"Best val_cmd_acc  : {self.best_cmd_acc:.4f}  (epoch {self.best_epoch})")
        _print_health_assessment(self.best_cmd_acc, val_metrics)
        print()
        print("Next steps:")
        print("  1. Run full CD/IR evaluation: python evaluation/evaluate_gen_torch.py")
        print("  2. Chain with bridge for end-to-end: python inference/pipeline.py")
        print(f"{'='*68}")

        return {
            'best_cmd_acc' : self.best_cmd_acc,
            'best_epoch'   : self.best_epoch,
            'final_metrics': val_metrics,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Structural validity check (cheap proxy for IR during training)
# ──────────────────────────────────────────────────────────────────────────────

def _count_struct_valid(pred_cmds: torch.Tensor, eos_idx: int) -> torch.Tensor:
    """
    Count sequences in pred_cmds [B, L] that pass structural heuristics:
        1. First token is SOL (idx=4)
        2. At least one EXT token (idx=5) before first EOS
        3. EOS appears somewhere in the sequence

    This is a cheap proxy for CadQuery validity — use IR metric for final eval.

    Returns: scalar tensor (count of valid sequences in batch)
    """
    SOL_IDX = 4
    EXT_IDX = 5

    B = pred_cmds.size(0)

    starts_with_sol = (pred_cmds[:, 0] == SOL_IDX)                     # [B]
    has_eos         = (pred_cmds == eos_idx).any(dim=1)                 # [B]
    has_ext         = (pred_cmds == EXT_IDX).any(dim=1)                 # [B]

    valid = starts_with_sol & has_eos & has_ext
    return valid.sum()


# ──────────────────────────────────────────────────────────────────────────────
# Health assessment
# ──────────────────────────────────────────────────────────────────────────────

def _print_health_assessment(best_cmd_acc: float, val_metrics: dict) -> None:
    args_acc   = val_metrics.get('val_args_acc', 0)
    struct_ok  = val_metrics.get('val_struct_ok', 0)

    print()
    if best_cmd_acc >= 0.85 and args_acc >= 0.90:
        print("  Status   : GOOD — decoder is well trained.")
        print("  Action   : Run full CD/IR evaluation on test split.")
    elif best_cmd_acc >= 0.75:
        print("  Status   : MARGINAL — decoder may be undertrained.")
        print("  Actions  : Increase epochs to 150, or reduce label_smoothing to 0.05.")
    else:
        print("  Status   : UNDERTRAINED — cmd_acc too low.")
        print("  Actions to try (in order):")
        print("    1. Increase epochs to 150")
        print("    2. Reduce batch_size to 64 (more gradient updates per epoch)")
        print("    3. Lower LR warmup to 3 epochs")
        print("    4. Check latent cache was built with augment=False + eval() mode")

    if struct_ok < 0.5:
        print(f"\n  ⚠ struct_ok={struct_ok:.3f} is low — many sequences lack SOL/EXT/EOS.")
        print("    Likely cause: decoder generating near-random tokens.")
        print("    Confirm val_cmd_acc is rising consistently before epoch 20.")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='CAD-JEPA Stage 3 Decoder Training')
    parser.add_argument('--epochs',      type=int,   default=None,
                        help='Override ConfigDecoder.epochs')
    parser.add_argument('--batch-size',  type=int,   default=None,
                        help='Override ConfigDecoder.batch_size')
    parser.add_argument('--lr',          type=float, default=None,
                        help='Override ConfigDecoder.lr')
    parser.add_argument('--resume',      type=str,   default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--run-tag',     type=str,   default='',
                        help='Tag appended to checkpoint filenames')
    args = parser.parse_args()

    cfg = ConfigDecoder()
    if args.epochs     is not None: cfg.epochs     = args.epochs
    if args.batch_size is not None: cfg.batch_size = args.batch_size
    if args.lr         is not None: cfg.lr         = args.lr

    trainer = DecoderTrainer(cfg, run_tag=args.run_tag)
    if args.resume:
        trainer.load_checkpoint(args.resume)
    trainer.run()