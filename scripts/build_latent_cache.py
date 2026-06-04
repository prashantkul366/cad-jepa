"""
scripts/build_latent_cache.py
One-time script — run once after Stage 1 epoch 300 checkpoint is ready.

Encodes all train and validation CAD sequences with the frozen JEPA encoder
and saves latent vectors to disk. Stage 2 training loads these instead of
running the encoder at every iteration.

Output files:
    latent_cache_train.npy  — dict {uid: np.float32[512]},  ~330 MB, 161k UIDs
    latent_cache_val.npy    — dict {uid: np.float32[512]},  ~18  MB,   9k UIDs

Runtime: ~30 min on A100 (train), ~2 min (val)

Usage (Colab):
    !python scripts/build_latent_cache.py
    # or with a specific checkpoint:
    !python scripts/build_latent_cache.py --ckpt /path/to/epoch_0300.pt
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config.configJEPA import ConfigJEPA
from dataset.cad_dataset import CADDataset
from model.jepa_encoder import CADJEPAEncoder


# ──────────────────────────────────────────────────────────────────────────────
# Defaults
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_CKPT  = '/content/drive/MyDrive/cad-jepa-checkpoints/pretrain/epoch_0300.pt'
DEFAULT_OUT   = '/content/drive/MyDrive/cad-jepa-data'
DEFAULT_DATA  = '/content'


# ──────────────────────────────────────────────────────────────────────────────
# Core
# ──────────────────────────────────────────────────────────────────────────────

def build_cache(
    enc         : CADJEPAEncoder,
    cfg         : ConfigJEPA,
    phase       : str,
    batch_size  : int = 256,
    num_workers : int = 4,
    device      : torch.device = None,
) -> dict:
    """
    Encode all sequences in `phase` split and return {uid: z[512]} dict.

    Args:
        enc       : frozen CADJEPAEncoder (already in eval mode)
        cfg       : ConfigJEPA (data_root, augment, etc.)
        phase     : 'train' or 'validation'  (NOT 'val')
        batch_size: inference batch size
        device    : target device
    """
    if device is None:
        device = next(enc.parameters()).device

    ds = CADDataset(phase, cfg)
    loader = DataLoader(
        ds,
        batch_size  = batch_size,
        shuffle     = False,       # preserve order for reproducibility
        num_workers = num_workers,
        pin_memory  = True,
        drop_last   = False,
    )

    cache     = {}
    n_batches = len(loader)
    t0        = time.time()

    print(f"\n[{phase}] {len(ds):,} samples  |  {n_batches} batches")

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Encoding {phase}", ncols=90):
            cmds = batch['command'].to(device)  # [B, 60]
            args = batch['args'].to(device)     # [B, 60, 16]

            # BF16 for speed — matches Stage 1 training dtype
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                z = enc.encode_mean(cmds, args)  # [B, 512]

            z_np = z.float().cpu().numpy()       # convert to fp32 before saving

            for uid, z_vec in zip(batch['id'], z_np):
                cache[uid] = z_vec.astype(np.float32)

    elapsed = time.time() - t0
    print(f"  Done in {elapsed/60:.1f} min  |  {len(cache):,} UIDs cached")

    return cache


def verify_cache(cache: dict, phase: str) -> None:
    """Sanity-check the cache: shapes, value range, z-norm."""
    uids = list(cache.keys())
    sample_z = np.stack([cache[u] for u in uids[:1000]])  # first 1k

    z_norm = np.linalg.norm(sample_z, axis=1)
    z_std  = sample_z.std(axis=0).mean()

    print(f"\n[{phase}] Cache verification")
    print(f"  UIDs         : {len(cache):,}")
    print(f"  Latent dim   : {sample_z.shape[1]}")
    print(f"  dtype        : {sample_z.dtype}")
    print(f"  z_norm mean  : {z_norm.mean():.3f}  (expect ~√512 ≈ 22.6 for unit std)")
    print(f"  z_norm std   : {z_norm.std():.3f}")
    print(f"  z feature std: {z_std:.3f}  (expect ≈ 1.0 after encoder output_norm)")
    print(f"  z min / max  : {sample_z.min():.3f} / {sample_z.max():.3f}")

    # Warn if representations look collapsed
    if z_std < 0.3:
        print(f"  ⚠ WARNING: z_std={z_std:.3f} is very low — representations may be collapsed.")
        print(f"    Check that Stage 1 training completed without collapse (rank > 0.70).")


def save_cache(cache: dict, out_path: str) -> None:
    """Save {uid: array} dict to .npy with allow_pickle."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, cache, allow_pickle=True)
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"  Saved → {out_path}  ({size_mb:.0f} MB)")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt',        default=DEFAULT_CKPT,
                        help='Path to Stage 1 encoder checkpoint (epoch_0300.pt)')
    parser.add_argument('--out-dir',     default=DEFAULT_OUT,
                        help='Output directory for .npy cache files')
    parser.add_argument('--data-root',   default=DEFAULT_DATA,
                        help='data_root for CADDataset (/content by default)')
    parser.add_argument('--batch-size',  type=int, default=256)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--phases',      nargs='+', default=['train', 'validation'],
                        choices=['train', 'validation', 'test'],
                        help='Which splits to encode (default: train validation)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── Load frozen encoder ───────────────────────────────────────────────────
    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(
            f"Checkpoint not found: {args.ckpt}\n"
            "Wait for Stage 1 epoch 300 to complete, or pass a different --ckpt."
        )

    print(f"Loading encoder from {args.ckpt} ...")
    cfg = ConfigJEPA()
    cfg.data_root = args.data_root
    cfg.augment   = False          # CRITICAL: no augmentation for cache building

    enc  = CADJEPAEncoder(cfg).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    enc.load_state_dict(ckpt['encoder'])
    enc.eval()
    for p in enc.parameters():
        p.requires_grad_(False)

    n_params = sum(p.numel() for p in enc.parameters())
    print(f"Encoder loaded: epoch={ckpt['epoch']}  params={n_params:,}")

    # ── Build and save caches ─────────────────────────────────────────────────
    print(f"\nOutput dir: {args.out_dir}")
    print("=" * 60)

    for phase in args.phases:
        fname    = f"latent_cache_{phase}.npy"
        out_path = os.path.join(args.out_dir, fname)

        if os.path.exists(out_path):
            print(f"\n[{phase}] Cache already exists at {out_path}  — skipping.")
            print("  Delete it manually to rebuild.")
            continue

        cache = build_cache(
            enc, cfg, phase,
            batch_size  = args.batch_size,
            num_workers = args.num_workers,
            device      = device,
        )
        verify_cache(cache, phase)
        save_cache(cache, out_path)

    print("\n" + "=" * 60)
    print("Cache building complete. Proceed to Stage 2 training:")
    print("  python trainer/train_bridge.py")


if __name__ == '__main__':
    main()