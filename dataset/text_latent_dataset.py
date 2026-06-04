"""
dataset/text_latent_dataset.py
Stage 2 — Text + Latent Cache Dataset

Pairs text descriptions with pre-computed CAD-JEPA latent vectors.
No CAD geometry loading at runtime — the encoder ran once offline (build_latent_cache.py).

Sample schema (per __getitem__):
    {
        'text'    : str          — raw text description
        'z_target': tensor[512]  — frozen encoder latent for this model
        'uid'     : str          — model UID  e.g. '0067/00675619'
        'level'   : str          — annotation level: 'beginner'/'intermediate'/'expert'
    }

Text2CAD annotation format expected (JSON):
    {
        "0067/00675619": {
            "beginner"    : "A simple cylindrical shape",
            "intermediate": "A cylinder with a smaller cylinder on top",
            "expert"      : "Two coaxial cylinders, outer 80mm, inner 40mm, height 20mm"
        },
        ...
    }

Label efficiency experiment:
    FRACTIONS = [0.10, 0.25, 0.50, 0.75, 1.00]
    Subsets are NESTED: 0.25 subset ⊂ 0.50 subset ⊂ 1.00 subset.
    Always pass the same seed= across fraction runs for reproducibility.
"""

from __future__ import annotations

import json
import random
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

ALL_LEVELS = ['beginner', 'intermediate', 'expert']


# ──────────────────────────────────────────────────────────────────────────────
# TextLatentDataset
# ──────────────────────────────────────────────────────────────────────────────

class TextLatentDataset(Dataset):
    """
    Stage 2 dataset: maps each (uid, level) pair to (text, z_target).

    Args:
        cache_path        : Path to .npy latent cache — dict {uid: np.array[512]}
                            Built by scripts/build_latent_cache.py after Stage 1.
        annotations_path  : Path to Text2CAD JSON — dict {uid: {level: text}}
        split_path        : Path to train_val_test_split.json (optional).
                            When provided, only UIDs in `phase` split are used.
                            Prevents train/val leakage.
        phase             : 'train' or 'validation'  (NOT 'val')
        label_fraction    : Fraction of UNIQUE UIDs to include (0.0–1.0).
                            Used for label efficiency ablation.
                            Nested: fraction 0.25 is always a subset of 0.50.
        levels            : Annotation levels to include. Default: all 3 levels.
                            Each level becomes an independent sample per UID.
        seed              : Random seed for reproducible UID subsampling.
    """

    def __init__(
        self,
        cache_path       : str,
        annotations_path : str,
        split_path       : Optional[str] = None,
        phase            : str           = 'train',
        label_fraction   : float         = 1.0,
        levels           : Optional[list[str]] = None,
        seed             : int           = 42,
    ):
        assert phase in ('train', 'validation', 'test'), \
            f"phase must be 'train', 'validation', or 'test' — got '{phase}'"
        assert 0.0 < label_fraction <= 1.0, \
            f"label_fraction must be in (0, 1] — got {label_fraction}"

        self.levels          = levels or ALL_LEVELS
        self.label_fraction  = label_fraction
        self.phase           = phase
        self.seed            = seed

        # ── Load latent cache ─────────────────────────────────────────────────
        print(f"[TextLatentDataset] Loading latent cache from {cache_path} ...")
        self.latent_cache = _load_latent_cache(cache_path)
        print(f"[TextLatentDataset] Cache size: {len(self.latent_cache):,} UIDs")

        # ── Load annotations ──────────────────────────────────────────────────
        print(f"[TextLatentDataset] Loading annotations from {annotations_path} ...")
        self.annotations = _load_annotations(annotations_path)
        print(f"[TextLatentDataset] Annotations: {len(self.annotations):,} UIDs")

        # ── Compute valid UID intersection ────────────────────────────────────
        valid_uids = self._intersect_uids(split_path, phase)
        print(f"[TextLatentDataset] Valid UIDs after intersection: {len(valid_uids):,}")

        # ── Apply label fraction (nested subsetting) ──────────────────────────
        selected_uids = _subsample_uids(valid_uids, label_fraction, seed)
        print(f"[TextLatentDataset] UIDs after fraction {label_fraction:.2f}: "
              f"{len(selected_uids):,}")

        # ── Build flat sample list ────────────────────────────────────────────
        # Each entry: (uid, level, text_string)
        self.samples = _build_samples(selected_uids, self.annotations, self.levels)
        print(f"[TextLatentDataset] Total samples (uid × levels): {len(self.samples):,}")
        print(f"[TextLatentDataset] Levels used: {self.levels}")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _intersect_uids(self, split_path: Optional[str], phase: str) -> list[str]:
        """Return UIDs present in cache ∩ annotations ∩ phase split (if given)."""
        cache_uids = set(self.latent_cache.keys())
        annot_uids = set(self.annotations.keys())
        valid      = cache_uids & annot_uids

        if split_path is not None:
            split_uids = _load_split_uids(split_path, phase)
            before = len(valid)
            valid  = valid & split_uids
            print(f"[TextLatentDataset] Split filter ({phase}): "
                  f"{before:,} → {len(valid):,} UIDs")

        missing_annot = len(cache_uids) - len(cache_uids & annot_uids)
        if missing_annot > 0:
            warnings.warn(
                f"{missing_annot:,} UIDs in cache have no annotations — skipped.",
                UserWarning
            )

        if len(valid) == 0:
            raise RuntimeError(
                "TextLatentDataset: zero valid UIDs after intersection. "
                "Check cache_path, annotations_path, and split_path."
            )

        return sorted(valid)   # sorted for reproducibility

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        uid, level, text = self.samples[idx]
        z = torch.from_numpy(self.latent_cache[uid].copy()).float()  # [512]
        return {
            'text'    : text,
            'z_target': z,
            'uid'     : uid,
            'level'   : level,
        }

    # ── Accessors ─────────────────────────────────────────────────────────────

    @property
    def uid_list(self) -> list[str]:
        """Sorted list of unique UIDs in this dataset split (deduped across levels)."""
        return sorted({uid for uid, _, _ in self.samples})

    @property
    def n_uids(self) -> int:
        return len(self.uid_list)

    # ── Statistics ────────────────────────────────────────────────────────────

    def stats(self) -> None:
        """Print dataset summary."""
        uids = self.uid_list
        print("=" * 55)
        print("TextLatentDataset — statistics")
        print("=" * 55)
        print(f"  Phase          : {self.phase}")
        print(f"  Label fraction : {self.label_fraction:.2f}")
        print(f"  Levels         : {self.levels}")
        print(f"  Unique UIDs    : {len(uids):,}")
        print(f"  Total samples  : {len(self.samples):,}  "
              f"({len(self.samples) // max(len(uids), 1)}× expansion)")
        print(f"  Latent dim     : {next(iter(self.latent_cache.values())).shape[0]}")

        # Level breakdown
        level_counts = {lvl: 0 for lvl in ALL_LEVELS}
        for _, lvl, _ in self.samples:
            level_counts[lvl] = level_counts.get(lvl, 0) + 1
        for lvl, cnt in level_counts.items():
            if cnt > 0:
                print(f"    {lvl:>15s} : {cnt:,}")
        print("=" * 55)

    # ── Convenience constructors ──────────────────────────────────────────────

    @classmethod
    def for_label_efficiency(
        cls,
        cache_path       : str,
        annotations_path : str,
        split_path       : str,
        fraction         : float,
        seed             : int = 42,
    ) -> 'TextLatentDataset':
        """
        Factory for label efficiency ablation.
        Always uses full level set and train phase.

        Usage:
            for frac in [0.10, 0.25, 0.50, 0.75, 1.00]:
                ds = TextLatentDataset.for_label_efficiency(
                    cache_path, annot_path, split_path, fraction=frac)
        """
        return cls(
            cache_path       = cache_path,
            annotations_path = annotations_path,
            split_path       = split_path,
            phase            = 'train',
            label_fraction   = fraction,
            levels           = ALL_LEVELS,
            seed             = seed,
        )


# ──────────────────────────────────────────────────────────────────────────────
# DataLoader factory
# ──────────────────────────────────────────────────────────────────────────────

def make_stage2_loaders(
    cache_path_train     : str,
    cache_path_val       : str,
    annotations_path     : str,
    split_path           : str,
    batch_size           : int   = 512,
    num_workers          : int   = 4,
    label_fraction       : float = 1.0,
    seed                 : int   = 42,
) -> tuple[DataLoader, DataLoader]:
    """
    Build train + val DataLoaders for Stage 2.

    Returns:
        train_loader, val_loader

    Usage:
        train_loader, val_loader = make_stage2_loaders(
            cache_path_train = '/content/drive/MyDrive/cad-jepa-data/latent_cache_train.npy',
            cache_path_val   = '/content/drive/MyDrive/cad-jepa-data/latent_cache_val.npy',
            annotations_path = '/content/drive/MyDrive/cad-jepa-data/text2cad_annotations.json',
            split_path       = '/content/train_val_test_split.json',
        )
    """
    train_ds = TextLatentDataset(
        cache_path       = cache_path_train,
        annotations_path = annotations_path,
        split_path       = split_path,
        phase            = 'train',
        label_fraction   = label_fraction,
        seed             = seed,
    )

    val_ds = TextLatentDataset(
        cache_path       = cache_path_val,
        annotations_path = annotations_path,
        split_path       = split_path,
        phase            = 'validation',   # NOT 'val'
        label_fraction   = 1.0,            # always eval on full val set
        seed             = seed,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = num_workers,
        pin_memory  = True,
        drop_last   = True,
        collate_fn  = _collate_text_latent,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size  = batch_size * 2,    # larger — no backward
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = True,
        drop_last   = False,
        collate_fn  = _collate_text_latent,
    )

    return train_loader, val_loader


# ──────────────────────────────────────────────────────────────────────────────
# Collate
# ──────────────────────────────────────────────────────────────────────────────

def _collate_text_latent(batch: list[dict]) -> dict:
    """
    Custom collate: stack z_targets, keep text/uid/level as lists.
    Default PyTorch collate would also work, but this is explicit.
    """
    return {
        'text'    : [s['text']     for s in batch],   # list[str]      len B
        'z_target': torch.stack([s['z_target'] for s in batch]),  # [B, 512]
        'uid'     : [s['uid']      for s in batch],   # list[str]      len B
        'level'   : [s['level']    for s in batch],   # list[str]      len B
    }


# ──────────────────────────────────────────────────────────────────────────────
# Private helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_latent_cache(cache_path: str) -> dict:
    """Load .npy latent cache → dict {uid: np.array[512]}."""
    path = Path(cache_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Latent cache not found: {cache_path}\n"
            "Run scripts/build_latent_cache.py first (needs Stage 1 epoch_0300.pt)."
        )
    cache = np.load(cache_path, allow_pickle=True).item()
    if not isinstance(cache, dict):
        raise ValueError(f"Expected dict in .npy, got {type(cache)}")
    return cache


def _load_annotations(annotations_path: str) -> dict:
    """
    Load text annotations JSON.
    Handles two formats:
      A) {uid: {level: text}}                         — Text2CAD native format
      B) {uid: text}                                  — flat (single level)
    Returns always as format A (wrapping B under level 'description').
    """
    path = Path(annotations_path)
    if not path.exists():
        raise FileNotFoundError(f"Annotations file not found: {annotations_path}")

    with open(path, 'r', encoding='utf-8') as f:
        raw = json.load(f)

    # Detect format
    first_val = next(iter(raw.values()))
    if isinstance(first_val, str):
        # Flat format — wrap under synthetic level 'description'
        warnings.warn(
            "Annotations appear to be flat (uid → text). "
            "Wrapping under level 'description'. "
            "For full 3× data, use Text2CAD multi-level format.",
            UserWarning
        )
        return {uid: {'description': text} for uid, text in raw.items()}
    elif isinstance(first_val, dict):
        return raw
    else:
        raise ValueError(f"Unrecognized annotation format: {type(first_val)}")


def _load_split_uids(split_path: str, phase: str) -> set[str]:
    """Load UIDs for a given phase from train_val_test_split.json."""
    with open(split_path, 'r') as f:
        split = json.load(f)
    if phase not in split:
        raise KeyError(
            f"Phase '{phase}' not found in split file. "
            f"Available: {list(split.keys())}"
        )
    return set(split[phase])


def _subsample_uids(
    uids           : list[str],
    label_fraction : float,
    seed           : int,
) -> list[str]:
    """
    Return a reproducible nested subset of UIDs.

    Nested property:
        subsample(uids, 0.25, seed) ⊂ subsample(uids, 0.50, seed) ⊂ ...
    Guaranteed because we shuffle once with fixed seed and take first N.
    """
    if label_fraction == 1.0:
        return uids

    rng = random.Random(seed)
    shuffled = uids[:]
    rng.shuffle(shuffled)

    n = max(1, int(len(shuffled) * label_fraction))
    return shuffled[:n]


def _build_samples(
    uids        : list[str],
    annotations : dict,
    levels      : list[str],
) -> list[tuple[str, str, str]]:
    """
    Expand uid list to (uid, level, text) triplets.
    Skips (uid, level) pairs where the level is absent in annotations.

    Returns list of (uid, level, text_string).
    """
    samples = []
    for uid in uids:
        annot = annotations.get(uid, {})
        for level in levels:
            text = annot.get(level, None)
            if text is None:
                continue
            if not isinstance(text, str) or len(text.strip()) == 0:
                continue
            samples.append((uid, level, text.strip()))
    return samples


# ──────────────────────────────────────────────────────────────────────────────
# Smoke test  (run with synthetic data — no real files needed)
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import tempfile, os

    print("Running smoke test with synthetic data...\n")

    # ── Build synthetic cache ─────────────────────────────────────────────────
    uids = [f"{i:04d}/{i:08d}" for i in range(200)]
    cache = {uid: np.random.randn(512).astype(np.float32) for uid in uids}

    # ── Build synthetic annotations ───────────────────────────────────────────
    annotations = {
        uid: {
            'beginner'    : f"Simple shape {i}",
            'intermediate': f"A more complex shape {i} with features",
            'expert'      : f"Precise shape {i}: 10mm radius, extruded 20mm",
        }
        for i, uid in enumerate(uids)
    }

    # ── Build synthetic split ─────────────────────────────────────────────────
    split = {
        'train'     : uids[:160],
        'validation': uids[160:180],
        'test'      : uids[180:],
    }

    with tempfile.TemporaryDirectory() as tmp:
        cache_path = os.path.join(tmp, 'cache.npy')
        annot_path = os.path.join(tmp, 'annotations.json')
        split_path = os.path.join(tmp, 'split.json')

        np.save(cache_path, cache, allow_pickle=True)
        with open(annot_path, 'w') as f:
            json.dump(annotations, f)
        with open(split_path, 'w') as f:
            json.dump(split, f)

        # ── Test 1: Full train dataset ────────────────────────────────────────
        print("[Test 1] Full train dataset")
        ds = TextLatentDataset(
            cache_path       = cache_path,
            annotations_path = annot_path,
            split_path       = split_path,
            phase            = 'train',
            label_fraction   = 1.0,
        )
        ds.stats()
        assert len(ds) == 160 * 3, f"Expected 480 samples, got {len(ds)}"

        sample = ds[0]
        assert sample['z_target'].shape == (512,), f"z_target shape wrong: {sample['z_target'].shape}"
        assert isinstance(sample['text'], str)
        assert sample['level'] in ALL_LEVELS
        print(f"  Sample 0: uid={sample['uid']}, level={sample['level']}, "
              f"text='{sample['text'][:40]}...'\n")

        # ── Test 2: Label fraction (25%) — nested ─────────────────────────────
        print("[Test 2] Label fraction = 0.25")
        ds_25  = TextLatentDataset(cache_path, annot_path, split_path,
                                   label_fraction=0.25, seed=42)
        ds_50  = TextLatentDataset(cache_path, annot_path, split_path,
                                   label_fraction=0.50, seed=42)
        ds_100 = TextLatentDataset(cache_path, annot_path, split_path,
                                   label_fraction=1.00, seed=42)

        uids_25  = set(ds_25.uid_list)
        uids_50  = set(ds_50.uid_list)
        uids_100 = set(ds_100.uid_list)

        assert uids_25 <= uids_50,  "25% subset must be subset of 50%"
        assert uids_50 <= uids_100, "50% subset must be subset of 100%"
        print(f"  25%: {ds_25.n_uids} UIDs → {len(ds_25)} samples")
        print(f"  50%: {ds_50.n_uids} UIDs → {len(ds_50)} samples")
        print(f"  100%: {ds_100.n_uids} UIDs → {len(ds_100)} samples")
        print(f"  Nested subset property: PASS\n")

        # ── Test 3: Validation split ──────────────────────────────────────────
        print("[Test 3] Validation split")
        ds_val = TextLatentDataset(
            cache_path       = cache_path,
            annotations_path = annot_path,
            split_path       = split_path,
            phase            = 'validation',
        )
        assert ds_val.n_uids == 20, f"Expected 20 val UIDs, got {ds_val.n_uids}"
        print(f"  Val UIDs: {ds_val.n_uids}, samples: {len(ds_val)}\n")

        # ── Test 4: DataLoader with collate ───────────────────────────────────
        print("[Test 4] DataLoader + collate_fn")
        loader = DataLoader(ds, batch_size=32, shuffle=True,
                            collate_fn=_collate_text_latent)
        batch  = next(iter(loader))

        assert isinstance(batch['text'], list), "text should be list[str]"
        assert batch['z_target'].shape == (32, 512)
        assert batch['z_target'].dtype == torch.float32
        print(f"  Batch text[0] : '{batch['text'][0][:50]}'")
        print(f"  z_target shape: {batch['z_target'].shape}")
        print(f"  z_target dtype: {batch['z_target'].dtype}\n")

        # ── Test 5: Single-level mode ─────────────────────────────────────────
        print("[Test 5] Single level: beginner only")
        ds_beg = TextLatentDataset(
            cache_path       = cache_path,
            annotations_path = annot_path,
            split_path       = split_path,
            levels           = ['beginner'],
        )
        assert len(ds_beg) == 160, f"Expected 160, got {len(ds_beg)}"
        assert all(lvl == 'beginner' for _, lvl, _ in ds_beg.samples)
        print(f"  Samples: {len(ds_beg)} (160 UIDs × 1 level) — PASS\n")

    print("All smoke tests passed.")