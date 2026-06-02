"""
1D CAD Multi-Block Mask Collator
Adapted from multiblock_ijepa_reference.py for 1D CAD sequences.

Differences from I-JEPA:
  - No 2D grid; blocks are [start, end] ranges in a 1D token sequence
  - Block boundaries come from SemanticBlockParser, not spatial sampling
  - Replaces aspect_ratio/scale sampling with semantic block lengths

Used as collate_fn in the DataLoader.
"""

from typing import List, Tuple

import torch

from dataset.masks.semantic_block import SemanticBlockParser, SemanticBlockMasker


class CADMultiBlockMaskCollator:

    def __init__(self, mask_ratio: float = 0.5, min_visible: int = 1):
        self.parser = SemanticBlockParser()
        self.masker = SemanticBlockMasker(mask_ratio, min_visible)

    def __call__(
        self, batch: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        batch: List of [T_i, 17] tensors (variable length from CADDataset)

        returns:
            padded_tokens : [B, T_max, 17]
            context_mask  : [B, T_max] bool  True = visible to context encoder
            target_mask   : [B, T_max] bool  True = positions predictor must predict

        TODO:
          For each seq:
            blocks = self.parser.parse(seq)
            vis_ids, msk_ids = self.masker.sample(blocks)
            build boolean context_mask and target_mask of length T_i
          Pad all to T_max (False for padding positions in both masks)
        """
        pass
