"""
CAD Multi-Block Mask Collator

Structure mirrors I-JEPA's MaskCollator.__call__:
  - default_collate the batch
  - per-item: parse blocks, sample masks
  - return (commands, args, context_mask, target_mask)

Replaces I-JEPA's 2D spatial block sampling with parse_blocks + sample_mask.
"""

import torch
from torch.utils.data import default_collate
from typing import List, Dict, Tuple

from dataset.masks.semantic_block import parse_blocks, sample_mask


class CADMaskCollator:
    """
    DataLoader collate_fn for CAD-JEPA pretraining.

    Returns:
      commands     : [B, S]       int64
      args         : [B, S, 16]   int64
      context_mask : [B, S]       bool  True = hide from context encoder
      target_mask  : [B, S]       bool  True = predictor must predict here
    """

    def __init__(self, mask_ratio: float = 0.5, min_visible: int = 1):
        self.mask_ratio  = mask_ratio
        self.min_visible = min_visible

    def __call__(self, batch: List[Dict]) -> Tuple[torch.Tensor, ...]:
        # Step 1: collate raw batch (mirrors I-JEPA default_collate)
        collated = default_collate(batch)
        commands = collated['command']    # [B, S]
        args     = collated['args']       # [B, S, 16]
        B, S     = commands.shape

        context_mask = torch.zeros(B, S, dtype=torch.bool)
        target_mask  = torch.zeros(B, S, dtype=torch.bool)

        # Step 2: per-item block parsing + mask sampling
        # (mirrors I-JEPA's per-image loop)
        for i in range(B):
            blocks = parse_blocks(commands[i])
            visible_ops, masked_ops = sample_mask(
                blocks, self.mask_ratio, self.min_visible)

            if masked_ops:
                context_mask[i, masked_ops] = True   # hide from context encoder
                target_mask[i,  masked_ops] = True   # predictor predicts these

        return commands, args, context_mask, target_mask