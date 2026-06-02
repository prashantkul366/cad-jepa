"""
CAD Semantic Block Parser  (state machine over command types, no ML)

Block types (paper section 3.3):
  SKETCH_EXTRUDE : SOF -> [Line/Arc/Circle]* -> SOL -> SOE -> Extrude
  FILLET_GROUP   : consecutive Fillet ops on a shared edge set
  BOOLEAN        : Cut / Join / Intersect operation
  PATTERN        : PolarArray or LinearArray

Usage:
    parser  = SemanticBlockParser()
    blocks  = parser.parse(token_sequence)      # List[Block]
    masker  = SemanticBlockMasker(0.5)
    vis_ids, msk_ids = masker.sample(blocks)
"""

import math
import random
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple

import torch


class BlockType(Enum):
    SKETCH_EXTRUDE = "sketch_extrude"
    FILLET_GROUP   = "fillet_group"
    BOOLEAN        = "boolean"
    PATTERN        = "pattern"
    OTHER          = "other"


@dataclass
class Block:
    block_type : BlockType
    start_idx  : int
    end_idx    : int

    @property
    def length(self) -> int:
        return self.end_idx - self.start_idx + 1

    def token_indices(self) -> List[int]:
        return list(range(self.start_idx, self.end_idx + 1))


class SemanticBlockParser:
    """
    Parses a [T, 17] token sequence into a list of semantic blocks.
    Command type constants live in cadlib/macro.py — import them in __init__.
    """

    def __init__(self):
        # TODO: import from cadlib.macro
        # e.g.  from cadlib.macro import SOF_IDX, SOL_IDX, SOE_IDX, ...
        # Store as self.CMD_SOF, self.CMD_SOL, etc.
        pass

    def parse(self, tokens: torch.Tensor) -> List[Block]:
        """
        tokens: [T, 17]  column 0 = command type integer
        returns: List[Block] covering the full sequence

        TODO: implement state machine
          State IDLE:
            on CMD_SOF  -> enter IN_SKETCH, record start_idx = current
          State IN_SKETCH:
            on CMD_SOE  -> transition to IN_EXTRUDE
          State IN_EXTRUDE:
            on CMD_SOF or CMD_EOS -> close block as SKETCH_EXTRUDE, back to IDLE
          Fillet / Boolean / Pattern ops -> wrap each as a 1-token block
          Any remaining tokens -> BlockType.OTHER
        """
        pass


class SemanticBlockMasker:

    def __init__(self, mask_ratio: float = 0.5, min_visible: int = 1):
        self.mask_ratio  = mask_ratio
        self.min_visible = min_visible

    def sample(self, blocks: List[Block]) -> Tuple[List[int], List[int]]:
        """
        Randomly choose which blocks to mask.
        Returns flat token index lists: (visible_ids, masked_ids)

        TODO:
          n_mask = ceil(mask_ratio * len(blocks))
          clamp so at least min_visible blocks stay visible
          shuffle block indices, split into masked / visible
          flatten each group to token indices via block.token_indices()
        """
        pass
