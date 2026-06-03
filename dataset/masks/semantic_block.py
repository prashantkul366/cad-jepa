"""
CAD Semantic Block Parser

State machine over command type sequence using cadlib.macro constants.

A block = one sketch-extrude feature:
  SOL ... [LINE/ARC/CIRCLE] ... SOL ... [prims] ... EXT
  Everything from the first SOL up to and including EXT = one block.

Example:
  [4,0,0,0,4,2,4,2,5,  4,2,5,  3,3,3]
   └──────block 0─────┘└─block1┘ EOS
"""

import random
from dataclasses import dataclass
from typing import List, Tuple

import torch

LINE_IDX, ARC_IDX, CIRCLE_IDX = 0, 1, 2
EOS_IDX, SOL_IDX, EXT_IDX     = 3, 4, 5
SKETCH_PRIMS = {LINE_IDX, ARC_IDX, CIRCLE_IDX}


@dataclass
class Block:
    start: int
    end:   int

    @property
    def indices(self) -> List[int]:
        return list(range(self.start, self.end + 1))


def parse_blocks(command_seq: torch.Tensor) -> List[Block]:
    """
    Parse command sequence into semantic blocks.
    command_seq: [S] int64 tensor
    """
    cmds        = command_seq.tolist()
    blocks      = []
    block_start = None

    for i, cmd in enumerate(cmds):
        if cmd == EOS_IDX:
            if block_start is not None:
                blocks.append(Block(block_start, i - 1))
                block_start = None
            break

        elif cmd == SOL_IDX:
            if block_start is None:
                block_start = i          # open a new block

        elif cmd == EXT_IDX:
            if block_start is not None:
                # blocks.append(Block(block_start, i))   # EXT closes the block
                blocks.append(Block(block_start, i - 1))   # sketch block ends before EXT
                blocks.append(Block(i, i))  # treat EXT as its own block (helps with short blocks and orphan EXT)
                block_start = None
            else:
                blocks.append(Block(i, i))             # orphan EXT

        elif cmd in SKETCH_PRIMS:
            if block_start is None:
                block_start = i          # orphan primitive, open block

    # Close any unclosed block (sketch without EXT)
    if block_start is not None:
        last = max(block_start,
                   max((i for i, c in enumerate(cmds) if c != EOS_IDX), default=block_start))
        blocks.append(Block(block_start, last))

    # Fallback: too short — treat each op as its own block
    if len(blocks) < 2:
        n = sum(1 for c in cmds if c != EOS_IDX)
        blocks = [Block(i, i) for i in range(n)]

    return blocks


def sample_mask(
    blocks: List[Block],
    mask_ratio: float = 0.5,
    min_visible: int  = 1,
) -> Tuple[List[int], List[int]]:
    """
    Randomly choose blocks to mask.
    Returns (visible_op_indices, masked_op_indices) sorted.
    """
    n      = len(blocks)
    n_mask = max(1, round(mask_ratio * n))
    n_mask = min(n_mask, n - min_visible)

    idx = list(range(n))
    random.shuffle(idx)
    masked_blocks  = set(idx[:n_mask])
    visible_blocks = set(idx[n_mask:])

    visible_ops = sorted(i for b in visible_blocks for i in blocks[b].indices)
    masked_ops  = sorted(i for b in masked_blocks  for i in blocks[b].indices)

    # ADD before the return statement:
    MAX_MASK_RATIO = 0.55
    all_token_count = sum(b.end - b.start + 1 for b in blocks)
    while True:
        masked_count = sum(blocks[b].end - blocks[b].start + 1 for b in masked_blocks)
        if masked_count / max(all_token_count, 1) <= MAX_MASK_RATIO:
            break
        largest = max(masked_blocks, key=lambda b: blocks[b].end - blocks[b].start + 1)
        masked_blocks.discard(largest)
        visible_blocks.add(largest)
        
    return visible_ops, masked_ops