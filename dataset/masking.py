import torch
import random

SOL_IDX = 4
EXT_IDX = 5
EOS_IDX = 3

def find_operation_blocks(commands):
    """Returns list of (start, end) inclusive per SOL→Ext block."""
    blocks, start = [], None
    for i, c in enumerate(commands.tolist()):
        if c == SOL_IDX:
            start = i
        elif c == EXT_IDX and start is not None:
            blocks.append((start, i))
            start = None
        elif c == EOS_IDX:
            break
    return blocks

def operation_block_mask(commands):
    """80% op-block, 20% random span. Returns context_mask, target_mask [S] bool."""
    S        = len(commands)
    real_len = (commands != EOS_IDX).sum().item()  # excludes EOS + padding
    
    if random.random() < 0.80:
        blocks = find_operation_blocks(commands)
        if len(blocks) >= 2:
            chosen = random.choice(blocks)          # mask 1 block
            target = torch.zeros(S, dtype=torch.bool)
            target[chosen[0]:chosen[1]+1] = True
            context = torch.zeros(S, dtype=torch.bool)
            context[:real_len] = True
            context &= ~target
            assert not (context & target).any()
            assert not target[real_len:].any()
            return context, target
    
    # fallback: random span
    span_len = random.randint(2, max(2, real_len // 3))
    start    = random.randint(0, max(0, real_len - span_len))
    target   = torch.zeros(S, dtype=torch.bool)
    target[start:start + span_len] = True
    context  = torch.zeros(S, dtype=torch.bool)
    context[:real_len] = True
    context &= ~target
    return context, target