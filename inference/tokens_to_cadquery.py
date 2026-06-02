"""
Rule-Based Token Sequence -> CadQuery Python  (no ML)
Implement after cadlib/macro.py constants are understood.
"""
import torch

def dequantize(val: int, min_val: float, max_val: float, bins: int = 256) -> float:
    return min_val + (val / (bins - 1)) * (max_val - min_val)

def tokens_to_cadquery(token_sequence: torch.Tensor) -> str:
    """tokens [T, 17] -> CadQuery Python string.  TODO: implement."""
    return 'import cadquery as cq\nresult = cq.Workplane("XY")  # TODO'
