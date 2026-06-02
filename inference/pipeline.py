"""
End-to-End Inference: text -> .STEP
Implement after Stage 2 (bridge) and Stage 3 (decoder) are trained.
"""
import torch
from inference.tokens_to_cadquery import tokens_to_cadquery

class CADJEPAPipeline:
    def __init__(self, bridge_ckpt: str, decoder_ckpt: str, device: str = "cuda"):
        self.device = device
        self.text_bridge = None   # TODO: load from bridge_ckpt
        self.decoder     = None   # TODO: load from decoder_ckpt

    def run(self, prompt: str, out_step: str) -> str:
        raise NotImplementedError("Implement after Stage 2 + 3 are trained")
