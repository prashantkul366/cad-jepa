"""
inference/runners/text2cad_hf.py
Template for HuggingFace-hosted models (Text2CAD, FlexCAD, etc.)
Fill in the HF model ID and the generation logic once you have access.
"""
from ..base_runner import BaseRunner
import numpy as np


class Text2CADHFRunner(BaseRunner):
    """
    Template for Text2CAD from HuggingFace.

    To use:
      1. Find the HF model ID (e.g. "SadilKhan/Text2CAD")
      2. Fill in load_model() and decode_output()
      3. The rest is handled by generate.py
    """

    def setup(self, hf_model_id: str, device: str = 'cuda', **kwargs):
        self.device = device
        print(f"Loading {hf_model_id} from HuggingFace...")

        # ── Fill this in based on the model's HF card ──────────────────
        # from transformers import AutoModelForCausalLM, AutoTokenizer
        # self.model = AutoModelForCausalLM.from_pretrained(hf_model_id)
        # self.tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
        # self.model.to(device).eval()
        raise NotImplementedError(
            f"Fill in the HF loading code for {hf_model_id}"
        )

    def generate_one(self, uid, text):
        # ── Fill in based on model's inference API ──────────────────────
        # tokens = self.tokenizer(text, return_tensors='pt').to(self.device)
        # output = self.model.generate(**tokens, max_length=256)
        # vec = self.decode_output(output)
        # return np.array(vec, dtype=np.int64) if vec else None
        raise NotImplementedError

    def decode_output(self, output) -> list | None:
        """Convert model output tokens to [N, 17] list."""
        raise NotImplementedError