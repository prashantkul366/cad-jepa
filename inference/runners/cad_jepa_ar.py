"""
inference/runners/cad_jepa_ar.py
Runner for CAD-JEPA AR decoder (V2 — the previous autoregressive version).
"""
import torch
import torch.nn as nn
from ..base_runner import BaseRunner


class CADJEPAARRunner(BaseRunner):

    def setup(self, decoder_ckpt: str, bridge_ckpt: str,
              correction_path: str, n_ext_head_path: str,
              device: str = 'cuda', beam_k: int = 5,
              temperature: float = 0.8, **kwargs):

        self.device = torch.device(device)
        self.beam_k = beam_k
        self.temperature = temperature

        from model.text_bridge import TextToLatentBridge
        self.bridge = TextToLatentBridge().to(self.device)
        ckpt_b = torch.load(bridge_ckpt, map_location=self.device)
        key = 'bridge' if 'bridge' in ckpt_b else 'model'
        self.bridge.load_state_dict(ckpt_b[key])
        self.bridge.eval()

        corr = torch.load(correction_path, map_location='cpu')
        scale = corr['norm_scale']
        feat_scale = corr['feat_scale'].to(self.device)
        self.apply_corr = lambda z: z * scale * feat_scale.unsqueeze(0)

        from model.decoder import CADSequenceDecoder  # V2 AR decoder
        ckpt_d = torch.load(decoder_ckpt, map_location=self.device)
        cfg    = ckpt_d.get('cfg', {})
        self.dec = CADSequenceDecoder(
            n_mem = cfg.get('n_mem', 8)
        ).to(self.device)
        self.dec.load_state_dict(ckpt_d['decoder'])
        self.dec.eval()

        self.head = nn.Sequential(
            nn.Linear(512,128), nn.ReLU(), nn.Linear(128,1)
        ).to(self.device)
        self.head.load_state_dict(
            torch.load(n_ext_head_path, map_location=self.device)
        )
        self.head.eval()

    def generate_one(self, uid, text):
        try:
            z = self.apply_corr(self.bridge.encode_text([text], self.device))
            with torch.no_grad():
                cmds, args = self.dec.generate(
                    z, beam_k=self.beam_k,
                    temperature=self.temperature,
                    n_ext_head=self.head
                )
            cmds_raw = [c.item() for c in cmds]
            args_raw = args.cpu().numpy()
            return self.post_process(cmds_raw, args_raw)
        except Exception:
            return None