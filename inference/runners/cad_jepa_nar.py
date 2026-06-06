"""
inference/runners/cad_jepa_nar.py
Runner for CAD-JEPA NAR decoder.
Works for ANY checkpoint of the NAR architecture.
"""
import torch
import torch.nn as nn
import sys, os
from ..base_runner import BaseRunner


class CADJEPANARRunner(BaseRunner):

    def setup(self, decoder_ckpt: str, bridge_ckpt: str,
              correction_path: str, n_ext_head_path: str,
              device: str = 'cuda', **kwargs):

        self.device = torch.device(device)

        # Bridge
        from model.text_bridge import TextToLatentBridge
        self.bridge = TextToLatentBridge().to(self.device)
        ckpt_b = torch.load(bridge_ckpt, map_location=self.device)
        key = 'bridge' if 'bridge' in ckpt_b else 'model'
        self.bridge.load_state_dict(ckpt_b[key])
        self.bridge.eval()

        # Norm correction
        corr = torch.load(correction_path, map_location='cpu')
        scale      = corr['norm_scale']
        feat_scale = corr['feat_scale'].to(self.device)
        self.apply_corr = lambda z: z * scale * feat_scale.unsqueeze(0)

        # NAR decoder
        from model.decoder_nar import CADDecoderNAR
        ckpt_d = torch.load(decoder_ckpt, map_location=self.device)
        cfg    = ckpt_d.get('cfg', {})
        self.dec = CADDecoderNAR(
            n_mem = cfg.get('n_mem', 8)
        ).to(self.device)
        self.dec.load_state_dict(ckpt_d['decoder'])
        self.dec.eval()
        print(f"  Decoder: epoch {ckpt_d.get('epoch','?')}  "
              f"cmd_acc={ckpt_d.get('val_cmd_acc', 0):.4f}")

        # n_ext head
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
                cmd_l, arg_l = self.dec(z)
            cmds_raw = [c.item() for c in cmd_l.argmax(-1)[0]]
            args_raw = (arg_l.argmax(-1)[0] - 1).cpu().numpy()
            return self.post_process(cmds_raw, args_raw)
        except Exception:
            return None