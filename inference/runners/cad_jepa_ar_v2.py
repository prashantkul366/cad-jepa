"""
inference/runners/cad_jepa_ar_v2.py

Inference runner for CADDecoderARV2.
Plugs into inference/generate.py RUNNERS dict.

Usage:
    python -m inference.generate \
        --runner     cad_jepa_ar_v2 \
        --exp_name   "AR-v2-100pct" \
        --decoder_ckpt /content/drive/MyDrive/cad-jepa-checkpoints/decoder_ar_v2/best.pt \
        --output     /content/drive/MyDrive/cad-jepa-results/pred_ar_v2_100pct.pkl
"""

import sys, os, json, torch, numpy as np, pickle
sys.path.insert(0, '/content/cad-jepa')

from inference.base_runner import BaseRunner
from model.decoder_ar_v2  import CADDecoderARV2, EOS, PAD_VAL, MAX_LEN, N_ARGS
from model.text_bridge     import TextToLatentBridge

import torch.nn as nn


class CADJEPAARv2Runner(BaseRunner):
    """
    Full text → CAD sequence pipeline using the AR V2 decoder.

    Pipeline:
        text → bridge → z* (corrected) → ARV2 decoder → CAD vec
    """

    def setup(self):
        device = self.device

        # ── Bridge (Stage 2) ───────────────────────────────────────────────
        bridge_path = self.cfg.get(
            'bridge_ckpt',
            '/content/drive/MyDrive/cad-jepa-checkpoints/bridge/best.pt'
        )
        from model.text_bridge import TextToLatentBridge
        self.bridge = TextToLatentBridge().to(device)
        ckpt_b = torch.load(bridge_path, map_location=device)
        self.bridge.load_state_dict(ckpt_b['bridge'])   # key is 'bridge' not 'model'
        self.bridge.eval()

        # ── Norm correction (empirical) ────────────────────────────────────
        # Compute from latent cache at runtime
        cache_path = self.cfg.get(
            'cache_train',
            '/content/drive/MyDrive/cad-jepa-data/latent_cache_train.npy'
        )
        cache = np.load(cache_path, allow_pickle=True).item()
        enc_norms = [np.linalg.norm(cache[u])
                     for u in list(cache.keys())[:500]]
        enc_mean  = float(np.mean(enc_norms))

        annot_path = self.cfg.get(
            'annotations',
            '/content/drive/MyDrive/cad-jepa-data/text2cad_annotations.json'
        )
        with open(annot_path) as f:
            annot = json.load(f)

        bridge_norms = []
        with torch.no_grad():
            for uid in list(cache.keys())[:200]:
                if uid not in annot:
                    continue
                text = (annot[uid].get('intermediate') or
                        annot[uid].get('beginner'))
                if not text:
                    continue
                z = self.bridge.encode_text([text], device)
                bridge_norms.append(z.norm().item())

        bridge_mean       = float(np.mean(bridge_norms))
        self.corr_scale   = enc_mean / bridge_mean
        print(f"Norm correction scale: {self.corr_scale:.4f} "
              f"(enc={enc_mean:.3f}, bridge={bridge_mean:.3f})")

        # ── n_ext head ─────────────────────────────────────────────────────
        head_path = self.cfg.get(
            'n_ext_head',
            '/content/drive/MyDrive/cad-jepa-checkpoints/n_ext_head_v2.pt'
        )
        self.n_ext_head = nn.Sequential(
            nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, 1)
        ).to(device)
        self.n_ext_head.load_state_dict(torch.load(head_path, map_location=device))
        self.n_ext_head.eval()

        # ── AR V2 Decoder (Stage 3) ────────────────────────────────────────
        dec_ckpt = torch.load(self.cfg['decoder_ckpt'], map_location=device)
        cfg_d    = dec_ckpt.get('cfg', {})
        self.decoder = CADDecoderARV2(
            latent_d  = cfg_d.get('latent_d',  512),
            d_model   = cfg_d.get('d_model',   256),
            n_heads   = cfg_d.get('n_heads',   8),
            n_layers  = cfg_d.get('n_layers',  8),
            d_ff      = cfg_d.get('d_ff',      1024),
            n_mem     = cfg_d.get('n_mem',     8),
            d_arg_emb = cfg_d.get('d_arg_emb', 32),
        ).to(device)
        self.decoder.load_state_dict(dec_ckpt['decoder'])
        self.decoder.eval()
        print(f"AR V2 Decoder loaded | epoch={dec_ckpt.get('epoch','?')} "
              f"| params={self.decoder.n_params:,}")

    @torch.no_grad()
    def predict(self, uid: str, text: str) -> np.ndarray:
        """
        text → np.int64[N, 17] in DeepCAD format, or None if failed.
        """
        device = self.device

        # Bridge → z*
        z = self.bridge.encode_text([text], device) * self.corr_scale  # [1, 512]

        # n_ext prediction
        n_ext_raw = self.n_ext_head(z).round().clamp(1, 10).long().item()

        # AR generation with grammar
        cmds, args = self.decoder.generate(
            z,
            n_ext_targets=[n_ext_raw],
            greedy=False,
            temperature=0.8,
        )   # cmds [1,60], args [1,60,16]

        cmds = cmds[0].cpu().numpy()   # [60]
        args = args[0].cpu().numpy()   # [60, 16]

        # Trim at first EOS
        eos_positions = np.where(cmds == EOS)[0]
        end = int(eos_positions[0]) if len(eos_positions) > 0 else len(cmds)
        if end == 0:
            return None

        cmds = cmds[:end]
        args = args[:end]

        # Pack into [N, 17]
        vec = np.concatenate([cmds[:, None], args], axis=1).astype(np.int64)
        return vec


# Register in RUNNERS dict (inference/generate.py picks this up)
RUNNER = CADJEPAARv2Runner