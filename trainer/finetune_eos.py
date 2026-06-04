"""
Fine-tune Stage 3 decoder from an existing checkpoint with EOS loss fix.
Run from /content/cad-jepa:
    python trainer/finetune_eos.py
"""
import sys, os
sys.path.insert(0, '/content/cad-jepa')
os.chdir('/content/cad-jepa')

import torch
from trainer.train_decoder import DecoderTrainer, ConfigDecoder

cfg = ConfigDecoder()
cfg.cache_train      = '/content/drive/MyDrive/cad-jepa-data/latent_cache_train.npy'
cfg.cache_val        = '/content/drive/MyDrive/cad-jepa-data/latent_cache_validation.npy'
cfg.data_root        = '/content'
cfg.ckpt_dir         = '/content/drive/MyDrive/cad-jepa-checkpoints/decoder_eos'

# Fine-tune config: lower LR, fewer epochs, save every epoch
cfg.lr               = 1e-5        # 10x lower than original
cfg.epochs           = 30
cfg.warmup_epochs    = 2
cfg.save_every       = 5
cfg.label_smoothing  = 0.05        # reduce smoothing — model is already calibrated

trainer = DecoderTrainer(cfg)

# Load existing weights but reset scheduler/optimizer for fine-tuning
ckpt = torch.load(
    '/content/drive/MyDrive/cad-jepa-checkpoints/decoder/best.pt',
    map_location='cuda'
)
trainer.model.load_state_dict(ckpt['model'])
print(f"Loaded decoder from epoch {ckpt['epoch']}  "
      f"(val_cmd_acc={ckpt.get('val_cmd_acc', '?'):.4f})")
print("Starting EOS fine-tuning...")

trainer.run()