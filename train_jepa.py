# """
# CAD-JEPA  Stage 1 Pretraining Entry Point

# Usage:
#     python train_jepa.py
#     python train_jepa.py --epochs 300 --batch_size 256
#     python train_jepa.py --resume checkpoints/pretrain/epoch_0100.pt
# """

# import argparse
# import torch

# from config.configJEPA import ConfigJEPA
# from dataset.masks.multiblock_cad import CADMultiBlockMaskCollator
# from model.jepa_encoder import CADEncoder
# from model.predictor import CADPredictor
# from trainer.trainerJEPA import TrainerJEPA


# def get_args():
#     p = argparse.ArgumentParser()
#     p.add_argument("--data_root",  type=str, default=None)
#     p.add_argument("--ckpt_dir",   type=str, default=None)
#     p.add_argument("--epochs",     type=int, default=None)
#     p.add_argument("--batch_size", type=int, default=None)
#     p.add_argument("--resume",     type=str, default=None)
#     return p.parse_args()


# def main():
#     args = get_args()
#     cfg  = ConfigJEPA()
#     if args.data_root:  cfg.data_root  = args.data_root
#     if args.ckpt_dir:   cfg.ckpt_dir   = args.ckpt_dir
#     if args.epochs:     cfg.epochs     = args.epochs
#     if args.batch_size: cfg.batch_size = args.batch_size

#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print(f"Device: {device}")

#     # Data
#     collator = CADMultiBlockMaskCollator(mask_ratio=cfg.mask_ratio)
#     # TODO: wire up real DataLoader
#     # from dataset.cad_dataset import CADDataset
#     # from torch.utils.data import DataLoader
#     # ds = CADDataset(cfg.vec_root, cfg.train_split, phase="train")
#     # train_loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
#     #                           num_workers=cfg.num_workers, collate_fn=collator)
#     train_loader = None  # replace

#     # Models
#     enc  = CADEncoder(cfg)
#     pred = CADPredictor(cfg)
#     print(f"Encoder params : {sum(p.numel() for p in enc.parameters()):,}")
#     print(f"Predictor params: {sum(p.numel() for p in pred.parameters()):,}")

#     trainer     = TrainerJEPA(enc, pred, train_loader, cfg, device)
#     start_epoch = 1

#     if args.resume:
#         ckpt = torch.load(args.resume, map_location=device)
#         enc.load_state_dict(ckpt["encoder"])
#         pred.load_state_dict(ckpt["predictor"])
#         trainer.ema.load_state_dict(ckpt["ema"])
#         start_epoch = ckpt["epoch"] + 1
#         print(f"Resumed from epoch {ckpt['epoch']}")

#     for epoch in range(start_epoch, cfg.epochs + 1):
#         loss = trainer.train_epoch(epoch)
#         tau  = trainer._tau_schedule(epoch)
#         rank = trainer.monitor._last_rank_frac
#         print(f"Epoch {epoch:4d}/{cfg.epochs} | loss {loss:.5f} | tau {tau:.4f} | rank {rank:.2f}")
#         if epoch % cfg.save_every == 0:
#             trainer.save_checkpoint(epoch)


# if __name__ == "__main__":
#     main()


import sys, os, torch
sys.path.insert(0, '/content/cad-jepa')

from types import SimpleNamespace
from config.configJEPA import ConfigJEPA
from dataset.cad_dataset import CADDataset
from dataset.masks.multiblock_cad import CADMaskCollator
from model.jepa_encoder import CADJEPAEncoder
from model.predictor import CADPredictor
from trainer.trainerJEPA import TrainerJEPA
from torch.utils.data import DataLoader

cfg = ConfigJEPA()
cfg.data_root   = '/content'
cfg.ckpt_dir    = '/content/drive/MyDrive/cad-jepa-checkpoints/pretrain'
cfg.num_workers = 4
cfg.batch_size  = 256

dataset  = CADDataset('train', cfg)
collator = CADMaskCollator(cfg.mask_ratio)
loader   = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True,
                      num_workers=cfg.num_workers, collate_fn=collator,
                      pin_memory=True)

enc  = CADJEPAEncoder(cfg)
pred = CADPredictor(cfg)

print(f"Encoder params : {sum(p.numel() for p in enc.parameters()):,}")
print(f"Predictor params: {sum(p.numel() for p in pred.parameters()):,}")
print(f"Batches per epoch: {len(loader)}")

trainer = TrainerJEPA(enc, pred, loader, cfg, device='cuda')

for epoch in range(1, cfg.epochs + 1):
    loss = trainer.train_epoch(epoch)
    tau  = trainer._tau_schedule(epoch)
    rank = trainer.monitor._last_rank_frac
    print(f"Epoch {epoch:4d}/{cfg.epochs} | loss={loss:.5f} | tau={tau:.4f} | rank={rank:.2f}")
    if epoch % cfg.save_every == 0:
        trainer.save_checkpoint(epoch)