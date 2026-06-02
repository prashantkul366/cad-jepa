"""
CAD-JEPA Stage 1 Trainer

Per-batch loop:
  1. context_enc( visible_tokens )          -> h_ctx  [B, T_vis, 768]
  2. target_enc ( full_tokens ).detach()    -> h_tgt  [B, T_all, 768]  STOP GRAD
  3. predictor  ( h_ctx, mask_positions )   -> h_pred [B, n_mask, 768]
  4. loss  = smooth_l1( h_pred, h_tgt[masked] )
  5. loss += vicreg( h_ctx )   if rank collapses
  6. loss.backward()
  7. optimizer.step()
  8. ema.update( context_enc )     <- AFTER optimizer step
  9. ema.set_tau( schedule[epoch] )
"""

import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from trainer.base import BaseTrainer
from model.ema import EMATargetEncoder
from utils.collapse_monitor import CollapseMonitor
from config.configJEPA import ConfigJEPA


class TrainerJEPA(BaseTrainer):

    def __init__(
        self,
        context_encoder : nn.Module,
        predictor        : nn.Module,
        train_loader,
        cfg              : ConfigJEPA,
        device           : str = "cuda",
    ):
        super().__init__()
        self.cfg     = cfg
        self.device  = device
        self.enc     = context_encoder.to(device)
        self.pred    = predictor.to(device)
        self.ema     = EMATargetEncoder(context_encoder, tau=cfg.ema_tau_start)
        self.loader  = train_loader
        self.monitor = CollapseMonitor(cfg.encoder_dim, cfg.rank_threshold)

        # TODO: uncomment once encoder/predictor are implemented
        # self.optimizer = torch.optim.AdamW(
        #     list(self.enc.parameters()) + list(self.pred.parameters()),
        #     lr=cfg.lr, weight_decay=cfg.weight_decay)
        # from utils.schedulers import WarmupCosineSchedule
        # self.scheduler = WarmupCosineSchedule(
        #     self.optimizer, warmup_steps=cfg.lr_warmup_epochs, t_total=cfg.epochs)
        self.optimizer = None
        self.scheduler = None

    def train_step(
        self,
        tokens       : torch.Tensor,
        context_mask : torch.Tensor,
        target_mask  : torch.Tensor,
    ) -> torch.Tensor:

        tokens       = tokens.to(self.device)
        context_mask = context_mask.to(self.device)
        target_mask  = target_mask.to(self.device)

        # Step 2 — target encoder, full sequence, NO GRAD (critical)
        with torch.no_grad():
            h_tgt = self.ema(tokens).detach()   # [B, T, 768]  stop-gradient

        # Step 1 — context encoder, visible tokens only
        # TODO: extract visible tokens using context_mask, run self.enc
        # h_ctx = self.enc(visible_tokens, padding_mask=...)
        h_ctx = None  # replace with real implementation

        # Step 3 — predictor
        # TODO: get masked positions from target_mask, run self.pred
        # h_pred = self.pred(h_ctx, masked_positions)
        h_pred = None  # replace

        # Step 4 — L1 loss
        # TODO: gather h_tgt at masked positions, compute loss
        # h_tgt_masked = h_tgt[target_mask]
        # loss = F.smooth_l1_loss(h_pred.reshape(-1, 768), h_tgt_masked)
        loss = torch.tensor(0.0, requires_grad=True)  # replace

        # Step 5 — VICReg safety net
        if h_ctx is not None:
            self.monitor.effective_rank(h_ctx)
            if self.monitor.is_collapsing():
                loss = loss + self.monitor.regularization_loss(
                    h_ctx, self.cfg.vicreg_lambda_v, self.cfg.vicreg_lambda_c)

        return loss

    def train_epoch(self, epoch: int) -> float:
        self.enc.train()
        self.pred.train()
        self.ema.eval()
        self.ema.set_tau(self._tau_schedule(epoch))

        total = 0.0
        for tokens, ctx_mask, tgt_mask in self.loader:
            self.optimizer.zero_grad()
            loss = self.train_step(tokens, ctx_mask, tgt_mask)
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.enc.parameters()) + list(self.pred.parameters()),
                self.cfg.grad_clip,
            )
            self.optimizer.step()
            self.ema.update(self.enc)   # AFTER optimizer step
            total += loss.item()

        if self.scheduler:
            self.scheduler.step()
        return total / max(len(self.loader), 1)

    def _tau_schedule(self, epoch: int) -> float:
        if epoch >= self.cfg.ema_tau_warmup:
            return self.cfg.ema_tau
        p = epoch / self.cfg.ema_tau_warmup
        return self.cfg.ema_tau_start + p * (self.cfg.ema_tau - self.cfg.ema_tau_start)

    def save_checkpoint(self, epoch: int) -> None:
        os.makedirs(self.cfg.ckpt_dir, exist_ok=True)
        path = os.path.join(self.cfg.ckpt_dir, f"epoch_{epoch:04d}.pt")
        torch.save({
            "epoch"    : epoch,
            "encoder"  : self.enc.state_dict(),
            "predictor": self.pred.state_dict(),
            "ema"      : self.ema.state_dict(),
            "optimizer": self.optimizer.state_dict() if self.optimizer else None,
        }, path)
        print(f"  Saved: {path}")
