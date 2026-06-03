"""
CAD-JEPA Stage 1 Trainer

WarmupCosineSchedule: from utils/schedulers.py (I-JEPA) — used directly
BaseTrainer checkpoint logic: simplified from trainer/base.py
"""

import os
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

from model.ema import EMATargetEncoder
from utils.collapse_monitor import CollapseMonitor
from utils.schedulers import WarmupCosineSchedule


class TrainerJEPA:

    def __init__(self, encoder, predictor, train_loader, cfg, device='cuda'):
        self.cfg     = cfg
        self.device  = device
        self.enc     = encoder.to(device)
        self.pred    = predictor.to(device)
        self.ema     = EMATargetEncoder(encoder, tau=cfg.ema_tau_start)
        self.loader  = train_loader
        self.monitor = CollapseMonitor(cfg.d_model, cfg.rank_threshold)
        self.epoch   = 0

        self.optimizer = torch.optim.AdamW(
            list(self.enc.parameters()) + list(self.pred.parameters()),
            lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.95))

        # WarmupCosineSchedule from utils/schedulers.py (I-JEPA)
        total_steps   = cfg.epochs * 1000   # approx; updated per-epoch if needed
        warmup_steps  = cfg.lr_warmup_epochs * 1000
        self.scheduler = WarmupCosineSchedule(
            self.optimizer,
            warmup_steps=warmup_steps,
            start_lr=1e-6,
            ref_lr=cfg.lr,
            T_max=total_steps,
            final_lr=cfg.lr_min,
        )

    # ── training step ─────────────────────────────────────────────────────
    def train_step(self, commands, args, context_mask, target_mask):
        commands     = commands.to(self.device)
        args         = args.to(self.device)
        context_mask = context_mask.to(self.device)
        target_mask  = target_mask.to(self.device)
        B            = commands.size(0)

        # Step 1 — target encoder: full sequence, STOP GRADIENT
        with torch.no_grad():
            h_tgt = self.ema(commands, args).detach()   # [B, S, d]

        # Step 2 — context encoder: masked blocks hidden via key_padding_mask
        h_ctx = self.enc(commands, args, jepa_mask=context_mask)  # [B, S, d]

        # Step 3 — gather masked position indices per sample
        masked_idx_list = [
            target_mask[b].nonzero(as_tuple=False).squeeze(-1)
            for b in range(B)
        ]
        max_n = max(m.numel() for m in masked_idx_list)

        if max_n == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        # Pad to max_n (repeat last index for short samples)
        padded = torch.stack([
            torch.cat([m, m[-1:].expand(max_n - m.numel())])
            if m.numel() < max_n else m[:max_n]
            for m in masked_idx_list
        ])  # [B, max_n]

        # Step 4 — predictor
        h_pred = self.pred(h_ctx, padded)               # [B, max_n, d]

        # Step 5 — gather targets at masked positions
        d = h_tgt.size(-1)
        h_tgt_masked = h_tgt.gather(
            1, padded.unsqueeze(-1).expand(-1, -1, d))  # [B, max_n, d]

        # Step 6 — L1 loss (only on real masked positions, not padded)
        loss = torch.tensor(0.0, device=self.device)
        for b in range(B):
            n_real = masked_idx_list[b].numel()
            if n_real == 0:
                continue
            loss += F.smooth_l1_loss(
                h_pred[b, :n_real],
                h_tgt_masked[b, :n_real].detach())
        loss = loss / B

        # Step 7 — VICReg safety net
        # self.monitor.effective_rank(h_ctx)
        with torch.amp.autocast("cuda", enabled=False):
            self.monitor.effective_rank(h_ctx.float().detach())
            if self.monitor.is_collapsing():
                loss = loss + self.monitor.regularization_loss(
                    h_ctx.float(), self.cfg.vicreg_lambda_v, self.cfg.vicreg_lambda_c)

        return loss

    # ── epoch loop ─────────────────────────────────────────────────────────
    def train_epoch(self, epoch: int) -> float:
        self.enc.train()
        self.pred.train()
        self.ema.eval()
        self.ema.set_tau(self._tau_schedule(epoch))

        total, n_steps = 0.0, 0
        for commands, args, ctx_mask, tgt_mask in self.loader:
            self.optimizer.zero_grad()
            # loss = self.train_step(commands, args, ctx_mask, tgt_mask)
            # with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                loss = self.train_step(commands, args, ctx_mask, tgt_mask)
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.enc.parameters()) + list(self.pred.parameters()),
                self.cfg.grad_clip)
            self.optimizer.step()
            self.scheduler.step()
            self.ema.update(self.enc)        # AFTER optimizer step
            total += loss.item()
            n_steps += 1

        return total / max(n_steps, 1)

    def _tau_schedule(self, epoch):
        if epoch >= self.cfg.ema_tau_warmup:
            return self.cfg.ema_tau
        p = epoch / self.cfg.ema_tau_warmup
        # return self.cfg.ema_tau_start + p * (self.cfg.ema_tau - self.cfg.ema_tau_start)
        return self.cfg.ema_tau_start + (self.cfg.ema_tau - self.cfg.ema_tau_start) * (1 - math.cos(math.pi * p)) / 2


    def save_checkpoint(self, epoch):
        os.makedirs(self.cfg.ckpt_dir, exist_ok=True)
        path = os.path.join(self.cfg.ckpt_dir, f'epoch_{epoch:04d}.pt')
        torch.save({
            'epoch'    : epoch,
            'encoder'  : self.enc.state_dict(),
            'predictor': self.pred.state_dict(),
            'ema'      : self.ema.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, path)
        print(f'Saved: {path}')