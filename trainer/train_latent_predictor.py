"""
Latent Predictor Trainer

Stage 2B: L1 loss against GT JEPA latents
Stage 3B: End-to-end — Q-Former (frozen) → predictor (trainable)
          → frozen decoder → CAD generation loss
          Gradient flows back through decoder to predictor only.
"""

import os, sys, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, '/content/cad-jepa')
from config.configJEPA            import ConfigJEPA
from model.qformer_bridge         import QFormerBridge
from model.latent_predictor       import LatentPredictor
from model.cad_decoder            import CADDecoder
from dataset.text_cad_dataset     import TextCADDataset
from utils.schedulers             import WarmupCosineSchedule
from cadlib.macro                 import CMD_ARGS_MASK, EOS_IDX
import torch


# ── loss helpers ───────────────────────────────────────────────────────────────
def predictor_l1_loss(pred_latents, gt_latents):
    """
    pred_latents : [B, 60, 512]
    gt_latents   : [B, 60, 512]
    """
    return F.smooth_l1_loss(pred_latents, gt_latents.detach())


_CMD_VALID = torch.tensor(CMD_ARGS_MASK, dtype=torch.bool)   # [6, 16]


def decoder_loss(cmd_logits, args_logits, cmd_target, args_target):
    """
    cmd_logits   : [B, S-1, 6]
    args_logits  : [B, S-1, 16, 256]
    cmd_target   : [B, S-1]     GT commands at positions 1..S
    args_target  : [B, S-1, 16] GT args at positions 1..S
    """
    B, S, _ = cmd_logits.shape

    # command loss — all positions
    cmd_loss = F.cross_entropy(
        cmd_logits.reshape(-1, 6),
        cmd_target.reshape(-1).long(),
    )

    # args loss — only valid arg positions (GT value >= 0)
    af = args_logits.reshape(-1, 256)   # [B*S*16, 256]
    at = args_target.reshape(-1).long() # [B*S*16]
    valid = at >= 0
    if valid.any():
        args_loss = F.cross_entropy(af[valid], at[valid])
    else:
        args_loss = torch.tensor(0.0, device=cmd_logits.device)

    return cmd_loss, args_loss


# ── Stage 2B: L1 training ──────────────────────────────────────────────────────
def train_stage2b(cfg):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # data
    train_ds  = TextCADDataset('train',      cfg)
    val_ds    = TextCADDataset('validation', cfg)
    train_ldr = DataLoader(train_ds, batch_size=cfg.bridge_batch_size,
                           shuffle=True,  num_workers=cfg.num_workers, pin_memory=True)
    val_ldr   = DataLoader(val_ds,   batch_size=cfg.bridge_batch_size,
                           shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    # models
    bridge    = QFormerBridge(cfg).to(device)
    predictor = LatentPredictor(cfg).to(device)

    # load bridge from Stage 2
    b_ckpt = torch.load(cfg.bridge_ckpt_best, map_location=device)
    bridge.load_state_dict(b_ckpt['bridge'])
    for p in bridge.parameters():
        p.requires_grad_(False)
    bridge.eval()
    print(f"Bridge loaded (frozen) — Stage 2 cos_sim={b_ckpt['cos_sim']:.4f}")

    trainable = list(predictor.parameters())
    print(f"Predictor params: {sum(p.numel() for p in trainable):,}")

    optimizer = torch.optim.AdamW(
        trainable, lr=cfg.predictor_lr,
        weight_decay=cfg.weight_decay, betas=(0.9, 0.95)
    )
    total_steps  = cfg.predictor_epochs * len(train_ldr)
    warmup_steps = cfg.predictor_warmup_epochs * len(train_ldr)
    scheduler    = WarmupCosineSchedule(
        optimizer, warmup_steps=warmup_steps,
        start_lr=1e-6, ref_lr=cfg.predictor_lr,
        T_max=total_steps, final_lr=1e-6,
    )

    os.makedirs(cfg.predictor_ckpt_dir, exist_ok=True)
    best_val_loss = float('inf')

    for epoch in range(1, cfg.predictor_epochs + 1):
        # ── train ──
        predictor.train()
        tr_loss, n = 0.0, 0
        for batch in train_ldr:
            ids  = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            lat  = batch['latent'].to(device)       # [B, 60, 512] GT

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                with torch.no_grad():
                    qf_out = bridge(ids, mask)       # [B, 64, 512]
                pred = predictor(qf_out)             # [B, 60, 512]
                loss = predictor_l1_loss(pred, lat)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, cfg.grad_clip)
            optimizer.step()
            scheduler.step()
            tr_loss += loss.item(); n += 1

        # ── val ──
        predictor.eval()
        vl_loss, vn = 0.0, 0
        cos_sims = []
        with torch.no_grad():
            for batch in val_ldr:
                ids  = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                lat  = batch['latent'].to(device)
                qf   = bridge(ids, mask)
                pred = predictor(qf)
                vl_loss += predictor_l1_loss(pred, lat).item(); vn += 1
                cos_sims.append(
                    F.cosine_similarity(
                        pred.mean(1), lat.mean(1), dim=-1
                    ).mean().item()
                )

        tl = tr_loss/n; vl = vl_loss/vn
        cs = sum(cos_sims)/len(cos_sims)
        print(f"[2B] Epoch {epoch:4d}/{cfg.predictor_epochs} | "
              f"L1={tl:.4f}/{vl:.4f} | cos={cs:.4f}")

        if vl < best_val_loss:
            best_val_loss = vl
            torch.save({
                'epoch'    : epoch,
                'predictor': predictor.state_dict(),
                'val_l1'   : vl,
                'cos_sim'  : cs,
            }, os.path.join(cfg.predictor_ckpt_dir, 'best.pt'))
            print(f"  → saved best (L1={vl:.4f})")

        if epoch % cfg.predictor_save_every == 0:
            torch.save({'epoch': epoch,
                        'predictor': predictor.state_dict()},
                       os.path.join(cfg.predictor_ckpt_dir, f'epoch_{epoch:04d}.pt'))


# ── Stage 3B: end-to-end finetune ─────────────────────────────────────────────
def train_stage3b(cfg):
    """
    Freeze: bridge (Q-Former) + decoder
    Trainable: latent predictor only
    Loss: teacher-forced CAD generation loss flows back to predictor
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_ds  = TextCADDataset('train',      cfg)
    val_ds    = TextCADDataset('validation', cfg)
    train_ldr = DataLoader(train_ds, batch_size=cfg.s3b_batch_size,
                           shuffle=True,  num_workers=cfg.num_workers, pin_memory=True)
    val_ldr   = DataLoader(val_ds,   batch_size=cfg.s3b_batch_size,
                           shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    # load all three models
    bridge    = QFormerBridge(cfg).to(device)
    predictor = LatentPredictor(cfg).to(device)
    decoder   = CADDecoder(cfg).to(device)

    b_ckpt = torch.load(cfg.bridge_ckpt_best,    map_location=device)
    p_ckpt = torch.load(cfg.predictor_ckpt_best, map_location=device)
    d_ckpt = torch.load(cfg.decoder_ckpt_best,   map_location=device)

    bridge.load_state_dict(b_ckpt['bridge'])
    predictor.load_state_dict(p_ckpt['predictor'])
    decoder.load_state_dict(d_ckpt['decoder'])

    # freeze bridge and decoder
    for p in bridge.parameters():   p.requires_grad_(False)
    for p in decoder.parameters():  p.requires_grad_(False)
    bridge.eval(); decoder.eval()

    # predictor is the only trainable component
    trainable = list(predictor.parameters())
    print(f"Stage 3B trainable (predictor only): "
          f"{sum(p.numel() for p in trainable):,}")

    optimizer = torch.optim.AdamW(
        trainable, lr=cfg.s3b_lr,
        weight_decay=cfg.weight_decay, betas=(0.9, 0.95)
    )
    total_steps  = cfg.s3b_epochs * len(train_ldr)
    warmup_steps = max(1, cfg.s3b_warmup_epochs * len(train_ldr))
    scheduler    = WarmupCosineSchedule(
        optimizer, warmup_steps=warmup_steps,
        start_lr=1e-7, ref_lr=cfg.s3b_lr,
        T_max=total_steps, final_lr=1e-7,
    )

    os.makedirs(cfg.s3b_ckpt_dir, exist_ok=True)
    best_val_loss = float('inf')

    for epoch in range(1, cfg.s3b_epochs + 1):
        # ── train ──
        predictor.train()
        tr_cmd, tr_args, n = 0.0, 0.0, 0

        for batch in train_ldr:
            ids   = batch['input_ids'].to(device)
            mask  = batch['attention_mask'].to(device)
            cmd   = batch['command'].to(device)   # [B, 60]
            args  = batch['args'].to(device)      # [B, 60, 16]

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                with torch.no_grad():
                    qf = bridge(ids, mask)           # frozen
                pred_lat = predictor(qf)             # [B, 60, 512] — has grad

                # teacher forcing: input = x[0..S-2], target = x[1..S-1]
                cl, al = decoder(cmd[:, :-1], args[:, :-1], pred_lat)
                cmd_loss, args_loss = decoder_loss(
                    cl, al, cmd[:, 1:], args[:, 1:])
                loss = cmd_loss + args_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, cfg.grad_clip)
            optimizer.step()
            scheduler.step()
            tr_cmd  += cmd_loss.item()
            tr_args += args_loss.item()
            n += 1

        # ── val ──
        predictor.eval()
        vl_cmd, vl_args, vn = 0.0, 0.0, 0
        with torch.no_grad():
            for batch in val_ldr:
                ids  = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                cmd  = batch['command'].to(device)
                args = batch['args'].to(device)
                qf   = bridge(ids, mask)
                pl   = predictor(qf)
                cl, al = decoder(cmd[:, :-1], args[:, :-1], pl)
                c, a   = decoder_loss(cl, al, cmd[:, 1:], args[:, 1:])
                vl_cmd += c.item(); vl_args += a.item(); vn += 1

        tc = tr_cmd/n; ta = tr_args/n
        vc = vl_cmd/vn; va = vl_args/vn
        print(f"[3B] Epoch {epoch:4d}/{cfg.s3b_epochs} | "
              f"cmd={tc:.4f}/{vc:.4f} | args={ta:.4f}/{va:.4f}")

        val_total = vc + va
        if val_total < best_val_loss:
            best_val_loss = val_total
            torch.save({
                'epoch'    : epoch,
                'predictor': predictor.state_dict(),
                'val_loss' : val_total,
            }, os.path.join(cfg.s3b_ckpt_dir, 'best.pt'))
            print(f"  → saved best (val={val_total:.4f})")


if __name__ == '__main__':
    cfg = ConfigJEPA()
    cfg.data_root             = '/content'
    cfg.latent_dir            = '/content/drive/MyDrive/cad-jepa-data/jepa_latents'
    cfg.annot_path            = '/content/drive/MyDrive/cad-jepa-data/text2cad_annotations.json'
    cfg.bridge_ckpt_best      = '/content/drive/MyDrive/cad-jepa-checkpoints/bridge/best.pt'
    cfg.predictor_ckpt_dir    = '/content/drive/MyDrive/cad-jepa-checkpoints/predictor'
    cfg.predictor_ckpt_best   = '/content/drive/MyDrive/cad-jepa-checkpoints/predictor/best.pt'
    cfg.decoder_ckpt_best     = '/content/drive/MyDrive/cad-jepa-checkpoints/decoder/best.pt'
    cfg.s3b_ckpt_dir          = '/content/drive/MyDrive/cad-jepa-checkpoints/stage3b'
    cfg.predictor_epochs      = 100
    cfg.predictor_warmup_epochs = 5
    cfg.predictor_lr          = 5e-5
    cfg.predictor_save_every  = 10
    cfg.n_predictor_blocks    = 6
    cfg.s3b_epochs            = 50
    cfg.s3b_warmup_epochs     = 3
    cfg.s3b_lr                = 1e-5
    cfg.s3b_batch_size        = 64

    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else '2b'
    if mode == '2b':
        train_stage2b(cfg)
    elif mode == '3b':
        train_stage3b(cfg)