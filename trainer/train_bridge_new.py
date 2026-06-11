"""
Stage 2: Q-Former Bridge Training

VICReg loss:
    L_align : smooth L1 — pooled Q-Former output vs pooled JEPA latents
    L_var   : variance  — per-dim std >= 1 across batch
    L_cov   : covariance — off-diagonal covariance → 0

Stop training when: val cos_sim > 0.82 AND val rank stable for 5 epochs.
"""

import os, sys, math, torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, '/content/cad-jepa')
from config.configJEPA           import ConfigJEPA
from model.qformer_bridge        import QFormerBridge
from dataset.text_cad_dataset    import get_text_cad_dataloader
from utils.schedulers            import WarmupCosineSchedule


# ── VICReg loss ────────────────────────────────────────────────────────────────
def vicreg_loss(C, z_star, lambda_a=25.0, lambda_v=25.0, lambda_c=1.0):
    """
    C      : [B, 64, 512]  Q-Former output
    z_star : [B, 60, 512]  GT JEPA latents (will be detached internally)
    """
    C_pool = C.mean(dim=1)              # [B, 512]
    Z_pool = z_star.mean(dim=1).detach()  # [B, 512]

    # alignment
    L_align = F.smooth_l1_loss(C_pool, Z_pool)

    # variance
    std_C  = C_pool.std(dim=0)
    L_var  = F.relu(1.0 - std_C).mean()

    # covariance
    B, D   = C_pool.shape
    C_norm = C_pool - C_pool.mean(dim=0)
    cov    = (C_norm.T @ C_norm) / (B - 1)
    off    = cov.pow(2)
    off.fill_diagonal_(0.0)
    L_cov  = off.sum() / D

    loss = lambda_a * L_align + lambda_v * L_var + lambda_c * L_cov

    with torch.no_grad():
        cos_sim = F.cosine_similarity(C_pool, Z_pool, dim=-1).mean().item()
        sv   = torch.linalg.svdvals(C_norm.float())
        p    = sv / (sv.sum() + 1e-8)
        rank = torch.exp(-(p * (p + 1e-8).log()).sum()).item()

    return loss, {
        'loss'   : loss.item(),
        'align'  : L_align.item(),
        'var'    : L_var.item(),
        'cov'    : L_cov.item(),
        'cos_sim': cos_sim,
        'rank'   : rank,
    }


# ── one epoch ──────────────────────────────────────────────────────────────────
def run_epoch(bridge, loader, optimizer, scheduler, cfg, device, train=True):
    bridge.train() if train else bridge.eval()
    totals = {'loss':0,'align':0,'var':0,'cov':0,'cos_sim':0,'rank':0}
    n = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attn_mask = batch['attention_mask'].to(device)
            latent    = batch['latent'].to(device)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                C = bridge(input_ids, attn_mask)
                loss, comp = vicreg_loss(
                    C.float(), latent.float(),
                    cfg.vicreg_lambda_a,
                    cfg.vicreg_lambda_v,
                    cfg.vicreg_lambda_c,
                )

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(bridge.trainable_params, cfg.grad_clip)
                optimizer.step()
                scheduler.step()

            for k in totals: totals[k] += comp[k]
            n += 1

    return {k: v/n for k, v in totals.items()}


# ── main ───────────────────────────────────────────────────────────────────────
def train_bridge(cfg):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_loader = get_text_cad_dataloader('train',      cfg)
    val_loader   = get_text_cad_dataloader('validation', cfg)

    bridge = QFormerBridge(cfg).to(device)
    print(f"Trainable params: {sum(p.numel() for p in bridge.trainable_params):,}")

    optimizer = torch.optim.AdamW(
        bridge.trainable_params,
        lr=cfg.bridge_lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.95)
    )
    total_steps  = cfg.bridge_epochs * len(train_loader)
    warmup_steps = cfg.bridge_warmup_epochs * len(train_loader)
    scheduler    = WarmupCosineSchedule(
        optimizer,
        warmup_steps=warmup_steps, start_lr=1e-6, ref_lr=cfg.bridge_lr,
        T_max=total_steps, final_lr=1e-6,
    )

    os.makedirs(cfg.bridge_ckpt_dir, exist_ok=True)
    best_cos_sim   = 0.0
    no_improve     = 0
    STOP_PATIENCE  = 10

    for epoch in range(1, cfg.bridge_epochs + 1):
        tr = run_epoch(bridge, train_loader, optimizer, scheduler, cfg, device, train=True)
        vl = run_epoch(bridge, val_loader,   optimizer, scheduler, cfg, device, train=False)

        print(
            f"Epoch {epoch:4d}/{cfg.bridge_epochs} | "
            f"loss={tr['loss']:.4f}/{vl['loss']:.4f} | "
            f"cos={tr['cos_sim']:.3f}/{vl['cos_sim']:.3f} | "
            f"rank={tr['rank']:.1f}/{vl['rank']:.1f} | "
            f"var={vl['var']:.3f} cov={vl['cov']:.4f}"
        )

        # checkpoint
        if vl['cos_sim'] > best_cos_sim:
            best_cos_sim = vl['cos_sim']
            no_improve   = 0
            torch.save({
                'epoch'   : epoch,
                'bridge'  : bridge.state_dict(),
                'cos_sim' : vl['cos_sim'],
                'rank'    : vl['rank'],
            }, os.path.join(cfg.bridge_ckpt_dir, 'best.pt'))
            print(f"  → saved best (cos_sim={best_cos_sim:.4f})")
        else:
            no_improve += 1

        if epoch % cfg.bridge_save_every == 0:
            torch.save({
                'epoch' : epoch,
                'bridge': bridge.state_dict(),
            }, os.path.join(cfg.bridge_ckpt_dir, f'epoch_{epoch:04d}.pt'))

        # stop when converged
        if vl['cos_sim'] > 0.82 and vl['rank'] > 20 and no_improve >= STOP_PATIENCE:
            print(f"Converged at epoch {epoch}. Best cos_sim={best_cos_sim:.4f}")
            break

    print(f"\nTraining complete. Best val cos_sim: {best_cos_sim:.4f}")


if __name__ == '__main__':
    cfg = ConfigJEPA()
    cfg.data_root           = '/content'
    cfg.latent_dir          = '/content/drive/MyDrive/cad-jepa-data/jepa_latents'
    cfg.annot_path          = '/content/drive/MyDrive/cad-jepa-data/text2cad_annotations.json'
    cfg.bridge_ckpt_dir     = '/content/drive/MyDrive/cad-jepa-checkpoints/bridge'
    cfg.bridge_batch_size   = 128
    cfg.bridge_epochs       = 150
    cfg.bridge_warmup_epochs = 10
    cfg.bridge_lr           = 1e-4
    cfg.bridge_save_every   = 10
    cfg.n_qformer_queries   = 64
    cfg.n_qformer_blocks    = 6
    cfg.max_text_len        = 128
    cfg.vicreg_lambda_a     = 25.0
    cfg.vicreg_lambda_v     = 25.0
    cfg.vicreg_lambda_c     = 1.0
    train_bridge(cfg)