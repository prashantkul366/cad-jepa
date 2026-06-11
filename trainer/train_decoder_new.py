"""
CAD Decoder Trainer — Stage 3A

Trains the AR decoder on GT JEPA latents only.
No text involved. This is the ceiling experiment:
what is achievable when the decoder gets perfect geometric conditioning?

Scheduled sampling schedule:
  Epochs 1–20  : pure teacher forcing (ss_prob=0)
  Epochs 20–60 : ss_prob linearly increases 0 → 0.5
  Epochs 60+   : ss_prob held at 0.5
"""

import os, sys, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, '/content/cad-jepa')
from config.configJEPA        import ConfigJEPA
from model.cad_decoder        import CADDecoder, VALID_NEXT, cmd_valid_args, decoder_loss
from model.jepa_encoder       import CADJEPAEncoder
from dataset.cad_dataset      import CADDataset
from utils.schedulers         import WarmupCosineSchedule
from cadlib.macro             import (
    SOL_IDX, EOS_IDX, CMD_ARGS_MASK, N_ARGS
)
import numpy as np


# ── precomputed latent dataset ─────────────────────────────────────────────────
class JEPALatentDataset(Dataset):
    """
    Loads precomputed JEPA latents from disk.
    Used for Stage 3A — no text involved.
    """
    def __init__(self, phase, cfg):
        self.latent_dir = os.path.join(cfg.latent_dir, phase)
        self.cad_ds     = CADDataset(phase, cfg)

        # only keep samples where a precomputed latent exists
        self.indices = []
        for i, uid in enumerate(self.cad_ds.all_data):
            prefix, name = uid.split('/')
            lpath = os.path.join(self.latent_dir, prefix, f"{name}.pt")
            if os.path.exists(lpath):
                self.indices.append((i, lpath))

        print(f"[JEPALatentDataset] phase={phase} | "
              f"found={len(self.indices):,}/{len(self.cad_ds):,}")

    def __len__(self): return len(self.indices)

    def __getitem__(self, idx):
        cad_idx, lpath = self.indices[idx]
        item   = self.cad_ds[cad_idx]
        latent = torch.load(lpath, weights_only=True).float()  # [60, 512]
        item['latent'] = latent
        return item


# ── scheduled sampling helpers ─────────────────────────────────────────────────
def get_ss_prob(epoch, ss_start=20, ss_ramp=40, ss_max=0.5):
    if epoch < ss_start:
        return 0.0
    progress = min(1.0, (epoch - ss_start) / ss_ramp)
    return ss_max * progress


def scheduled_sampling_input(commands, args, model, memory, ss_prob, device):
    """
    Two-pass scheduled sampling.
    Pass 1: teacher forcing, get predictions (no grad)
    Pass 2: mixed input (GT + predictions) — this is the actual forward pass

    Returns mixed (commands, args) — decoder is called externally after this.
    """
    if ss_prob <= 0.0:
        return commands, args

    B, S = commands.shape
    cva  = cmd_valid_args(device)   # [6, 16]

    # Pass 1: get predictions (no gradient)
    with torch.no_grad():
        cl, al = model(commands, args, memory)
        pred_cmd  = cl.argmax(-1)                       # [B, S]
        pred_args = al.argmax(-1)                       # [B, S, 16] values 0-255

    # coin flips: for each position t (1..S-1), replace input[t] with pred[t-1]?
    coin = (torch.rand(B, S, device=device) < ss_prob)
    coin[:, 0] = False   # never replace SOL (position 0)

    # replace commands
    # at position t, we use pred_cmd[:, t-1] when coin[:, t] is True
    shifted_pred_cmd  = torch.cat([pred_cmd[:, :1], pred_cmd[:, :-1]], dim=1)   # [B, S]
    shifted_pred_args = torch.cat([pred_args[:, :1], pred_args[:, :-1]], dim=1) # [B, S, 16]

    new_cmd  = torch.where(coin, shifted_pred_cmd, commands)  # [B, S]

    # replace args — only valid positions for the new command
    valid_mask = cva[new_cmd]            # [B, S, 16] — True where arg is active
    pad_mask   = coin.unsqueeze(-1).expand_as(args) & ~valid_mask
    rep_mask   = coin.unsqueeze(-1).expand_as(args) &  valid_mask

    # shifted_pred_args are in [0,255]; convert to arg values for the decoder
    # (decoder embedding adds 1, so arg value -1 → index 0, value k → index k+1)
    # args stored as -1 to 255, we keep -1 for PAD
    new_args = args.clone()
    new_args[rep_mask] = shifted_pred_args[rep_mask]  # predicted values (0-255)
    new_args[pad_mask] = -1                           # invalid positions → PAD

    return new_cmd, new_args


# ── syntactic validity check ────────────────────────────────────────────────────
def syntactic_validity(cmd_seqs):
    """
    cmd_seqs: list of 1D tensors or [B, S] tensor
    Returns fraction of sequences that follow the grammar.
    """
    if isinstance(cmd_seqs, torch.Tensor):
        cmd_seqs = [cmd_seqs[b] for b in range(cmd_seqs.size(0))]

    valid = 0
    for seq in cmd_seqs:
        ok   = True
        prev = SOL_IDX
        for cmd in seq:
            cmd = int(cmd)
            if cmd == EOS_IDX:
                break
            if cmd not in VALID_NEXT.get(prev, set()):
                ok = False
                break
            prev = cmd
        valid += int(ok)
    return valid / len(cmd_seqs)


# ── main training loop ─────────────────────────────────────────────────────────
def train_decoder(cfg):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_ds  = JEPALatentDataset('train',      cfg)
    val_ds    = JEPALatentDataset('validation', cfg)
    train_ldr = DataLoader(train_ds, batch_size=cfg.decoder_batch_size,
                           shuffle=True,  num_workers=cfg.num_workers, pin_memory=True)
    val_ldr   = DataLoader(val_ds,   batch_size=cfg.decoder_batch_size,
                           shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    decoder = CADDecoder(cfg).to(device)
    print(f"Decoder params: {sum(p.numel() for p in decoder.parameters()):,}")

    optimizer = torch.optim.AdamW(
        decoder.parameters(),
        lr=cfg.decoder_lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.95)
    )
    total_steps  = cfg.decoder_epochs * len(train_ldr)
    warmup_steps = cfg.decoder_warmup_epochs * len(train_ldr)
    scheduler    = WarmupCosineSchedule(
        optimizer, warmup_steps=warmup_steps,
        start_lr=1e-6, ref_lr=cfg.decoder_lr,
        T_max=total_steps, final_lr=1e-6,
    )

    os.makedirs(cfg.decoder_ckpt_dir, exist_ok=True)
    best_val_loss = float('inf')
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(1, cfg.decoder_epochs + 1):
        ss_prob = get_ss_prob(epoch)

        # ── train ──
        decoder.train()
        tr_cmd, tr_args, n = 0.0, 0.0, 0

        for batch in train_ldr:
            cmd    = batch['command'].to(device)     # [B, 60]
            args   = batch['args'].to(device)        # [B, 60, 16]
            latent = batch['latent'].to(device)      # [B, 60, 512]

            # teacher forcing input: x[0..S-2] → predict x[1..S-1]
            cmd_in  = cmd[:, :-1]                    # [B, 59]
            args_in = args[:, :-1]                   # [B, 59, 16]
            cmd_tgt = cmd[:, 1:]                     # [B, 59]
            args_tgt= args[:, 1:]                    # [B, 59, 16]

            # scheduled sampling: mix GT and predicted inputs
            if ss_prob > 0.0:
                cmd_in, args_in = scheduled_sampling_input(
                    cmd_in, args_in, decoder, latent, ss_prob, device)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                cl, al = decoder(cmd_in, args_in, latent)
                cmd_loss, args_loss = decoder_loss(cl, al, cmd_tgt, args_tgt)
                loss = cmd_loss + args_loss

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(decoder.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            tr_cmd  += cmd_loss.item()
            tr_args += args_loss.item()
            n       += 1

        # ── val ──
        decoder.eval()
        vl_cmd, vl_args, vn = 0.0, 0.0, 0
        val_gen_cmds = []

        with torch.no_grad():
            for i, batch in enumerate(val_ldr):
                cmd    = batch['command'].to(device)
                args   = batch['args'].to(device)
                latent = batch['latent'].to(device)

                cl, al = decoder(cmd[:, :-1], args[:, :-1], latent)
                c, a   = decoder_loss(cl, al, cmd[:, 1:], args[:, 1:])
                vl_cmd += c.item(); vl_args += a.item(); vn += 1

                # generate a few samples for validity check (first 2 batches)
                if i < 2:
                    gen_cmd, _ = decoder.generate(latent, constrained=True)
                    val_gen_cmds.append(gen_cmd)

        tc = tr_cmd/n; ta = tr_args/n
        vc = vl_cmd/vn; va = vl_args/vn

        syn_val = 0.0
        if val_gen_cmds:
            all_gen = torch.cat(val_gen_cmds, dim=0)
            syn_val = syntactic_validity(all_gen)

        print(f"[3A] Epoch {epoch:4d}/{cfg.decoder_epochs} | "
              f"cmd={tc:.4f}/{vc:.4f} | args={ta:.4f}/{va:.4f} | "
              f"syn_val={syn_val:.3f} | ss={ss_prob:.2f}")

        val_total = vc + va
        if val_total < best_val_loss:
            best_val_loss = val_total
            torch.save({
                'epoch'  : epoch,
                'decoder': decoder.state_dict(),
                'val_loss': val_total,
                'syn_val': syn_val,
            }, os.path.join(cfg.decoder_ckpt_dir, 'best.pt'))
            print(f"  → saved best (val={val_total:.4f}, syn_val={syn_val:.3f})")

        if epoch % cfg.decoder_save_every == 0:
            torch.save({
                'epoch'  : epoch,
                'decoder': decoder.state_dict(),
            }, os.path.join(cfg.decoder_ckpt_dir, f'epoch_{epoch:04d}.pt'))


if __name__ == '__main__':
    cfg = ConfigJEPA()
    cfg.data_root            = '/content'
    cfg.latent_dir           = '/content/drive/MyDrive/cad-jepa-data/jepa_latents'
    cfg.decoder_ckpt_dir     = '/content/drive/MyDrive/cad-jepa-checkpoints/decoder'
    cfg.decoder_batch_size   = 128
    cfg.decoder_epochs       = 300
    cfg.decoder_warmup_epochs = 20
    cfg.decoder_lr           = 1e-4
    cfg.decoder_save_every   = 25
    cfg.n_decoder_layers     = 12
    train_decoder(cfg)