"""
trainer/train_decoder_ar_v2.py

Training script for CADDecoderARV2.
Run as a Colab cell or standalone script.

Training uses encoder z* from latent cache (NOT bridge z*).
Bridge z* is only used at inference.

Usage (Colab):
    %cd /content/cad-jepa
    exec(open('trainer/train_decoder_ar_v2.py').read())

Or standalone:
    python trainer/train_decoder_ar_v2.py
"""

import os, sys, json, time, random
import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

sys.path.insert(0, '/content/cad-jepa')
from model.decoder_ar_v2 import CADDecoderARV2, EOS, PAD_VAL, MAX_LEN, N_ARGS

# ══════════════════════════════════════════════════════════════════════════
# Config — edit these before running
# ══════════════════════════════════════════════════════════════════════════

CFG = dict(
    # Data
    data_root       = '/content',                  # contains cad_vec/ and train_val_test_split.json
    split_json      = '/content/train_val_test_split.json',
    cache_train     = '/content/drive/MyDrive/cad-jepa-data/latent_cache_train.npy',
    cache_val       = '/content/drive/MyDrive/cad-jepa-data/latent_cache_validation.npy',

    # Checkpoint
    ckpt_dir        = '/content/drive/MyDrive/cad-jepa-checkpoints/decoder_ar_v2',

    # Architecture
    latent_d        = 512,
    d_model         = 256,
    n_heads         = 8,
    n_layers        = 8,
    d_ff            = 1024,
    n_mem           = 8,
    d_arg_emb       = 32,
    dropout         = 0.1,

    # Training
    epochs          = 100,
    batch_size      = 128,
    lr              = 3e-4,
    lr_min          = 1e-5,
    warmup_epochs   = 5,
    grad_clip       = 1.0,
    kl_tolerance    = 3,
    kl_alpha        = 2.0,

    # Scheduled sampling (last 20% of training, ramps 0 → 0.25)
    # Disabled for now — enable by setting ss_start_epoch manually
    ss_start_epoch  = None,   # e.g. 80 to start at epoch 80
    ss_max_prob     = 0.25,

    # Misc
    num_workers     = 2,
    seed            = 42,
    save_every      = 10,       # save checkpoint every N epochs
    val_every       = 5,        # validate every N epochs
    device          = 'cuda',
)

# ══════════════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════════════

class DeepCADDataset(Dataset):
    """
    Loads DeepCAD h5 sequences and pairs them with precomputed JEPA latents.

    Returns: (z_star, cmds, args)
        z_star : [512]    float32 — from latent cache
        cmds   : [60]     int64   — command tokens
        args   : [60, 16] int64   — quantized args (-1 for PAD)
    """

    def __init__(self, uids, cad_root, latent_cache):
        self.uids    = uids
        self.root    = cad_root
        self.cache   = latent_cache   # dict {uid: np.float32[512]}

        # Filter to UIDs present in both h5 files and latent cache
        self.uids = [u for u in uids if u in self.cache and
                     os.path.exists(self._h5_path(u))]
        print(f"  Dataset: {len(self.uids)} UIDs")

    def _h5_path(self, uid):
        folder, name = uid.split('/')
        return os.path.join(self.root, 'cad_vec', folder, name + '.h5')

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, idx):
        uid    = self.uids[idx]
        z_star = torch.tensor(self.cache[uid], dtype=torch.float32)

        with h5py.File(self._h5_path(uid), 'r') as f:
            vec = f['vec'][:].astype(np.int64)   # [N, 17]

        N = vec.shape[0]
        # Pad or trim to MAX_LEN
        cmds = np.full(MAX_LEN, EOS, dtype=np.int64)
        args = np.full((MAX_LEN, N_ARGS), PAD_VAL, dtype=np.int64)

        n = min(N, MAX_LEN)
        cmds[:n]    = vec[:n, 0]
        args[:n, :] = vec[:n, 1:17]

        return (
            z_star,
            torch.tensor(cmds, dtype=torch.long),
            torch.tensor(args, dtype=torch.long),
        )


# ══════════════════════════════════════════════════════════════════════════
# Validation metric helpers
# ══════════════════════════════════════════════════════════════════════════

def compute_val_metrics(decoder, val_loader, device, n_batches=50):
    """
    Compute validation teacher-forcing accuracy on n_batches.
    Also checks AR structural validity on a small sample.
    """
    decoder.eval()
    cmd_accs, args_accs = [], []

    with torch.no_grad():
        for i, (z, cmds, args) in enumerate(val_loader):
            if i >= n_batches:
                break
            z, cmds, args = z.to(device), cmds.to(device), args.to(device)
            loss_d = decoder.compute_loss(z, cmds, args)

            # Teacher-forcing cmd accuracy
            bos_c  = torch.full((z.shape[0], 1), 6, dtype=torch.long, device=device)
            bos_a  = torch.full((z.shape[0], 1, N_ARGS), PAD_VAL, dtype=torch.long, device=device)
            in_c   = torch.cat([bos_c, cmds[:, :-1]], dim=1)
            in_a   = torch.cat([bos_a, args[:, :-1]], dim=1)
            cl, al = decoder.forward(z, in_c, in_a)

            pred_c = cl.argmax(-1)                    # [B, T]
            cmd_acc = (pred_c == cmds).float().mean().item()
            cmd_accs.append(cmd_acc)

            valid_args = (args != PAD_VAL)
            pred_a = al.argmax(-1) - 1                # shift back to -1..255
            if valid_args.sum() > 0:
                args_acc = ((pred_a == args) & valid_args).float().sum().item() / valid_args.float().sum().item()
                args_accs.append(args_acc)

    # AR structural validity on 30 samples
    struct_valid = 0
    n_try = 0
    with torch.no_grad():
        for z, cmds, args in val_loader:
            z = z[:30].to(device)
            gen_cmds, _ = decoder.generate(z, greedy=True)
            for b in range(z.shape[0]):
                c = gen_cmds[b].tolist()
                # Trim at first EOS
                if EOS in c:
                    c = c[:c.index(EOS)]
                # Valid: has SOL and EXT and no consecutive SOL
                has_sol = 4 in c
                has_ext = 5 in c
                no_consec_sol = not any(c[i] == 4 and c[i+1] == 4
                                       for i in range(len(c)-1))
                if has_sol and has_ext and no_consec_sol:
                    struct_valid += 1
                n_try += 1
            break   # one batch is enough

    return {
        'val_cmd_acc':    float(np.mean(cmd_accs)),
        'val_args_acc':   float(np.mean(args_accs)) if args_accs else 0.0,
        'struct_valid':   struct_valid / max(n_try, 1),
    }


# ══════════════════════════════════════════════════════════════════════════
# Training loop
# ══════════════════════════════════════════════════════════════════════════

def train():
    random.seed(CFG['seed'])
    torch.manual_seed(CFG['seed'])
    device = torch.device(CFG['device'])

    os.makedirs(CFG['ckpt_dir'], exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────
    print("Loading latent caches...")
    cache_tr = np.load(CFG['cache_train'], allow_pickle=True).item()
    cache_va = np.load(CFG['cache_val'],   allow_pickle=True).item()
    print(f"  Train cache: {len(cache_tr):,} UIDs")
    print(f"  Val cache:   {len(cache_va):,} UIDs")

    with open(CFG['split_json']) as f:
        split = json.load(f)
    # Note: key is 'validation' NOT 'val'
    train_uids = split['train']
    val_uids   = split['validation']

    print("Building datasets...")
    train_ds = DeepCADDataset(train_uids, CFG['data_root'], cache_tr)
    val_ds   = DeepCADDataset(val_uids,   CFG['data_root'], cache_va)

    train_loader = DataLoader(train_ds, batch_size=CFG['batch_size'],
                              shuffle=True,  num_workers=CFG['num_workers'],
                              pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=64,
                              shuffle=False, num_workers=CFG['num_workers'],
                              pin_memory=True)
    print(f"  Train: {len(train_ds):,}  Val: {len(val_ds):,}")

    # ── Model ──────────────────────────────────────────────────────────────
    decoder = CADDecoderARV2(
        latent_d  = CFG['latent_d'],
        d_model   = CFG['d_model'],
        n_heads   = CFG['n_heads'],
        n_layers  = CFG['n_layers'],
        d_ff      = CFG['d_ff'],
        n_mem     = CFG['n_mem'],
        d_arg_emb = CFG['d_arg_emb'],
        dropout   = CFG['dropout'],
    ).to(device)
    print(f"Decoder: {decoder.n_params:,} parameters")

    # ── Optimiser + scheduler ──────────────────────────────────────────────
    optimizer = AdamW(decoder.parameters(), lr=CFG['lr'], weight_decay=1e-2)

    steps_per_epoch = len(train_loader)
    total_steps     = CFG['epochs'] * steps_per_epoch
    warmup_steps    = CFG['warmup_epochs'] * steps_per_epoch

    warmup_sched = LinearLR(optimizer, start_factor=0.1,
                            end_factor=1.0, total_iters=warmup_steps)
    cosine_sched = CosineAnnealingLR(optimizer,
                                     T_max=max(total_steps - warmup_steps, 1),
                                     eta_min=CFG['lr_min'])
    scheduler = SequentialLR(optimizer, [warmup_sched, cosine_sched],
                              milestones=[warmup_steps])

    # ── Resume from checkpoint ─────────────────────────────────────────────
    start_epoch = 1
    best_val    = float('inf')
    ckpt_path   = os.path.join(CFG['ckpt_dir'], 'best.pt')

    latest = sorted([f for f in os.listdir(CFG['ckpt_dir'])
                     if f.startswith('epoch_')], reverse=True)
    if latest:
        ckpt = torch.load(os.path.join(CFG['ckpt_dir'], latest[0]),
                          map_location=device)
        decoder.load_state_dict(ckpt['decoder'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        best_val    = ckpt.get('best_val', float('inf'))
        print(f"Resumed from epoch {ckpt['epoch']} "
              f"(val_cmd_acc={ckpt.get('val_cmd_acc', '?'):.4f})")

    # ── Train ──────────────────────────────────────────────────────────────
    print(f"\nTraining for {CFG['epochs']} epochs "
          f"| batch={CFG['batch_size']} | lr={CFG['lr']}")

    for epoch in range(start_epoch, CFG['epochs'] + 1):
        decoder.train()
        t0 = time.time()
        total_loss = cmd_total = args_total = 0.0

        for batch_idx, (z, cmds, args) in enumerate(train_loader):
            z, cmds, args = z.to(device), cmds.to(device), args.to(device)

            loss_d = decoder.compute_loss(
                z, cmds, args,
                kl_tolerance=CFG['kl_tolerance'],
                kl_alpha=CFG['kl_alpha'],
            )

            optimizer.zero_grad()
            loss_d['loss'].backward()
            nn.utils.clip_grad_norm_(decoder.parameters(), CFG['grad_clip'])
            optimizer.step()
            scheduler.step()

            total_loss += loss_d['loss'].item()
            cmd_total  += loss_d['cmd_loss'].item()
            args_total += loss_d['args_loss'].item()

        n = len(train_loader)
        elapsed = time.time() - t0
        lr_now  = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:03d}/{CFG['epochs']} | "
              f"loss={total_loss/n:.4f} "
              f"cmd={cmd_total/n:.4f} args={args_total/n:.4f} | "
              f"lr={lr_now:.2e} | {elapsed:.0f}s")

        # ── Validation ──────────────────────────────────────────────────────
        if epoch % CFG['val_every'] == 0 or epoch == CFG['epochs']:
            metrics = compute_val_metrics(decoder, val_loader, device)
            print(f"  VAL: cmd_acc={metrics['val_cmd_acc']:.4f}  "
                  f"args_acc={metrics['val_args_acc']:.4f}  "
                  f"struct_valid={metrics['struct_valid']:.3f}")

            val_loss = -metrics['val_cmd_acc']   # use cmd_acc as proxy for best
            if val_loss < best_val:
                best_val = val_loss
                torch.save({
                    'epoch':        epoch,
                    'decoder':      decoder.state_dict(),
                    'optimizer':    optimizer.state_dict(),
                    'scheduler':    scheduler.state_dict(),
                    'val_cmd_acc':  metrics['val_cmd_acc'],
                    'val_args_acc': metrics['val_args_acc'],
                    'struct_valid': metrics['struct_valid'],
                    'best_val':     best_val,
                    'cfg':          CFG,
                }, ckpt_path)
                print(f"  ✓ Saved best.pt (cmd_acc={metrics['val_cmd_acc']:.4f})")
            decoder.train()

        # ── Epoch checkpoint ────────────────────────────────────────────────
        if epoch % CFG['save_every'] == 0:
            path = os.path.join(CFG['ckpt_dir'], f'epoch_{epoch:04d}.pt')
            torch.save({
                'epoch':     epoch,
                'decoder':   decoder.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_val':  best_val,
                'cfg':       CFG,
            }, path)
            print(f"  Saved {path}")

    print("\nTraining complete.")
    print(f"Best checkpoint: {ckpt_path}")


# ══════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    train()