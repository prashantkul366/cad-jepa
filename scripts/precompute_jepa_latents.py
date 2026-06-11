"""
Run once after Stage 1 completes.
Saves JEPA latents as individual .pt files mirroring the h5 directory structure.

Usage:
    python scripts/precompute_jepa_latents.py \
        --ckpt /content/drive/MyDrive/cad-jepa-checkpoints/pretrain/epoch_0300.pt \
        --annot /content/drive/MyDrive/cad-jepa-data/text2cad_annotations.json \
        --out_dir /content/drive/MyDrive/cad-jepa-data/jepa_latents \
        --data_root /content
"""

import sys, os, json, argparse, torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, '/content/cad-jepa')
from config.configJEPA   import ConfigJEPA
from model.jepa_encoder  import CADJEPAEncoder
from dataset.cad_dataset import CADDataset


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt',        required=True)
    p.add_argument('--annot',       required=True)
    p.add_argument('--out_dir',     required=True)
    p.add_argument('--data_root',   default='/content')
    p.add_argument('--batch_size',  type=int, default=256)
    p.add_argument('--num_workers', type=int, default=4)
    return p.parse_args()


class AnnotatedSubset(Dataset):
    """Filters a CADDataset to only UIDs that have Text2CAD annotations."""
    def __init__(self, base_dataset, annotated_uids):
        uid_to_idx = {uid: i for i, uid in enumerate(base_dataset.all_data)}
        self.base    = base_dataset
        self.indices = []
        self.uids    = []
        for uid in annotated_uids:
            if uid in uid_to_idx:
                self.indices.append(uid_to_idx[uid])
                self.uids.append(uid)
        print(f"  Annotated UIDs found in split: {len(self.uids):,}")

    def __len__(self): return len(self.indices)

    def __getitem__(self, i):
        item = self.base[self.indices[i]]
        item['uid'] = self.uids[i]
        return item


def main():
    args   = get_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with open(args.annot) as f:
        annot = json.load(f)
    annotated_uids = list(annot.keys())
    print(f"Total annotated UIDs: {len(annotated_uids):,}")

    cfg = ConfigJEPA()
    cfg.data_root = args.data_root
    enc = CADJEPAEncoder(cfg).to(device).eval()
    ckpt = torch.load(args.ckpt, map_location=device)
    enc.load_state_dict(ckpt['encoder'])
    print(f"Encoder loaded from {args.ckpt}")

    for phase in ['train', 'validation']:
        print(f"\nPhase: {phase}")
        cfg2 = ConfigJEPA(); cfg2.data_root = args.data_root
        base_ds = CADDataset(phase, cfg2)
        ds      = AnnotatedSubset(base_ds, annotated_uids)
        loader  = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

        phase_out = os.path.join(args.out_dir, phase)
        skipped   = 0

        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Encoding {phase}"):
                latents = enc(
                    batch['command'].to(device),
                    batch['args'].to(device)
                )                                          # [B, 60, 512]

                for i, uid in enumerate(batch['uid']):
                    prefix, name = uid.split('/')
                    save_dir  = os.path.join(phase_out, prefix)
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, f"{name}.pt")

                    if os.path.exists(save_path):
                        skipped += 1
                        continue

                    torch.save(latents[i].cpu().half(), save_path)

        print(f"  Done. Skipped {skipped} existing files.")

    print("\nPrecomputation complete.")


if __name__ == '__main__':
    main()