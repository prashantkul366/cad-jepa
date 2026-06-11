"""
Text-CAD paired dataset for Stage 2 bridge training.

Each sample:
    input_ids      : [T]       BERT token ids
    attention_mask : [T]       BERT attention mask
    latent         : [60, 512] precomputed JEPA encoder output
    uid            : str
    level          : str       annotation level used (beginner/intermediate/expert)
"""

import os, json, random, torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


BERT_MODEL = 'bert-base-uncased'


class TextCADDataset(Dataset):
    def __init__(self, phase, cfg):
        self.phase        = phase
        self.latent_dir   = cfg.latent_dir
        self.max_text_len = cfg.max_text_len
        self.levels       = ['beginner', 'intermediate', 'expert']

        with open(cfg.annot_path) as f:
            annot = json.load(f)

        # only keep UIDs where the precomputed latent exists
        phase_latent_dir = os.path.join(self.latent_dir, phase)
        self.samples = []
        for uid, texts in annot.items():
            prefix, name = uid.split('/')
            lpath = os.path.join(phase_latent_dir, prefix, f"{name}.pt")
            if os.path.exists(lpath):
                self.samples.append((uid, texts, lpath))

        self.tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
        print(f"[TextCADDataset] phase={phase} | samples={len(self.samples):,}")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        uid, texts, lpath = self.samples[idx]

        # all three annotation levels during training, expert at val/test
        level = random.choice(self.levels) if self.phase == 'train' else 'expert'
        text  = texts[level]

        tok = self.tokenizer(
            text,
            max_length    = self.max_text_len,
            padding       = 'max_length',
            truncation    = True,
            return_tensors= 'pt',
        )

        latent = torch.load(lpath, weights_only=True).float()  # [60, 512]

        return {
            'input_ids'     : tok['input_ids'].squeeze(0),       # [T]
            'attention_mask': tok['attention_mask'].squeeze(0),  # [T]
            'latent'        : latent,
            'uid'           : uid,
            'level'         : level,
        }


def get_text_cad_dataloader(phase, cfg, shuffle=None):
    is_shuffle = (phase == 'train') if shuffle is None else shuffle
    ds     = TextCADDataset(phase, cfg)
    loader = DataLoader(ds, batch_size=cfg.bridge_batch_size,
                        shuffle=is_shuffle, num_workers=cfg.num_workers,
                        pin_memory=True)
    print(f"[Dataloader] phase={phase} | batches={len(loader):,}")
    return loader