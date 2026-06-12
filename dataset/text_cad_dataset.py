# """
# Text-CAD paired dataset for Stage 2 bridge training.

# Each sample:
#     input_ids      : [T]       BERT token ids
#     attention_mask : [T]       BERT attention mask
#     latent         : [60, 512] precomputed JEPA encoder output
#     uid            : str
#     level          : str       annotation level used (beginner/intermediate/expert)
# """

# import os, json, random, torch
# from torch.utils.data import Dataset, DataLoader
# from transformers import AutoTokenizer


# BERT_MODEL = 'bert-base-uncased'


# class TextCADDataset(Dataset):
#     def __init__(self, phase, cfg):
#         self.phase        = phase
#         self.latent_dir   = cfg.latent_dir
#         self.max_text_len = cfg.max_text_len
#         self.levels       = ['beginner', 'intermediate', 'expert']

#         with open(cfg.annot_path) as f:
#             annot = json.load(f)

#         # only keep UIDs where the precomputed latent exists
#         phase_latent_dir = os.path.join(self.latent_dir, phase)
#         self.samples = []
#         for uid, texts in annot.items():
#             prefix, name = uid.split('/')
#             lpath = os.path.join(phase_latent_dir, prefix, f"{name}.pt")
#             if os.path.exists(lpath):
#                 self.samples.append((uid, texts, lpath))

#         self.tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
#         print(f"[TextCADDataset] phase={phase} | samples={len(self.samples):,}")

#     def __len__(self): return len(self.samples)

#     def __getitem__(self, idx):
#         uid, texts, lpath = self.samples[idx]

#         # all three annotation levels during training, expert at val/test
#         level = random.choice(self.levels) if self.phase == 'train' else 'expert'
#         text  = texts[level]

#         tok = self.tokenizer(
#             text,
#             max_length    = self.max_text_len,
#             padding       = 'max_length',
#             truncation    = True,
#             return_tensors= 'pt',
#         )

#         latent = torch.load(lpath, weights_only=True).float()  # [60, 512]

#         return {
#             'input_ids'     : tok['input_ids'].squeeze(0),       # [T]
#             'attention_mask': tok['attention_mask'].squeeze(0),  # [T]
#             'latent'        : latent,
#             'uid'           : uid,
#             'level'         : level,
#         }


# def get_text_cad_dataloader(phase, cfg, shuffle=None):
#     is_shuffle = (phase == 'train') if shuffle is None else shuffle
#     ds     = TextCADDataset(phase, cfg)
#     loader = DataLoader(ds, batch_size=cfg.bridge_batch_size,
#                         shuffle=is_shuffle, num_workers=cfg.num_workers,
#                         pin_memory=True)
#     print(f"[Dataloader] phase={phase} | batches={len(loader):,}")
#     return loader


"""
Text-CAD paired dataset for Stage 2 bridge training.
Uses precomputed BERT tokens and local JEPA latents for fast loading.
"""

import os, json, random, torch, numpy as np
from torch.utils.data import Dataset, DataLoader


class TextCADDataset(Dataset):
    def __init__(self, phase, cfg):
        self.phase      = phase
        self.latent_dir = os.path.join(
            getattr(cfg, 'latent_dir_local', cfg.latent_dir), phase)
        self.levels     = ['beginner', 'intermediate', 'expert']

        # load annotations
        with open(cfg.annot_path) as f:
            annot = json.load(f)

        # load precomputed BERT tokens if available
        bert_path = getattr(cfg, 'bert_tokens_path', None)
        if bert_path and os.path.exists(bert_path):
            print(f"Loading precomputed BERT tokens from {bert_path}...")
            tok_data         = torch.load(bert_path, weights_only=True)
            self.input_ids   = tok_data['input_ids']
            self.attn_masks  = tok_data['attention_mask']
            self.use_precomp = True
            print("  loaded.")
        else:
            from transformers import AutoTokenizer
            self.tokenizer   = AutoTokenizer.from_pretrained('bert-base-uncased')
            self.use_precomp = False
            self.max_text_len = cfg.max_text_len

        # build sample list — only UIDs with latent on disk
        self.samples = []
        for uid, texts in annot.items():
            prefix, name = uid.split('/')
            lpath = os.path.join(self.latent_dir, prefix, f"{name}.pt")
            if os.path.exists(lpath):
                self.samples.append((uid, texts, lpath))

        print(f"[TextCADDataset] phase={phase} | samples={len(self.samples):,}")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        uid, texts, lpath = self.samples[idx]

        level = random.choice(self.levels) if self.phase == 'train' else 'expert'

        # BERT tokens
        if self.use_precomp:
            key         = f"{uid}_{level}"
            input_ids   = torch.from_numpy(
                self.input_ids[key].astype(np.int64))
            attn_mask   = torch.from_numpy(
                self.attn_masks[key].astype(np.int64))
        else:
            text = texts[level]
            tok  = self.tokenizer(
                text, max_length=self.max_text_len,
                padding='max_length', truncation=True,
                return_tensors='pt',
            )
            input_ids = tok['input_ids'].squeeze(0)
            attn_mask = tok['attention_mask'].squeeze(0)

        # latent
        latent = torch.load(lpath, weights_only=True).float()   # [60, 512]

        return {
            'input_ids'     : input_ids,
            'attention_mask': attn_mask,
            'latent'        : latent,
            'uid'           : uid,
            'level'         : level,
        }


def get_text_cad_dataloader(phase, cfg, shuffle=None):
    is_shuffle = (phase == 'train') if shuffle is None else shuffle
    ds     = TextCADDataset(phase, cfg)
    loader = DataLoader(
        ds, batch_size=cfg.bridge_batch_size,
        shuffle=is_shuffle, num_workers=cfg.num_workers,
        pin_memory=True, persistent_workers=True,
        prefetch_factor=4,
    )
    print(f"[Dataloader] phase={phase} | batches={len(loader):,}")
    return loader