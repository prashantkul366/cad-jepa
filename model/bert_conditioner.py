"""
model/bert_conditioner.py

Replaces TextToLatentBridge for V4 pipeline.
Returns both:
  z_bridge  [B, 512]     — CLS-based latent for n_ext_head + norm correction
  bert_hidden [B, 77, 768] — all token embeddings for decoder cross-attention
"""

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

BERT_DIM  = 768
LATENT_D  = 512
MAX_TOKENS = 77


class BERTConditioner(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert      = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        # Freeze layers 0-9, train 10-11 + pooler
        for p in self.bert.embeddings.parameters():
            p.requires_grad = False
        for i, layer in enumerate(self.bert.encoder.layer):
            for p in layer.parameters():
                p.requires_grad = (i >= 10)

        # CLS token → z_bridge [512] — for n_ext_head and norm correction
        self.z_proj = nn.Sequential(
            nn.Linear(BERT_DIM, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, LATENT_D)
        )

    def forward(self, texts):
        """
        texts: list of B strings

        Returns:
            z_bridge    [B, 512]      CLS-based latent
            bert_hidden [B, 77, 768]  all token embeddings
        """
        device = next(self.parameters()).device
        enc    = self.tokenizer(
            texts, padding='max_length', truncation=True,
            max_length=MAX_TOKENS, return_tensors='pt'
        ).to(device)

        out        = self.bert(**enc)
        hidden     = out.last_hidden_state   # [B, 77, 768]
        z_bridge   = self.z_proj(hidden[:, 0, :])   # CLS → [B, 512]

        return z_bridge, hidden

    def encode_text(self, texts, device):
        """Compatibility method — returns only z_bridge."""
        z, _ = self.forward(texts)
        return z