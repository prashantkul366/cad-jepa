"""
Q-Former Bridge — Stage 2

BERT (frozen) → [B, T, 768]
→ linear(768→512) → [B, T, 512]
64 learned queries → 6 Q-Former blocks → LayerNorm → [B, 64, 512]

Output is trained (via VICReg) to live in JEPA latent space.
"""

import torch
import torch.nn as nn
from transformers import AutoModel

BERT_MODEL = 'bert-base-uncased'


class QFormerBlock(nn.Module):
    def __init__(self, d_model, n_heads, ff_dim):
        super().__init__()
        self.self_attn  = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ffn        = nn.Sequential(
            nn.Linear(d_model, ff_dim), nn.GELU(), nn.Linear(ff_dim, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, q, text, text_pad_mask=None):
        # self-attention among queries
        q = q + self.self_attn(
            self.norm1(q), self.norm1(q), self.norm1(q)
        )[0]
        # cross-attention: queries attend to all text tokens
        q = q + self.cross_attn(
            self.norm2(q), text, text,
            key_padding_mask=text_pad_mask
        )[0]
        q = q + self.ffn(self.norm3(q))
        return q


class QFormerBridge(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d_model   = cfg.d_model              # 512
        n_queries = cfg.n_qformer_queries    # 64
        n_blocks  = cfg.n_qformer_blocks     # 6
        n_heads   = cfg.n_heads              # 8
        ff_dim    = cfg.dim_feedforward      # 2048

        # frozen BERT
        self.bert = AutoModel.from_pretrained(BERT_MODEL)
        for p in self.bert.parameters():
            p.requires_grad_(False)

        # BERT dim → JEPA dim
        self.text_proj = nn.Linear(768, d_model)

        # learnable queries — initialized small
        self.queries = nn.Parameter(torch.randn(1, n_queries, d_model) * 0.02)

        self.blocks = nn.ModuleList([
            QFormerBlock(d_model, n_heads, ff_dim)
            for _ in range(n_blocks)
        ])

        self.norm = nn.LayerNorm(d_model)

    def forward(self, input_ids, attention_mask):
        """
        input_ids      : [B, T]
        attention_mask : [B, T]
        returns        : [B, 64, 512]
        """
        with torch.no_grad():
            bert_out = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).last_hidden_state                             # [B, T, 768]

        text    = self.text_proj(bert_out)                 # [B, T, 512]
        pad_mask = (attention_mask == 0)                   # [B, T] True=padding

        B = text.size(0)
        q = self.queries.expand(B, -1, -1).clone()        # [B, 64, 512]

        for block in self.blocks:
            q = block(q, text, text_pad_mask=pad_mask)

        return self.norm(q)                                # [B, 64, 512]

    @property
    def trainable_params(self):
        return [p for p in self.parameters() if p.requires_grad]