"""
model/text_bridge.py
Stage 2 — Text-to-Latent Bridge

Maps raw text descriptions → CAD-JEPA latent space (512-dim).

Architecture:
    CLIP text encoder (openai/clip-vit-base-patch32)
        └── 12 transformer layers total
            ├── layers 0–9  : FROZEN
            └── layers 10–11: trainable  +  final_layer_norm: trainable
    MLP projector: 512 → 1024 → 512 (fully trainable)

Freeze rationale:
    Early CLIP layers encode general language syntax — no benefit in fine-tuning.
    Last 2 layers learn task-specific pooling toward CAD semantics.
    Projector maps CLIP's language manifold onto the JEPA CAD latent manifold.

Training vs inference:
    Training  → tokenize() + forward()            (gradients flow)
    Inference → encode_text()                     (@no_grad convenience)

Expected param counts (clip-vit-base-patch32):
    Total CLIP text model : ~63M
    Frozen               : ~58M  (embeddings + layers 0-9)
    Trainable CLIP       :  ~5M  (layers 10-11 + final_layer_norm)
    Projector            :  ~1M  (512→1024→512)
    Total trainable      :  ~6M
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import CLIPTextModel, CLIPTokenizer


# ──────────────────────────────────────────────────────────────────────────────
# TextToLatentBridge
# ──────────────────────────────────────────────────────────────────────────────

class TextToLatentBridge(nn.Module):
    """
    CLIP text encoder + MLP projector bridging text → CAD-JEPA latent.

    Args:
        clip_model  : HuggingFace model ID (default: openai/clip-vit-base-patch32)
        jepa_d      : Target latent dimension — must match CADJEPAEncoder d_model (512)
        proj_hidden : Hidden width of MLP projector (default: 1024)
        dropout     : Dropout probability in projector (default: 0.1)
        n_freeze    : Number of CLIP transformer layers to freeze from the bottom
                      (default: 10, keeps last 2 of 12 trainable)

    Example:
        bridge = TextToLatentBridge().cuda()

        # Training
        tokens = bridge.tokenize(text_list, device)
        z_pred = bridge(tokens)                          # [B, 512], grad enabled
        loss   = bridge_loss(z_pred, z_target)

        # Inference
        z_pred = bridge.encode_text(["a bracket with a slot"], device)  # [B, 512]
    """

    CLIP_DIM     = 512   # pooler_output dim for clip-vit-base-patch32
    MAX_TEXT_LEN = 77    # CLIP's fixed context window

    def __init__(
        self,
        clip_model  : str   = "openai/clip-vit-base-patch32",
        jepa_d      : int   = 512,
        proj_hidden : int   = 1024,
        dropout     : float = 0.1,
        n_freeze    : int   = 10,
    ):
        super().__init__()

        self.clip_model_id = clip_model
        self.n_freeze      = n_freeze

        # ── CLIP text encoder ─────────────────────────────────────────────────
        self.tokenizer = CLIPTokenizer.from_pretrained(clip_model)
        self.clip_text = CLIPTextModel.from_pretrained(clip_model)
        self._apply_clip_freeze()

        # ── MLP projector: CLIP_DIM → proj_hidden → jepa_d ───────────────────
        self.projector = nn.Sequential(
            nn.Linear(self.CLIP_DIM, proj_hidden),
            nn.LayerNorm(proj_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_hidden, jepa_d),
        )

    # ── Freeze / unfreeze ─────────────────────────────────────────────────────

    def _apply_clip_freeze(self) -> None:
        """
        Freeze CLIP embeddings and first n_freeze transformer layers.
        Keep last (12 - n_freeze) layers + final_layer_norm trainable.
        """
        # Freeze everything first
        for p in self.clip_text.parameters():
            p.requires_grad = False

        # text_model = self.clip_text.text_model
        text_model = self.clip_text

        # Unfreeze last (total - n_freeze) transformer layers
        for i, layer in enumerate(text_model.encoder.layers):
            if i >= self.n_freeze:
                for p in layer.parameters():
                    p.requires_grad = True

        # Unfreeze final layer norm (semantically belongs with the last layers)
        for p in text_model.final_layer_norm.parameters():
            p.requires_grad = True

    # ── Parameter groups ─────────────────────────────────────────────────────

    def param_groups(
        self,
        lr_proj     : float = 1e-4,
        lr_clip     : float = 1e-5,
        weight_decay: float = 0.01,
    ) -> list[dict]:
        """
        Return two optimizer param groups with separate learning rates.

        Usage:
            optimizer = torch.optim.AdamW(bridge.param_groups(), ...)
        """
        clip_trainable = [p for p in self.clip_text.parameters() if p.requires_grad]
        proj_params    = list(self.projector.parameters())

        assert len(clip_trainable) > 0, "No trainable CLIP params — check n_freeze"
        assert len(proj_params)    > 0, "No projector params"

        return [
            {
                'params'      : proj_params,
                'lr'          : lr_proj,
                'weight_decay': weight_decay,
                'name'        : 'projector',
            },
            {
                'params'      : clip_trainable,
                'lr'          : lr_clip,
                'weight_decay': weight_decay,
                'name'        : 'clip_finetune',
            },
        ]

    # ── Tokenization (separated from forward for training loops) ─────────────

    def tokenize(self, text_strings: list[str], device: torch.device) -> dict:
        """
        Tokenize a list of text strings.
        Returns dict ready to pass into forward().

        Used in training:
            tokens = bridge.tokenize(texts, device)
            z_pred = bridge(tokens)   # gradients flow
        """
        return self.tokenizer(
            text_strings,
            return_tensors = 'pt',
            padding        = 'max_length',
            truncation     = True,
            max_length     = self.MAX_TEXT_LEN,
        ).to(device)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, text_inputs: dict) -> torch.Tensor:
        """
        Forward pass — gradients flow through trainable CLIP layers + projector.

        Args:
            text_inputs: tokenizer output dict
                         {'input_ids': [B, 77], 'attention_mask': [B, 77]}
        Returns:
            z_pred: [B, jepa_d]  — predicted CAD-JEPA latent
        """
        outputs = self.clip_text(**text_inputs)
        pooled  = outputs.pooler_output   # [B, 512] — EOS token representation
        return self.projector(pooled)     # [B, jepa_d]

    # ── Inference convenience ─────────────────────────────────────────────────

    @torch.no_grad()
    def encode_text(self, text_strings: list[str], device: torch.device) -> torch.Tensor:
        """
        Inference convenience: tokenise + forward, no gradient tracking.
        Used in Stage 3 inference pipeline.

        Args:
            text_strings: list of B raw text prompts
            device:       target device
        Returns:
            z_pred: [B, jepa_d]
        """
        tokens = self.tokenize(text_strings, device)
        return self.forward(tokens)

    # ── Checkpoint helpers ────────────────────────────────────────────────────

    def save(self, path: str, epoch: int, optimizer=None, val_cos_sim: float = 0.0) -> None:
        """Save bridge checkpoint (projector + trainable CLIP layers)."""
        payload = {
            'epoch'        : epoch,
            'val_cos_sim'  : val_cos_sim,
            'bridge'       : self.state_dict(),
            'clip_model_id': self.clip_model_id,
            'n_freeze'     : self.n_freeze,
        }
        if optimizer is not None:
            payload['optimizer'] = optimizer.state_dict()
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str, device: torch.device, **init_kwargs) -> 'TextToLatentBridge':
        """
        Load bridge from checkpoint.

        Usage:
            bridge = TextToLatentBridge.load('epoch_050.pt', device)
        """
        ckpt   = torch.load(path, map_location=device)
        bridge = cls(
            clip_model = ckpt.get('clip_model_id', 'openai/clip-vit-base-patch32'),
            n_freeze   = ckpt.get('n_freeze', 10),
            **init_kwargs,
        )
        bridge.load_state_dict(ckpt['bridge'])
        bridge.to(device)
        print(f"[TextBridge] Loaded epoch {ckpt['epoch']} | val_cos_sim={ckpt['val_cos_sim']:.4f}")
        return bridge

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def param_summary(self) -> None:
        """Print parameter count split by component and trainability."""
        def count(params):
            return sum(p.numel() for p in params)

        clip_frozen    = [p for p in self.clip_text.parameters() if not p.requires_grad]
        clip_trainable = [p for p in self.clip_text.parameters() if     p.requires_grad]
        proj_params    = list(self.projector.parameters())

        print("=" * 50)
        print("TextToLatentBridge — parameter summary")
        print("=" * 50)
        print(f"  CLIP frozen    : {count(clip_frozen):>12,}")
        print(f"  CLIP trainable : {count(clip_trainable):>12,}")
        print(f"  Projector      : {count(proj_params):>12,}")
        print("-" * 50)
        total_train = count(clip_trainable) + count(proj_params)
        total_all   = count(list(self.parameters()))
        print(f"  Total trainable: {total_train:>12,}")
        print(f"  Total params   : {total_all:>12,}")
        print("=" * 50)

    def verify_freeze(self) -> None:
        """Assert freeze pattern is correct — call after init."""
        # text_model = self.clip_text.text_model
        text_model = self.clip_text
        layers     = text_model.encoder.layers

        for i, layer in enumerate(layers):
            for p in layer.parameters():
                expected = (i >= self.n_freeze)
                assert p.requires_grad == expected, (
                    f"Layer {i}: expected requires_grad={expected}, got {p.requires_grad}"
                )

        for p in text_model.final_layer_norm.parameters():
            assert p.requires_grad, "final_layer_norm should be trainable"

        for p in self.projector.parameters():
            assert p.requires_grad, "projector should be trainable"

        print(f"[TextBridge] Freeze verified: layers 0–{self.n_freeze - 1} frozen, "
              f"layers {self.n_freeze}–{len(layers) - 1} + final_layer_norm trainable.")


# ──────────────────────────────────────────────────────────────────────────────
# Loss function
# ──────────────────────────────────────────────────────────────────────────────

def bridge_loss(
    z_pred  : torch.Tensor,
    z_target: torch.Tensor,
    mse_w   : float = 0.9,
    cos_w   : float = 0.1,
) -> tuple[torch.Tensor, dict]:
    """
    Combined MSE + cosine loss for Stage 2 training.

    Args:
        z_pred   : [B, 512] bridge output
        z_target : [B, 512] frozen encoder latent (from cache)
        mse_w    : weight for MSE term  (default 0.9)
        cos_w    : weight for cosine term (default 0.1)

    Returns:
        loss   : scalar
        metrics: dict with individual loss components for logging
    """
    import torch.nn.functional as F

    mse = F.mse_loss(z_pred, z_target)
    cos = 1.0 - F.cosine_similarity(z_pred, z_target, dim=-1).mean()

    loss = mse_w * mse + cos_w * cos

    metrics = {
        'loss_total' : loss.item(),
        'loss_mse'   : mse.item(),
        'loss_cos'   : cos.item(),
        'cos_sim'    : (1.0 - cos).item(),   # actual cosine similarity (higher = better)
    }
    return loss, metrics


# ──────────────────────────────────────────────────────────────────────────────
# Smoke test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print("\n[1/4] Building bridge...")
    bridge = TextToLatentBridge(jepa_d=512).to(device)

    print("\n[2/4] Verifying freeze pattern...")
    bridge.verify_freeze()

    print("\n[3/4] Parameter summary...")
    bridge.param_summary()

    print("\n[4/4] Forward pass smoke test...")
    texts  = [
        "a simple cylindrical shape",
        "a rectangular block with a slot cut through the middle",
        "a bracket with two mounting holes",
        "a hollow tube with flanged ends",
    ]

    # Training path: tokenize → forward (with grad)
    bridge.train()
    tokens = bridge.tokenize(texts, device)
    z_pred = bridge(tokens)
    print(f"  z_pred shape : {z_pred.shape}")    # [4, 512]
    print(f"  z_pred std   : {z_pred.std():.3f}")
    assert z_pred.requires_grad, "z_pred must have grad in training mode"

    # Fake target (would come from latent cache in real training)
    z_target = torch.randn_like(z_pred).detach()
    loss, metrics = bridge_loss(z_pred, z_target)
    loss.backward()
    print(f"  loss         : {metrics['loss_total']:.4f}")
    print(f"  cos_sim      : {metrics['cos_sim']:.4f}")
    print(f"  backward     : OK")

    # Inference path: encode_text (@no_grad)
    bridge.eval()
    z_inf = bridge.encode_text(texts, device)
    print(f"  encode_text  : {z_inf.shape}, requires_grad={z_inf.requires_grad}")
    assert not z_inf.requires_grad, "encode_text must not track grad"

    # Param groups check
    groups = bridge.param_groups()
    assert len(groups) == 2
    print(f"\n  Param groups : {[g['name'] for g in groups]}")
    print(f"  LR projector : {groups[0]['lr']}")
    print(f"  LR clip      : {groups[1]['lr']}")

    print("\nAll checks passed.")