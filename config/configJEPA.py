"""
CAD-JEPA Configuration
Attribute names aligned with DeepCAD convention so CADEmbedding/Encoder work directly.
"""

class ConfigJEPA:
    # ── Encoder (names match DeepCAD's CADEmbedding + Encoder) ────────────
    n_commands      = 6       # LINE ARC CIRCLE EOS SOL EXT
    # d_model         = 256     # start with 256 (same as DeepCAD AE); scale to 768 later
    d_model         = 512     # start with 256 (same as DeepCAD AE); scale to 768 later
    n_heads         = 8
    # dim_feedforward = 1024    # 4 × d_model
    dim_feedforward = 2048    # 4 × d_model
    dropout         = 0.1
    n_layers        = 12      # context encoder depth
    args_dim        = 256     # ARGS_DIM from macro.py
    n_args          = 16      # N_ARGS from macro.py
    max_total_len   = 60      # MAX_TOTAL_LEN from macro.py
    max_n_loops     = 6       # N_P  – max sketch loops per extrusion
    max_n_curves    = 6       # N_C  – max curves per loop

    # ── Predictor ──────────────────────────────────────────────────────────
    # predictor_d       = 128   # narrow: half of d_model
    predictor_d       = 256   # narrow: half of d_model
    predictor_layers  = 3
    predictor_heads = 8

    # ── EMA ────────────────────────────────────────────────────────────────
    ema_tau           = 0.996
    ema_tau_start     = 0.990
    ema_tau_warmup    = 40

    # ── Masking ────────────────────────────────────────────────────────────
    mask_ratio        = 0.50
    min_blocks_visible = 1

    # ── Training ───────────────────────────────────────────────────────────
    epochs            = 300
    batch_size        = 256
    lr                = 1.5e-4
    lr_min            = 1e-6
    lr_warmup_epochs  = 40
    weight_decay      = 0.05
    grad_clip         = 1.0

    # ── Collapse prevention ────────────────────────────────────────────────
    vicreg_lambda_v   = 25.0
    vicreg_lambda_c   = 1.0
    # vicreg_lambda_v = 5.0    # was 25.0
    # vicreg_lambda_c = 0.1    # was 1.0
    rank_threshold    = 0.70

    # ── Data ───────────────────────────────────────────────────────────────
    data_root         = '/content'
    augment           = False
    num_workers       = 4
    train_split_key   = 'train'
    val_split_key     = 'validation'

    # ── Checkpointing ──────────────────────────────────────────────────────
    ckpt_dir          = './checkpoints/pretrain'
    log_dir           = './logs/pretrain'
    model_dir         = './checkpoints/pretrain'
    save_every        = 50
    log_interval      = 50

    # ── Stage 2 / 3 (fill in later) ───────────────────────────────────────
    dim_z             = 256   # same as d_model (no bottleneck in JEPA)
    n_layers_decode   = 4
    