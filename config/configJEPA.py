"""
CAD-JEPA Pretraining Configuration
All values taken directly from the paper unless marked TUNE.
"""


class ConfigJEPA:
    # Model
    encoder_layers    = 12
    encoder_dim       = 768
    encoder_heads     = 8
    encoder_dropout   = 0.0
    predictor_layers  = 3
    predictor_dim     = 256

    # Tokenizer  (DeepCAD cad_vec format)
    num_commands      = 8        # Line Arc Circle EOS SOL SOF SOE Extrude
    num_params        = 16
    param_vocab_size  = 257      # 0-255 quantized + 256 padding
    max_seq_len       = 80

    # EMA
    ema_tau           = 0.996
    ema_tau_start     = 0.990
    ema_tau_warmup    = 40       # epochs

    # Masking
    mask_ratio        = 0.50
    min_blocks_visible = 1

    # Training
    epochs            = 300
    batch_size        = 256
    lr                = 1.5e-4
    lr_warmup_epochs  = 20
    weight_decay      = 0.05
    grad_clip         = 1.0

    # Collapse prevention (VICReg — safety net, rarely fires)
    vicreg_lambda_v   = 25.0
    vicreg_lambda_c   = 1.0
    rank_threshold    = 0.70

    # Data
    data_root         = "./data/cad_json"
    vec_root          = "./data/cad_vec"
    train_split       = "./data/train_val_test_split.json"
    num_workers       = 8

    # Checkpointing
    ckpt_dir          = "./checkpoints/pretrain"
    save_every        = 50
    log_every         = 10

    # Stage 2 text bridge
    clip_model        = "openai/clip-vit-base-patch32"
    bridge_hidden_dim = 1024
    bridge_lr         = 1e-4
    bridge_epochs     = 50
    encoded_z_path    = "./data/encoded_z.h5"

    # Stage 3 decoder
    decoder_layers    = 4
    decoder_dim       = 512
    decoder_heads     = 8
    decoder_lr        = 1e-4
    decoder_epochs    = 100
    beam_size         = 5
