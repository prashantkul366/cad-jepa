"""
inference/generate.py
Generate predictions for ALL test samples using any registered runner.
Run on Colab (GPU). Output: pred_vecs pkl for Windows eval.

Usage:
  # CAD-JEPA NAR (our model)
  python -m inference.generate \
    --runner     cad_jepa_nar \
    --exp_name   "CAD-JEPA-NAR-100pct" \
    --decoder_ckpt  /content/drive/MyDrive/cad-jepa-checkpoints/decoder_nar/best.pt \
    --bridge_ckpt   /content/drive/MyDrive/cad-jepa-checkpoints/bridge/best.pt \
    --correction    /content/drive/MyDrive/cad-jepa-checkpoints/bridge/correction.pt \
    --n_ext_head    /content/drive/MyDrive/cad-jepa-checkpoints/n_ext_head_v2.pt \
    --output     /content/drive/MyDrive/pred_cad_jepa_nar_100pct.pkl

  # New architecture (just change --runner and --decoder_ckpt)
  python -m inference.generate \
    --runner     cad_jepa_nar \
    --exp_name   "CAD-JEPA-NAR-v3-hierarchical" \
    --decoder_ckpt  /content/drive/MyDrive/cad-jepa-checkpoints/decoder_v3/best.pt \
    ...
"""

import argparse, sys, os, pickle, json
import numpy as np
from tqdm import tqdm

sys.path.insert(0, '/content/cad-jepa')

# ── Registered runners ────────────────────────────────────────────────────────
RUNNERS = {
    'cad_jepa_nar' : 'inference.runners.cad_jepa_nar.CADJEPANARRunner',
    'cad_jepa_ar'  : 'inference.runners.cad_jepa_ar.CADJEPAARRunner',
    'text2cad_hf'  : 'inference.runners.text2cad_hf.Text2CADHFRunner',
    # Add new runners here:
    # 'my_new_model' : 'inference.runners.my_model.MyRunner',
}

def load_runner(name: str):
    path = RUNNERS[name]
    mod, cls = path.rsplit('.', 1)
    import importlib
    return getattr(importlib.import_module(mod), cls)

# ── Args ──────────────────────────────────────────────────────────────────────
p = argparse.ArgumentParser()
p.add_argument('--runner',      required=True, choices=list(RUNNERS.keys()))
p.add_argument('--exp_name',    required=True, help='Experiment name (for tracking)')
p.add_argument('--output',      required=True, help='Output pkl path')
p.add_argument('--annotations', default='/content/drive/MyDrive/cad-jepa-data/text2cad_annotations.json')
p.add_argument('--split',       default='/content/train_val_test_split.json')
p.add_argument('--text_level',  default='intermediate', help='beginner/intermediate/expert')
p.add_argument('--n',           type=int, default=None)
# Runner-specific args (passed through to runner.setup())
p.add_argument('--decoder_ckpt',  default=None)
p.add_argument('--bridge_ckpt',   default='/content/drive/MyDrive/cad-jepa-checkpoints/bridge/best.pt')
p.add_argument('--correction',    default='/content/drive/MyDrive/cad-jepa-checkpoints/bridge/correction.pt')
p.add_argument('--n_ext_head',    default='/content/drive/MyDrive/cad-jepa-checkpoints/n_ext_head_v2.pt')
p.add_argument('--hf_model_id',   default=None)
p.add_argument('--device',        default='cuda')
args = p.parse_args()

# ── Setup runner ──────────────────────────────────────────────────────────────
print(f"Runner: {args.runner}  |  Experiment: {args.exp_name}")
RunnerClass = load_runner(args.runner)
runner = RunnerClass()
runner.setup(
    decoder_ckpt   = args.decoder_ckpt,
    bridge_ckpt    = args.bridge_ckpt,
    correction_path = args.correction,
    n_ext_head_path = args.n_ext_head,
    hf_model_id    = args.hf_model_id,
    device         = args.device,
)

# ── Load data ─────────────────────────────────────────────────────────────────
import json as _json
annotations = _json.load(open(args.annotations))
split_data  = _json.load(open(args.split))
test_uids   = split_data['test']
if args.n:
    test_uids = test_uids[:args.n]

# ── Generate ──────────────────────────────────────────────────────────────────
pred_vecs = {}; failed = []

for uid in tqdm(test_uids):
    ann  = annotations.get(uid, {})
    text = ann.get(args.text_level) or ann.get('intermediate') or \
           ann.get('beginner') or ann.get('expert')
    if not text:
        failed.append(uid); continue

    result = runner.generate_one(uid, text)
    if result is not None:
        pred_vecs[uid] = result.tolist()   # list for cross-numpy compat
    else:
        failed.append(uid)

print(f"\nGenerated: {len(pred_vecs)}/{len(test_uids)}  Failed: {len(failed)}")

# ── Save with metadata ────────────────────────────────────────────────────────
with open(args.output, 'wb') as f:
    pickle.dump(pred_vecs, f, protocol=2)

# Save experiment metadata alongside
meta = {
    'exp_name'    : args.exp_name,
    'runner'      : args.runner,
    'decoder_ckpt': args.decoder_ckpt,
    'n_generated' : len(pred_vecs),
    'n_failed'    : len(failed),
    'text_level'  : args.text_level,
    'output'      : args.output,
}
meta_path = args.output.replace('.pkl', '_meta.json')
with open(meta_path, 'w') as f:
    _json.dump(meta, f, indent=2)

print(f"Saved: {args.output}")
print(f"Meta:  {meta_path}")