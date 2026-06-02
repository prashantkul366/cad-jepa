# CAD-JEPA

Self-Supervised Design-Intent Representations via Masked Latent Prediction
over Parametric CAD Construction Sequences.

## Daily workflow

```
VS Code (local)  ->  git push  ->  Colab A100  ->  git pull + python train_jepa.py
```

## Colab setup (once per session)

```python
!git clone https://github.com/YOUR_USERNAME/cad-jepa.git
%cd cad-jepa
!pip install -r requirements.txt
!bash data/scripts/download_deepcad.sh
!git pull origin main
!python train_jepa.py
```

## File status

| File | Status |
|------|--------|
| cadlib/, json2vec.py, evaluation/ | Stable — from DeepCAD, do not modify |
| utils/schedulers.py, tensors.py   | Stable — from I-JEPA |
| model/ema.py, collapse_monitor.py, configJEPA.py | Complete |
| model/jepa_encoder.py, predictor.py | TODO |
| dataset/masks/semantic_block.py   | TODO |
| trainer/trainerJEPA.py train_step | TODO |
