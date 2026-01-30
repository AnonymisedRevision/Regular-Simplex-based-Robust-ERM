
# SR-ERM (Regular Simplex Shifting)

This repo is a project-ready reference implementation of the **sampling-based pathway** in Appendix C
(Algorithms 1–3) of *Regular Simplex Shifting*, adapted to an image dataset already split into
`train/` and `test/` folders (shifted test). The backbone is **MobileNetV3-Small** pretrained on ImageNet,
used as both **encoder** (feature extractor) and **final predictor**.

Key implementation points:
- **Algorithm 2**: `τ_R` defaults to mean radial spread around the latent centroid; preventive displacement
  `v = (2-ε) τ_R / d̂`; per-vertex subsets are built with K-NN in latent space.  
- **SR-ERM objective**: `min_θ max_k L̂_{S_k}(f_θ)` (Eq. 38).  
- **Dual monitors**: `Γ̂_train` for early stopping and `Γ̂_val` for model selection (Def. 16 / Cond. 1).

## Dataset layout
Expected layout (as the provided `SKIN.zip`):
```
SKIN/
  train/
    BENIGN/
    MALIGNANT/
  test/
    BENIGN/
    MALIGNANT/
```

## Install
```bash
git lfs install
git clone https://github.com/AnonymisedRevision/Regular-Simplex-based-Robust-ERM.git
cd .\Regular-Simplex-based-Robust-ERM
git lfs pull
pip install -r requirements.txt
```

## Run SR-ERM training (end-to-end)
```bash
python -m srerm_skin.scripts.train_srerm --data_root SKIN.zip --out_dir runs/skin_srerm --latent_dim 10 --vertex_k 800 --epsilon 0.01 --steps_per_epoch 100 --batch_size 32 --balanced_loss --device cuda
```


This will:
1) train baseline ERM on `S_{-1}` (unshifted train split)  
2) compute embeddings and build simplex subsets `{S_k}` (k=0..d̂)  
3) train SR-ERM with the hard-max step and `Γ̂` monitors  
4) evaluate baseline and SR-ERM on the shifted `test/` split.

## Outputs
- `out_dir/checkpoints/` — baseline + SR-ERM checkpoints
- `out_dir/subsets.json` — the sampled indices per vertex
- `out_dir/metrics.json` — baseline and SR-ERM metrics (acc, AUC, F1, confusion matrix)
- `out_dir/tensorboard/` — TensorBoard logs (optional; enabled by default)

