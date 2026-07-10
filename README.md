# AZO λmax predictor — Streamlit app bundle

A self-contained folder for serving the pre-trained AZO-dye **λmax** (π→π*
absorption maximum, nm) MLP as a Streamlit web app. Drop in a SMILES string
(one or many) and get a predicted λmax per molecule.

This bundle carries the **exact production model** — the wide-and-deed MLP that
achieves **MAE ≈ 27.2 nm (R² ≈ 0.69)** on the 18-molecule *new-azo* out-of-
distribution holdout. The featurization + preprocessing path here is byte-for-byte
identical to `pytorch_mlp.predict_single_value` in the parent repo, and the
holdout MAE is reproduced exactly from these artifacts (verified: 27.24 nm).

## Files in this folder

| File | What it is |
|---|---|
| `saved_model_pca.pt` | Trained MLP weights (PyTorch `state_dict`). |
| `prep_stats_pca.npz` | Train-set preprocessing artifacts: column `keep_mask`, per-column imputation `medians`, standardization `feature_mean`/`feature_std`, and PCA `pca_components`/`pca_mean`. |
| `MLP_model_params.txt` | Model architecture params (neurons=250, input_dims=54, num_hidden_layers=1, skip_connection=True). |
| `azo_predictor.py` | Self-contained inference module: defines the model class, the Mordred featurizer, and the `AZOPredictor` that replays the exact training-time preprocessing. Also runnable as a CLI. |
| `app.py` | The Streamlit UI. |
| `requirements.txt` | Minimal runtime deps. |

The three artifact files (`*.pt`, `*.npz`, `*.txt`) are the only model outputs
needed — the app does **not** depend on the parent repo's training code, the
4 MB `621-azo_Mordred.pkl` descriptor cache, or any CSV.

## The model & how it is applied

**Architecture** (`AZO_MLP` in `azo_predictor.py`, matching `pytorch_mlp.MLP_Model`):
BatchNorm1d → Linear+Sigmoid(54→250) → Linear+Sigmoid(250→250) →
[skip: concatenate raw input] → Linear(304→1). `skip_connection=True` routes the
54-dim PCA input straight to the output layer alongside the 250-dim hidden
representation. Trained with SGD, lr=2e-3, weight_decay=0.2, 2500 epochs, MSE,
on 621 curated azo dyes (Murcko scaffold split, seed 132).

**Prediction pipeline** (must match training; implemented in
`AZOPredictor._preprocess`):

1. **Featurize**: SMILES → ion-clean → RDKit mol → 2D Mordred descriptors
   (`skfp.MordredFingerprint(use_3D=False)`). This yields a fixed 1613-column
   matrix in a fixed order — the same columns/order the model was trained on, so
   the saved `keep_mask` aligns column-for-column. (Verified: a fresh
   `MordredFingerprint` reproduces the cached training featurizer's 1613 columns
   exactly.)
2. **Clean/select**: `inf → nan`, then select the 828 columns flagged by the
   train-fit `keep_mask`.
3. **Impute** remaining NaNs with the train-set per-column `medians`.
4. **Standardize** with train `feature_mean`/`feature_std` (zero-std columns
   guarded to 1.0).
5. **Sanitize**: `nan_to_num`, clip to ±100 (caps out-of-distribution extremes;
   no-op in-distribution).
6. **PCA project**: `(X − pca_mean) @ pca_components.T` → 54 components (95%
   explained variance, fit on train only).
7. **Forward pass** through the MLP → predicted λmax in **nm** (targets were raw
   Lmax, no log transform, so no inverse is applied).

Invalid SMILES are caught at parse time: they are skipped (not fed to the
model), reported in the UI, and get a NaN prediction rather than crashing the
whole batch.

## Quickstart

### Option A — reuse the parent repo's `ml-env` (recommended, exact deps)

The parent repo already has a pinned venv with the right versions of
torch / scikit-fingerprints / mordredcommunity / rdkit. Just add Streamlit:

```bash
cd <parent-repo>
./ml-env/bin/pip install streamlit
./ml-env/bin/streamlit run streamlit_app/app.py
```

### Option B — fresh environment

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r streamlit_app/requirements.txt
streamlit run streamlit_app/app.py
```

Then open the printed `http://localhost:8501` URL. Paste SMILES (one per line),
click **Predict**.

> macOS note: the predictor pins `OMP_NUM_THREADS=1` internally because Mordred's
> OpenMP backend can segfault with unconstrained threads on macOS. This is set
> in `azo_predictor.py` via `os.environ.setdefault`, so it won't override a value
> you've already exported.

## Using the predictor without Streamlit

`azo_predictor.py` is a plain Python module and a CLI:

```python
from azo_predictor import AZOPredictor
p = AZOPredictor()                       # loads the 3 artifacts from this dir
preds, invalid = p.predict(["N=Nc1ccccc1", "bad_smiles"])
# preds -> array([418.5, nan]); invalid -> [1]
```

```bash
OMP_NUM_THREADS=1 python azo_predictor.py "N=Nc1ccccc1" "CNC1=CC=C(/N=N/C2=C([N+]([O-])=O)C=C([N+]([O-])=O)C=C2)C3=CC=CC=C31"
# N=Nc1ccccc1                       418.5 nm
# CNC1=...=C31                     540.5 nm
```

`AZOPredictor(model_dir=...)` can point elsewhere if you move the artifacts.

## Reproducing the holdout metric from this bundle

```python
import pandas as pd, numpy as np
from azo_predictor import AZOPredictor
df = pd.read_csv("../new-azo.csv")          # parent repo
p = AZOPredictor()
pred, _ = p.predict(df["SMILES"].tolist())
print(np.mean(np.abs(pred - df["Lmax"].values)))   # 27.24 nm
```

## Provenance / caveats

- **Training set**: 621 curated azo dyes, 2D Mordred descriptors, Murcko scaffold
  split (seed 132, 80/20). Targets are raw λmax (nm), no transform.
- **Holdout**: 18 new azo dyes in a *scaffold-novel* regime (the hardest case for
  this model). MAE 27.2 nm / R² 0.69. Individual errors range from <1 nm to ~70
  nm — the model is biased on the most scaffold-novel carbamate/OMe dyes. See the
  parent repo's `README.md` and memory notes for the full diagnostic history
  (router attempts, OMNI-P2x / TDDFT baselines, etc.).
- **Applicability domain**: azo dyes only. Predictions are **not** reliable for
  non-azo scaffolds or substituent patterns far from the 621 training dyes.
  Treat outputs as estimates, not measurements.
- **Why this MLP and not SVR/chemprop**: the Mordred MLP and SVR both reach ~27–28
  nm on new-azo; the MLP is shipped here because its `.pt` + `.npz` artifacts make
  for a clean stateless deployment. The polynomial-kernel SVR is equally accurate
  but is a fitted sklearn `Pipeline` that is less portable.

## Regenerating the artifacts

If you ever retrain in the parent repo (`python azo_model.py`), it writes
`saved_model_pca.pt`, `prep_stats_pca.npz`, and `MLP_model_params.txt` to the
repo root — copy those three into this folder to refresh the app's model.