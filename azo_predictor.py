"""Self-contained AZO lambda_max (nm) predictor for the Streamlit app.

Loads the pre-trained wide-and-deep MLP (the one that gives ~27 nm MAE on the
18-molecule new-azo holdout) plus the train-set preprocessing artifacts it was
fit with, and applies the *identical* featurization / preprocessing path that
`pytorch_mlp.predict_single_value` uses, so a user-supplied SMILES is treated
exactly like the hold-out molecules.

Pipeline (must match training in pytorch_mlp.py / azo_model.py):
  SMILES
   -> ion-clean (fingerprints.clean_smiles)
   -> RDKit mol objects (skfp MolFromSmilesTransformer)
   -> 2D Mordred descriptors (skfp MordredFingerprint, use_3D=False) -> 1613 cols
   -> inf->nan, keep_mask column select (828 cols)
   -> train-set median imputation of NaNs
   -> standardize with train feature_mean / feature_std
   -> nan_to_num, clip +/-100
   -> PCA projection to 54 comps (train components_ / mean_)
   -> MLP forward pass -> predicted Lmax in nm (raw target, no inverse)

The three artifact files live next to this script:
  saved_model_pca.pt      -- model state_dict
  prep_stats_pca.npz      -- keep_mask, medians, feature_mean/std, PCA comps/mean
  MLP_model_params.txt    -- architecture params (neurons, input_dims, ...)
"""
from __future__ import annotations

import os

# Single-thread the numeric backends BEFORE numpy/torch are imported. We force
# (not setdefault) so a host that pre-sets OMP_NUM_THREADS -- e.g. Streamlit
# Cloud's base image -- still runs single-threaded. A multithreaded OpenMP/MKL
# runtime makes fork() unsafe: a child that segfaults can corrupt the parent's
# OpenMP state and take the parent down too. With one thread there is no pool
# to corrupt.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import warnings
import multiprocessing as mp
from typing import List, Union

import numpy as np
import torch
import torch.nn as nn

# Mordred can emit benign RuntimeWarnings for uncomputable descriptors; the
# preprocessing imputes them, so silence the noise.
warnings.filterwarnings("ignore", message=".*encountered in matmul.*",
                        category=RuntimeWarning)

_HERE = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------- #
# Model architecture -- a byte-for-byte match of pytorch_mlp.MLP_Model so the
# saved state_dict loads with weights_only=True. Kept self-contained here so
# the app does not depend on the training codebase.
# --------------------------------------------------------------------------- #
class AZO_MLP(nn.Module):
    """Wide-and-deep MLP: BatchNorm -> input sigmoid(250) -> 1 sigmoid hidden
    (250) -> [skip: cat raw input] -> linear out(1). Matches the trained model
    (num_hidden_layers=1 -> 2 sigmoid hidden layers total, skip_connection=True).
    """

    def __init__(self, neurons: int, input_dims: int, num_hidden_layers: int,
                 num_classes: int = 1, skip_connection: bool = True):
        super().__init__()
        self.neurons = neurons
        self.input_dims = input_dims
        self.num_hidden_layers = num_hidden_layers
        self.num_classes = num_classes
        self.skip_connection = skip_connection

        self.batchnorm = nn.BatchNorm1d(self.input_dims)
        self.linear_input = nn.Sequential(
            nn.Linear(self.input_dims, self.neurons),
            nn.Sigmoid())
        self.linear_sigmoid = nn.Sequential(
            nn.Linear(self.neurons, self.neurons),
            nn.Sigmoid())
        out_in = self.neurons + (self.input_dims if self.skip_connection else 0)
        self.linear_output = nn.Linear(out_in, 1)
        # Present in the trained model (classifier path unused here); kept so the
        # state_dict keys match. LogSoftmax has no params so dim choice is inert.
        self.linear_class_out = nn.Linear(out_in, self.num_classes)
        self.classifier_output = nn.LogSoftmax(dim=1)

    def forward(self, x):
        skip = x
        x = self.batchnorm(x)
        x = self.linear_input(x)
        for _ in range(self.num_hidden_layers):
            x = self.linear_sigmoid(x)
        if self.skip_connection:
            x = torch.cat([x, skip], dim=1)
        return self.linear_output(x)


# --------------------------------------------------------------------------- #
# Featurization -- replicates fingerprints.get_fingerprints.transform() for the
# 2D Mordred path (no conformer generation). A fresh MordredFingerprint(use_3D=
# False) yields exactly the same 1613 columns in the same order as the cached
# training featurizer, so the saved keep_mask aligns column-for-column.
# --------------------------------------------------------------------------- #
_IONS_TO_CLEAN = ['[Na+].', '[Cl-].', '[Ca+].', '[K+].', '.[Na+]', '.[Cl-]',
                  '.[Ca+]', '.[K+]', '[Br-].', '[I-].', '[F-].', '.[Br-]',
                  '.[I-]', '.[F-]']


def _clean_smiles(smiles_list: List[str]) -> List[str]:
    """Strip common counter-ions (mirrors fingerprints.clean_smiles)."""
    out = []
    for smi in smiles_list:
        for ion in _IONS_TO_CLEAN:
            smi = smi.replace(ion, "")
        out.append(smi)
    return out


class Featurizer:
    """SMILES -> 1613-dim 2D Mordred descriptor matrix. Stateless after init.

    Robust to invalid SMILES: each input is parsed with RDKit and only valid
    molecules are featurized; invalid inputs get an all-NaN row (later
    median-imputed by the predictor into a "default" prediction) and are
    reported back so the UI can flag them, instead of crashing the whole batch.
    """

    def __init__(self, n_jobs: int = -1):
        from skfp.fingerprints import MordredFingerprint
        self._mordred = MordredFingerprint(use_3D=False, n_jobs=n_jobs)

    def _transform_safe(self, mols):
        """Featurize molecules one at a time, isolating failures.

        Mordred can raise on a single pathological-but-parseable molecule, and
        because it processes the whole batch at once one bad input would
        otherwise take the entire batch down. This fallback computes each
        molecule independently and returns a descriptor array aligned 1:1 with
        `mols` (an all-NaN row for any that crashed) plus the set of positions
        that crashed, so they can be reported as invalid rather than aborting.
        """
        rows, crashed = [], set()
        n_cols = None
        for i, m in enumerate(mols):
            try:
                r = np.asarray(self._mordred.transform([m]), dtype=np.float64)
                if n_cols is None:
                    n_cols = r.shape[1]
                rows.append(r[0])
            except Exception:
                crashed.add(i)
        if n_cols is None:  # every molecule crashed
            return np.full((len(mols), 0), np.nan, dtype=np.float64), crashed
        Xv = np.full((len(mols), n_cols), np.nan, dtype=np.float64)
        kept = [i for i in range(len(mols)) if i not in crashed]
        Xv[kept] = np.vstack(rows)
        return Xv, crashed

    def transform(self, smiles_list: Union[str, List[str]]):
        """Return (X, invalid_idx).

        X: float64 array shape (n_smiles, 1613) with all-NaN rows for inputs that
        failed to parse (or that crashed Mordred). invalid_idx: list of
        positions whose SMILES did not parse or could not be featurized.
        """
        from rdkit import Chem
        if isinstance(smiles_list, str):
            smiles_list = [smiles_list]
        cleaned = _clean_smiles(smiles_list)

        # Parse each SMILES; keep positions and mol objects for the valid ones.
        valid_pos, valid_mols, invalid_idx = [], [], []
        for i, smi in enumerate(cleaned):
            m = Chem.MolFromSmiles(smi)
            if m is None:
                invalid_idx.append(i)
            else:
                valid_pos.append(i)
                valid_mols.append(m)

        X = np.full((len(cleaned), 1), np.nan, dtype=np.float64)
        if valid_mols:
            try:
                Xv = np.asarray(self._mordred.transform(valid_mols),
                                dtype=np.float64)
            except Exception:
                # A pathological molecule made Mordred throw on the whole batch;
                # fall back to per-molecule featurization so one bad input
                # doesn't kill the rest. Xv is 1:1 with valid_mols (NaN rows for
                # the crashed ones), so a single keep-filter aligns it with the
                # surviving valid_pos.
                Xv, crashed = self._transform_safe(valid_mols)
                for k in crashed:
                    invalid_idx.append(valid_pos[k])
                keep = [k for k in range(len(valid_mols)) if k not in crashed]
                valid_pos = [valid_pos[k] for k in keep]
                Xv = Xv[keep]
            if Xv.size:
                X = np.full((len(cleaned), Xv.shape[1]), np.nan, dtype=np.float64)
                X[valid_pos] = Xv
        return X, invalid_idx


# --------------------------------------------------------------------------- #
# Predictor -- ties featurizer + saved preprocessing stats + model together.
# --------------------------------------------------------------------------- #
def _isolated_predict_worker(model_dir, smiles_list, conn):
    """Prediction in a *spawned* (fresh) child process.

    A malformed SMILES can segfault the C++ layer (RDKit parser or Mordred's
    descriptor engine); that is a SIGSEGV, which Python try/except cannot catch.
    Running the work in a separate process means a crash kills only this child
    -- the parent detects the non-zero exit code and reports a graceful error.

    We use 'spawn' (not 'fork') so the child is a clean interpreter with no
    inherited OpenMP/MKL/torch threads. Forking from a multithreaded parent is
    unsafe: a segfaulting child can corrupt the parent's OpenMP runtime and
    take the parent (the Streamlit server) down with it. The child builds its
    own AZOPredictor from disk so nothing un-picklable (the torch model) has to
    cross the process boundary.
    """
    try:
        torch.set_num_threads(1)
        predictor = AZOPredictor(model_dir=model_dir)
        pred, invalid = predictor.predict(smiles_list)
        conn.send(("ok", pred, invalid))
    except Exception as e:  # any *Python* error surfaces as a friendly message
        conn.send(("err", f"{type(e).__name__}: {e}"))
    finally:
        conn.close()


def _get_spawn_ctx():
    # 'spawn' is the safe choice on Linux/macOS; fall back to the platform
    # default if it is somehow unavailable.
    try:
        return mp.get_context("spawn")
    except ValueError:
        return mp.get_context()


class AZOPredictor:
    def __init__(self, model_dir: str = _HERE, device: str = "cpu"):
        self.device = torch.device(device)
        self.model_dir = model_dir
        self._load_params()
        self._load_prep_stats()
        self._load_model()
        self._featurizer = None  # lazily built (Mordred import is slow)

    # -- artifact loading --
    def _load_params(self):
        path = os.path.join(self.model_dir, "MLP_model_params.txt")
        with open(path, "r") as f:
            lines = f.readlines()
        self.neurons = int(lines[0].split()[1])
        self.input_dims = int(lines[1].split()[1])
        self.num_hidden_layers = int(lines[2].split()[1])
        self.skip_connection = (lines[5].split(":")[1].strip() == "True"
                                if len(lines) > 5 else False)

    def _load_prep_stats(self):
        d = np.load(os.path.join(self.model_dir, "prep_stats_pca.npz"))
        self.keep_mask = d["keep_mask"]
        self.medians = d["medians"]
        self.feature_mean = d["feature_mean"]
        self.feature_std = d["feature_std"]
        self.pca_components = d["pca_components"] if bool(d["has_pca"]) else None
        self.pca_mean = d["pca_mean"] if bool(d["has_pca"]) else None

    def _load_model(self):
        self.model = AZO_MLP(
            neurons=self.neurons, input_dims=self.input_dims,
            num_hidden_layers=self.num_hidden_layers,
            skip_connection=self.skip_connection).to(self.device)
        sd = torch.load(os.path.join(self.model_dir, "saved_model_pca.pt"),
                        map_location=self.device, weights_only=True)
        self.model.load_state_dict(sd)
        self.model.eval()

    @property
    def featurizer(self):
        if self._featurizer is None:
            self._featurizer = Featurizer()
        return self._featurizer

    # -- preprocessing (exact replay of pytorch_mlp.predict_single_value) --
    def _preprocess(self, X: np.ndarray) -> np.ndarray:
        X = np.where(np.isinf(X), np.nan, X)
        X = X[:, self.keep_mask]
        inds = np.where(np.isnan(X))
        if inds[0].size:
            X = X.copy()
            X[inds] = np.take(self.medians, inds[1])
        std = np.asarray(self.feature_std, dtype=np.float64).copy()
        std[std == 0] = 1.0
        X = (X - np.asarray(self.feature_mean, dtype=np.float64)) / std
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X = np.clip(X, -100.0, 100.0)
        if self.pca_components is not None:
            comps = np.asarray(self.pca_components, dtype=np.float64)
            pmean = np.asarray(self.pca_mean, dtype=np.float64)
            X = (X - pmean) @ comps.T
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return X

    # -- public API --
    def predict(self, smiles_list: Union[str, List[str]]):
        """Predict lambda_max (nm) for each SMILES.

        Returns (predictions, invalid_idx):
          predictions -- float array, one per input; NaN for SMILES that did
            not parse (those are skipped, not fed to the model).
          invalid_idx -- list of input positions whose SMILES failed to parse.
        """
        if isinstance(smiles_list, str):
            smiles_list = [smiles_list]
        X, invalid_idx = self.featurizer.transform(smiles_list)
        valid_mask = np.ones(len(smiles_list), dtype=bool)
        valid_mask[invalid_idx] = False
        preds = np.full(len(smiles_list), np.nan, dtype=np.float64)
        if valid_mask.any():
            Xv = self._preprocess(X[valid_mask])
            t = torch.from_numpy(np.ascontiguousarray(Xv, dtype=np.float32)) \
                .to(self.device)
            with torch.no_grad():
                pv = self.model(t).detach().cpu().numpy().reshape(-1)
            preds[valid_mask] = pv
        return preds, invalid_idx


def predict_isolated(model_dir, smiles_list, timeout: float = 240.0):
    """Run prediction for a batch in a spawned child process.

    A malformed SMILES can segfault the C++ engine (SIGSEGV), which Python
    cannot catch and which would kill the host process. This runs the work in a
    spawned child so a crash is contained: on a segfault or hang we raise a
    RuntimeError with a friendly message, rather than taking the process down.
    """
    if isinstance(smiles_list, str):
        smiles_list = [smiles_list]
    ctx = _get_spawn_ctx()

    parent_conn, child_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_isolated_predict_worker,
                       args=(model_dir, smiles_list, child_conn), daemon=True)
    proc.start()
    child_conn.close()  # parent only reads

    proc.join(timeout)
    if proc.is_alive():
        proc.terminate()
        proc.join(5)
        raise RuntimeError(
            "Prediction timed out — a SMILES may have hung the RDKit/Mordred "
            "engine. Please remove it and try again.")
    if proc.exitcode in (None, 0):
        if not parent_conn.poll():
            raise RuntimeError("Prediction produced no output — please try again.")
        tag, *rest = parent_conn.recv()
        if tag == "ok":
            return rest[0], rest[1]
        raise RuntimeError(rest[0])
    # Non-zero exit code -> child was killed by a signal (e.g. -11 SIGSEGV).
    raise RuntimeError(
        "Prediction crashed — a malformed SMILES tripped the RDKit/Mordred "
        "C++ engine (segmentation fault). Please fix or remove it and try "
        "again.")


def predict_safe(model_dir, smiles_list, timeout: float = 240.0,
                 per_item_timeout: float = 60.0):
    """Resilient prediction for untrusted input (the Streamlit entry point).

    Tries the whole batch in one spawned child first (fast path). If that
    crashes or times out -- meaning at least one SMILES is pathological --
    falls back to running each SMILES in its own spawned child, so a single bad
    input is reported as invalid instead of costing the predictions for the
    whole batch. Returns (preds, invalid_idx) where invalid_idx covers both
    parse failures and per-molecule crashes.
    """
    if isinstance(smiles_list, str):
        smiles_list = [smiles_list]
    try:
        return predict_isolated(model_dir, smiles_list, timeout=timeout)
    except RuntimeError:
        preds = np.full(len(smiles_list), np.nan, dtype=np.float64)
        invalid = []
        for i, smi in enumerate(smiles_list):
            try:
                pv, _ = predict_isolated(model_dir, [smi],
                                         timeout=per_item_timeout)
                preds[i] = pv[0]
            except RuntimeError:
                invalid.append(i)
        return preds, invalid


if __name__ == "__main__":
    # Quick CLI sanity check: print predictions for any SMILES passed as args.
    import sys
    smis = sys.argv[1:] or ["N=Nc1ccccc1"]
    p = AZOPredictor()
    out, invalid = p.predict(smis)
    for i, (smi, val) in enumerate(zip(smis, out)):
        if i in set(invalid):
            print(f"{smi}\tPARSE FAILED")
        else:
            print(f"{smi}\t{val:.1f} nm")