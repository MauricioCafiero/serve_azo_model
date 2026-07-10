"""Persistent subprocess worker for the AZO Streamlit app.

Started ONCE by app.py at module import -- a single fork of the Streamlit
parent -- rather than a fork on every Predict click. (Repeated forking of
Streamlit's multithreaded parent, after numpy/OpenMP has been used there, was
crashing the server on the *second* run.) Communicates over stdin/stdout with
newline-delimited JSON. stdout (fd 1) is reserved for responses only; all
library noise (RDKit parse errors, torch/loky warnings) is sent to stderr so it
can't corrupt the line protocol.

If this process segfaults on a bad input, only it dies: app.py sees the dead
pipe, restarts the worker, and retries -- the Streamlit server stays up.

Requests (one JSON object per line):
  {"mode": "predict", "model_dir": "...", "smiles": ["...", ...]}
  {"mode": "render",  "items": [["smiles", "legend"], ...]}

Responses (one JSON object per line):
  {"pred": [...], "invalid": [...]}     # predict (NaN for unparseable)
  {"png": "<base64>"}                    # render  ("" on any failure)
"""
import json
import os
import sys

# Single-thread the numeric backends before importing anything heavy.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

# Reserve fd 1 for responses; route everything else to stderr so RDKit/torch
# prints can't break the newline-delimited JSON protocol.
_real_stdout = os.fdopen(os.dup(1), "w", buffering=1)
sys.stdout = sys.stderr

_predictor = None  # lazy: load model on first predict, keep it across calls


def _get_predictor(model_dir):
    global _predictor
    if _predictor is None:
        import torch
        torch.set_num_threads(1)
        from azo_predictor import AZOPredictor
        # n_jobs=1 keeps Mordred single-process inside this worker.
        _predictor = AZOPredictor(model_dir=model_dir or _HERE, n_jobs=1)
    return _predictor


def _predict(req):
    p = _get_predictor(req.get("model_dir"))
    pred, invalid = p.predict(req["smiles"])
    # tolist() emits NaN tokens for unparseable rows; Python's json.loads
    # accepts NaN by default, so the parent reads them back as float NaN.
    return {"pred": pred.tolist(), "invalid": [int(i) for i in invalid]}


def _render(req):
    """Draw each molecule in its OWN fresh drawer (Draw.MolToImage -> PIL),
    composite onto an explicit white grid with Pillow, return PNG base64.

    A shared drawer drawing multiple molecules (DrawMolecules) produced an
    all-black second panel on Streamlit Cloud's cairo build -- drawing each
    molecule separately avoids that cross-molecule state corruption, and the
    explicit white RGB composite guarantees the background can't be black.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, Draw
        from PIL import Image
        import io
        import base64

        mols, legs = [], []
        for smi, leg in req["items"]:
            m = Chem.MolFromSmiles(smi)
            if m is None:
                continue
            try:
                AllChem.Compute2DCoords(m)
            except Exception:
                pass
            mols.append(m)
            legs.append(leg)
        if not mols:
            return {"png": ""}

        cols = min(4, len(mols))
        sub = (260, 200)
        rows = (len(mols) + cols - 1) // cols
        grid = Image.new("RGB", (cols * sub[0], rows * sub[1]), (255, 255, 255))
        for idx, (m, leg) in enumerate(zip(mols, legs)):
            panel = Draw.MolToImage(m, size=sub, legend=leg)  # fresh drawer each
            if panel.mode != "RGB":
                panel = panel.convert("RGB")
            r, c = divmod(idx, cols)
            grid.paste(panel, (c * sub[0], r * sub[1]))
        buf = io.BytesIO()
        grid.save(buf, format="PNG")
        return {"png": base64.b64encode(buf.getvalue()).decode()}
    except Exception:
        return {"png": ""}


def main():
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            mode = req.get("mode")
            if mode == "predict":
                result = _predict(req)
            elif mode == "render":
                result = _render(req)
            else:
                result = {"error": f"unknown mode: {mode}"}
        except Exception as e:
            result = {"error": f"{type(e).__name__}: {e}"}
        _real_stdout.write(json.dumps(result) + "\n")
        _real_stdout.flush()


if __name__ == "__main__":
    main()