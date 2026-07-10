"""Subprocess worker for the AZO Streamlit app.

Reads a JSON request from stdin, does the work (prediction or rendering), and
writes a JSON response to stdout. It runs in a *fresh* interpreter launched by
``subprocess.run`` -- not ``multiprocessing`` -- so there is no persistent
resource tracker, no inherited multiprocessing state, and no shared memory
between this process and the Streamlit server. If the RDKit/Mordred C++ engine
segfaults on a bad input, only this short-lived process dies; the parent sees a
non-zero exit code and reports a graceful error. The parent never imports
torch/RDKit/Mordred.

Request shapes:
  {"mode": "predict", "model_dir": "...", "smiles": ["...", ...]}
  {"mode": "render",  "items": [["smiles", "legend"], ...]}

Response shapes:
  {"pred": [...], "invalid": [...]}     # predict
  {"svg": "<svg ...>"}                  # render
"""
import json
import os
import sys

# Single-thread the numeric backends before importing anything heavy.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_HERE = os.path.dirname(os.path.abspath(__file__))


def _predict(req):
    import torch
    torch.set_num_threads(1)
    from azo_predictor import AZOPredictor
    # n_jobs=1 keeps Mordred single-process: no loky worker pool spawning from
    # inside this already-separate process.
    predictor = AZOPredictor(model_dir=req.get("model_dir") or _HERE, n_jobs=1)
    pred, invalid = predictor.predict(req["smiles"])
    return {"pred": pred.tolist(), "invalid": [int(i) for i in invalid]}


def _render(req):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Chem.Draw import rdMolDraw2D

    mols, leg = [], []
    for smi, legend in req["items"]:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        # Ensure 2D coords exist; a degenerate layout can render as a black
        # blob on some RDKit builds, so generate them explicitly.
        try:
            AllChem.Compute2DCoords(m)
        except Exception:
            pass
        mols.append(m)
        leg.append(legend)
    if not mols:
        return {"svg": ""}
    ncol = min(4, len(mols))
    sub = (260, 200)
    nrow = (len(mols) + ncol - 1) // ncol
    drawer = rdMolDraw2D.MolDraw2DSVG(ncol * sub[0], nrow * sub[1], sub[0], sub[1])
    opts = drawer.drawOptions()
    opts.clearBackground = True
    opts.backgroundColour = (1, 1, 1, 1)  # opaque white, any theme
    drawer.SetDrawOptions(opts)
    drawer.DrawMolecules(mols, legends=leg)
    drawer.FinishDrawing()
    return {"svg": drawer.GetDrawingText()}


def main():
    req = json.loads(sys.stdin.read())
    mode = req["mode"]
    if mode == "predict":
        result = _predict(req)
    elif mode == "render":
        result = _render(req)
    else:
        raise ValueError(f"unknown mode: {mode}")
    json.dump(result, sys.stdout)


if __name__ == "__main__":
    main()