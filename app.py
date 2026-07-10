"""Streamlit app: AZO dye lambda_max (nm) predictor.

Takes one or more user SMILES, featurizes them with 2D Mordred descriptors, and
predicts the pi->pi* absorption maximum (lambda_max, nm) with the pre-trained
wide-and-deep MLP (~27 nm MAE on an 18-molecule out-of-distribution azo-dye
holdout). Run with:

    streamlit run app.py
"""
import os

# Force the numeric backends to a single thread BEFORE numpy/torch are
# imported. spawn() (like fork()) still calls fork() in this parent process;
# if MKL/OpenMP has a multi-threaded pool here, that fork is unsafe and a
# child that segfaults can take this server down with it. One thread => no
# pool to corrupt. Must run before `import numpy`.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np
import streamlit as st

st.set_page_config(page_title="AZO lambda_max predictor", page_icon="🧪",
                   layout="centered")

# Prediction and rendering run in a PERSISTENT subprocess (azo_worker.py)
# started once at import -- NOT a fresh subprocess per Predict click. The
# RDKit/Mordred C++ engine is never loaded in this process, so a segfault in
# the worker can't take the Streamlit server down; and because we fork the
# Streamlit parent exactly once (at import, before heavy numpy use here)
# instead of on every click, the repeated-fork crash that killed the server
# on the second run is gone. If the worker dies, we kill and restart it.
_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKER = os.path.join(_MODEL_DIR, "azo_worker.py")

import json
import queue
import subprocess
import sys
import threading

_WORKER_LOCK = threading.Lock()
_WORKER_PROC = None


def _kill_worker():
    global _WORKER_PROC
    if _WORKER_PROC is not None:
        try:
            _WORKER_PROC.kill()
        except Exception:
            pass
        try:
            _WORKER_PROC.wait(timeout=5)
        except Exception:
            pass
        _WORKER_PROC = None


def _ensure_worker():
    """Return a live worker process, starting one if none exists."""
    global _WORKER_PROC
    if _WORKER_PROC is not None and _WORKER_PROC.poll() is None:
        return _WORKER_PROC
    _WORKER_PROC = subprocess.Popen(
        [sys.executable, _WORKER],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        cwd=_MODEL_DIR, text=True, bufsize=1)
    return _WORKER_PROC


def _call_worker(payload, timeout):
    """Send one JSON request to the persistent worker, read one JSON response.
    Raises RuntimeError on timeout or worker death so callers fall back; on
    either, the worker is killed and the next call starts a fresh one."""
    req = json.dumps(payload) + "\n"
    with _WORKER_LOCK:
        proc = _ensure_worker()
        try:
            proc.stdin.write(req)
            proc.stdin.flush()
        except Exception:
            _kill_worker()
            proc = _ensure_worker()
            proc.stdin.write(req)
            proc.stdin.flush()
        # Read one response line with a timeout (proc.stdout.readline blocks,
        # so run it in a daemon thread and get from a queue).
        resp_q = queue.Queue()

        def _reader():
            try:
                resp_q.put(proc.stdout.readline())
            except Exception:
                resp_q.put("")

        threading.Thread(target=_reader, daemon=True).start()
        try:
            line = resp_q.get(timeout=timeout)
        except queue.Empty:
            _kill_worker()  # worker likely hung -> resync by restarting
            raise RuntimeError("timed out")
        if not line:
            _kill_worker()  # worker died (segfault) -> restart on next call
            raise RuntimeError("worker died")
        return json.loads(line)


def predict_safe(model_dir, smiles_list, timeout=240.0, per_item_timeout=60.0):
    """Predict lambda_max via the persistent worker. On a batch-level crash,
    fall back to one request per SMILES so a single bad input is reported
    invalid instead of costing the whole batch."""
    if isinstance(smiles_list, str):
        smiles_list = [smiles_list]
    try:
        r = _call_worker({"mode": "predict", "model_dir": model_dir,
                          "smiles": smiles_list}, timeout)
        return np.array(r["pred"], dtype=np.float64), list(r["invalid"])
    except RuntimeError:
        preds = np.full(len(smiles_list), np.nan, dtype=np.float64)
        invalid = []
        for i, smi in enumerate(smiles_list):
            try:
                r = _call_worker({"mode": "predict", "model_dir": model_dir,
                                  "smiles": [smi]}, per_item_timeout)
                preds[i] = r["pred"][0]
            except RuntimeError:
                invalid.append(i)
        return preds, invalid


def render_smiles_png(items, timeout=120.0):
    """Render molecules to a single PNG (base64) via the persistent worker.
    Returns '' on crash/timeout/parse-failure so the UI shows a fallback."""
    if not items:
        return ""
    try:
        r = _call_worker({"mode": "render", "items": items}, timeout)
        return r.get("png", "")
    except RuntimeError:
        return ""


st.title("AZO dye λmax predictor")
st.caption(
    "Wide-and-deep MLP on 2D Mordred descriptors, trained on 621 curated azo "
    "dyes. Reports the predicted π→π* absorption maximum (nm). Held-out "
    "performance on 18 new azo dyes: **MAE ≈ 27 nm, R² ≈ 0.69**."
)

# Build marker -- bump with every deploy so the live page shows which code is
# running.
_BUILD = "build-11 (persistent worker, per-mol PNG render, 2026-07-10)"
st.caption(f"`{_BUILD}`")

st.markdown(
    "**Caveat:** the model was trained on a fixed azo-dye chemical space. "
    "Predictions are unreliable far outside it (e.g. non-azo scaffolds, very "
    "different substituent patterns). Treat outputs as estimates, not "
    "measurements."
)

# ---- input -------------------------------------------------------------- #
example = "CN(N=C1C)C(C)=C1/N=N/C2=C(OC)C=CC=C2OC"
smiles_in = st.text_area(
    "Enter SMILES (one per line; blank lines are ignored):",
    value=example, height=120, help="Canonical or non-canonical SMILES are fine.",
)

batch = [s.strip() for s in smiles_in.splitlines() if s.strip()]

# Optional reference Lmax for comparison (same order, one per line)
with st.expander("Optional: paste known λmax (nm) values to compare (one per line)"):
    truth_in = st.text_area("Known λmax (nm), one per line:", value="", height=80,
                            key="truth")
truths = [t.strip() for t in truth_in.splitlines() if t.strip()] if truth_in else []

if st.button("Predict", type="primary", width="stretch"):
    if not batch:
        st.warning("Enter at least one SMILES.")
    else:
        try:
            with st.spinner("Featurizing with Mordred and running the MLP…"):
                # predict_safe runs in spawned child(ren) so a malformed SMILES
                # that segfaults the C++ engine (SIGSEGV) can't take this server
                # down -- a single bad input is reported as invalid instead.
                pred, invalid = predict_safe(_MODEL_DIR, batch)
        except Exception as e:  # never let a bad batch kill the whole app
            st.error(f"Could not process this batch: {e}")
            st.info("Check your SMILES for malformed entries and try again.")
            with st.expander("Error details"):
                st.code(f"{type(e).__name__}: {e}")
            st.stop()
        invalid_set = set(invalid)

        if invalid_set:
            bad = "\n".join(f"- line {i+1}: `{batch[i]}`"
                            for i in sorted(invalid_set))
            st.error(f"{len(invalid_set)} SMILES did not parse and were "
                     f"skipped:\n\n{bad}")

        st.subheader("Predictions")
        rows = []
        for i, (smi, val) in enumerate(zip(batch, pred)):
            if i in invalid_set:
                row = {"SMILES": smi, "Predicted λmax (nm)": "— (invalid)"}
            else:
                row = {"SMILES": smi, "Predicted λmax (nm)": f"{val:.1f}"}
                if i < len(truths):
                    try:
                        tv = float(truths[i])
                        row["Known λmax (nm)"] = f"{tv:.1f}"
                        row["Error (nm)"] = f"{abs(val - tv):.1f}"
                    except ValueError:
                        row["Known λmax (nm)"] = truths[i]
            rows.append(row)
        st.dataframe(rows, width="stretch", hide_index=True)

        # Summary when every (valid) prediction has a reference value
        valid_idx = [i for i in range(len(batch)) if i not in invalid_set]
        if truths and len(truths) == len(batch) and valid_idx:
            tv = np.array([float(truths[i]) for i in valid_idx])
            pv = pred[valid_idx]
            mae = float(np.mean(np.abs(pv - tv)))
            from sklearn.metrics import r2_score
            r2 = float(r2_score(tv, pv))
            c1, c2 = st.columns(2)
            c1.metric("MAE (nm)", f"{mae:.2f}")
            c2.metric("R²", f"{r2:.3f}")

        # Draw the molecules. Rendering runs in the persistent worker (see
        # render_smiles_png): each molecule is drawn in its own fresh drawer
        # and composited onto an explicit white grid, so there is no shared-
        # drawer black-panel bug and the background can't be black. The parent
        # never imports RDKit -- it just decodes the PNG and displays it.
        smi_leg = [(batch[i], f"{pred[i]:.0f} nm") for i in valid_idx]
        png_b64 = render_smiles_png(smi_leg)
        st.subheader("Structures")
        if png_b64:
            import base64
            st.image(base64.b64decode(png_b64), width="stretch")
        else:
            st.info("(Molecule rendering unavailable for this batch.)")
