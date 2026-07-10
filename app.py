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

from azo_predictor import predict_safe, render_smiles_svg

st.set_page_config(page_title="AZO lambda_max predictor", page_icon="🧪",
                   layout="centered")

# The model + Mordred featurizer load in the spawned prediction child, not
# here -- the parent (this Streamlit process) never touches the C++ engine, so
# a segfault in a child can't take the server down. We only need the path to
# the artifacts.
_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))


st.title("AZO dye λmax predictor")
st.caption(
    "Wide-and-deep MLP on 2D Mordred descriptors, trained on 621 curated azo "
    "dyes. Reports the predicted π→π* absorption maximum (nm). Held-out "
    "performance on 18 new azo dyes: **MAE ≈ 27 nm, R² ≈ 0.69**."
)

# Build marker -- bump with every deploy so the live page shows which code is
# running.
_BUILD = "build-9 (single-thread env forced before numpy import, 2026-07-10)"
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

        # Draw the molecules. Rendering also runs in a spawned child (see
        # render_smiles_svg): RDKit's 2D drawer is C++ and can segfault on
        # certain inputs/builds, and that must not take this server down. The
        # parent never imports or calls RDKit -- it just displays the SVG.
        smi_leg = [(batch[i], f"{pred[i]:.0f} nm") for i in valid_idx]
        svg = render_smiles_svg(smi_leg)
        st.subheader("Structures")
        if svg:
            # st.image renders an SVG string as a data-URI <img> (it matches
            # the leading <?xml?>/<svg> and base64-encodes it). width="stretch"
            # is the recommended sizing for SVGs.
            st.image(svg, width="stretch")
        else:
            st.info("(Molecule rendering unavailable for this batch.)")
