"""Streamlit app: AZO dye lambda_max (nm) predictor.

Takes one or more user SMILES, featurizes them with 2D Mordred descriptors, and
predicts the pi->pi* absorption maximum (lambda_max, nm) with the pre-trained
wide-and-deep MLP (~27 nm MAE on an 18-molecule out-of-distribution azo-dye
holdout). Run with:

    streamlit run app.py
"""
import numpy as np
import streamlit as st

from azo_predictor import AZOPredictor

st.set_page_config(page_title="AZO lambda_max predictor", page_icon="🧪",
                   layout="centered")

# Cached so the model + Mordred featurizer load once per session.
@st.cache_resource
def get_predictor():
    return AZOPredictor()


st.title("AZO dye λmax predictor")
st.caption(
    "Wide-and-deep MLP on 2D Mordred descriptors, trained on 621 curated azo "
    "dyes. Reports the predicted π→π* absorption maximum (nm). Held-out "
    "performance on 18 new azo dyes: **MAE ≈ 27 nm, R² ≈ 0.69**."
)

st.markdown(
    "**Caveat:** the model was trained on a fixed azo-dye chemical space. "
    "Predictions are unreliable far outside it (e.g. non-azo scaffolds, very "
    "different substituent patterns). Treat outputs as estimates, not "
    "measurements."
)

# ---- input -------------------------------------------------------------- #
example = "CN(N=C1C)C(C)=C1/N=N/C2=C(OC)C=CC=C2OC1"
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

if st.button("Predict", type="primary", use_container_width=True):
    if not batch:
        st.warning("Enter at least one SMILES.")
    else:
        with st.spinner("Featurizing with Mordred and running the MLP…"):
            pred, invalid = get_predictor().predict(batch)
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
        st.dataframe(rows, use_container_width=True, hide_index=True)

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

        # Draw the molecules (lightweight RDKit rendering, no 3D needed)
        try:
            from rdkit import Chem
            from rdkit.Chem import Draw
            good = [(i, batch[i]) for i in valid_idx]
            if good:
                mols = [Chem.MolFromSmiles(s) for _, s in good]
                leg = [f"{pred[i]:.0f} nm" for i, _ in good]
                img = Draw.MolsToGridImage(mols, molsPerRow=4,
                                           subImgSize=(260, 200), legends=leg)
                st.subheader("Structures")
                st.image(img)
        except Exception as e:  # rendering is cosmetic; never block on it
            st.info(f"(Molecule rendering unavailable: {e})")
