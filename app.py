"""Small Streamlit research demo for the saved BRD pipeline."""
from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ARTIFACT_PATH = Path(__file__).parent / "outputs" / "models" / "best_brd_model.joblib"
SUMMARY_PATH = Path(__file__).parent / "outputs" / "results" / "run_summary.json"

st.set_page_config(page_title="Early BRD Risk Detection", page_icon="🐄", layout="centered")
st.title("Early BRD Risk Detection")
st.subheader("Machine Learning Research Prototype")
st.caption("Model: " + ARTIFACT_PATH.name)
st.warning("This is an academic machine-learning research prototype and is not a veterinary diagnostic tool.")

with st.sidebar:
    st.markdown("### Project snapshot")
    summary = json.loads(SUMMARY_PATH.read_text()) if SUMMARY_PATH.exists() else {}
    st.metric("Dataset records", f"{summary.get('dataset', {}).get('rows', 'n/a'):,}" if summary.get("dataset", {}).get("rows") else "n/a")
    st.metric("Unique calves", summary.get("dataset", {}).get("unique_calves", "n/a"))
    st.caption("The model was tested on calves kept completely separate from training.")

if not ARTIFACT_PATH.exists():
    st.error("Train the project with `python main.py` before opening the demo.")
    st.stop()

artifact = joblib.load(ARTIFACT_PATH)
features = artifact["feature_columns"]
raw = {}
for feature in features:
    if feature in artifact["numeric_features"]:
        raw[feature] = st.number_input(feature, value=0.0, format="%.4f")
    else:
        raw[feature] = st.text_input(feature, value="")

if st.button("Predict risk", type="primary"):
    sample = pd.DataFrame([raw], columns=features)
    probability = float(artifact["pipeline"].predict_proba(sample)[0, 1])
    threshold = float(artifact.get("threshold", 0.5))  # tuned on training calves (F2-optimal)
    prediction = int(probability >= threshold)
    if probability is not None:
        st.divider()
        st.markdown("### Prediction overview")
        result_col, score_col = st.columns([1.4, 1])
        with result_col:
            if prediction:
                st.error("Elevated predicted BRD risk")
            else:
                st.success("Low predicted BRD risk")
            st.progress(float(probability), text=f"Pre-BRD probability: {probability:.1%}")
        with score_col:
            st.metric("Risk score", f"{probability:.1%}", delta="Above model threshold" if prediction else "Below model threshold")

        # A simple probability gauge is easier to explain during a live presentation.
        figure, axis = plt.subplots(figsize=(8, 0.9))
        axis.barh([0], [0.30], color="#4d8b68", height=0.45)
        axis.barh([0], [0.40], left=0.30, color="#d6a84f", height=0.45)
        axis.barh([0], [0.30], left=0.70, color="#c95c4b", height=0.45)
        axis.axvline(probability, color="#18252b", linewidth=3)
        axis.axvline(threshold, color="#18252b", linewidth=1, linestyle="--")
        axis.text(probability, 0.36, f"{probability:.1%}", ha="center", va="bottom", fontweight="bold")
        axis.set_xlim(0, 1); axis.set_yticks([]); axis.set_xticks([0, .3, .7, 1]); axis.set_xticklabels(["0%", "30%", "70%", "100%"])
        axis.set_title("Model probability guide", loc="left", fontsize=11, pad=8)
        for spine in axis.spines.values(): spine.set_visible(False)
        figure.tight_layout()
        st.pyplot(figure, clear_figure=True, use_container_width=True)

        entered = pd.Series(raw).replace("", pd.NA).notna()
        summary = pd.DataFrame({"Input group": ["Numeric measurements", "Categorical observations"], "Entered fields": [int(entered[artifact["numeric_features"]].sum()), int(entered[artifact["categorical_features"]].sum())]})
        st.caption("Input completeness")
        st.bar_chart(summary.set_index("Input group"), horizontal=True)
    else:
        st.info("This model does not provide a probability score.")
