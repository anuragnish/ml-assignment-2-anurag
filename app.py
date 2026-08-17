"""Interactive Streamlit dashboard for comparing classification models."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
TARGET = "target"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "kNN": "knn.joblib",
    "Gaussian Naive Bayes": "gaussian_naive_bayes.joblib",
    "Random Forest (Ensemble)": "random_forest.joblib",
    "Support Vector Machine (Additional)": "support_vector_machine.joblib",
}

MODEL_NOTES = {
    "Logistic Regression": "Linear probabilistic baseline; standardized inputs improve optimization.",
    "Decision Tree": "Interpretable nonlinear rules with depth and leaf-size controls.",
    "kNN": "Instance-based classifier using seven distance-weighted neighbours.",
    "Gaussian Naive Bayes": "Fast probabilistic model assuming conditional Gaussian features.",
    "Random Forest (Ensemble)": "Bagged decision trees that reduce variance and improve robustness.",
    "Support Vector Machine (Additional)": "RBF-kernel classifier added to reconcile the brief's six-model wording.",
}


@st.cache_resource
def load_artifacts():
    with (MODEL_DIR / "metrics.json").open(encoding="utf-8") as handle:
        metrics = json.load(handle)
    with (MODEL_DIR / "metadata.json").open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    models = {
        name: joblib.load(MODEL_DIR / filename)
        for name, filename in MODEL_FILES.items()
    }
    return models, metrics, metadata


def validate_upload(frame: pd.DataFrame, expected_features: list[str]) -> tuple[bool, str]:
    missing = [column for column in expected_features if column not in frame.columns]
    if missing:
        return False, f"Missing {len(missing)} required feature(s): {', '.join(missing[:6])}"
    duplicated = frame.columns[frame.columns.duplicated()].tolist()
    if duplicated:
        return False, f"Duplicate column names detected: {', '.join(duplicated)}"
    if frame[expected_features].isna().any().any():
        return False, "The uploaded feature columns contain missing values. Please clean them first."
    return True, "Valid test file"


st.set_page_config(
    page_title="Diagnostic ML Model Lab",
    page_icon="🧬",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {background: linear-gradient(180deg, #f7fbff 0%, #ffffff 35%);}
    .hero {padding: 1.2rem 1.4rem; border-radius: 18px; color: white;
           background: linear-gradient(120deg, #12355b, #007f86); margin-bottom: 1rem;}
    .hero h1 {margin: 0 0 .25rem 0; font-size: 2rem;}
    .hero p {margin: 0; opacity: .92;}
    [data-testid="stMetric"] {background: white; border: 1px solid #dce8ef;
                              padding: .65rem; border-radius: 12px;}
    </style>
    <div class="hero">
      <h1>Diagnostic ML Model Lab</h1>
      <p>Compare six classifiers on the UCI Breast Cancer Wisconsin Diagnostic test set.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

models, stored_metrics, metadata = load_artifacts()

with st.sidebar:
    st.header("Experiment controls")
    selected_model = st.selectbox("Choose a classification model", list(MODEL_FILES))
    st.caption(MODEL_NOTES[selected_model])
    uploaded_file = st.file_uploader("Upload test data (CSV)", type=["csv"])
    st.download_button(
        "Download compatible test_data.csv",
        data=(ROOT / "test_data.csv").read_bytes(),
        file_name="test_data.csv",
        mime="text/csv",
    )
    st.divider()
    st.caption(
        f"Training rows: {metadata['training_rows']} · Test rows: {metadata['test_rows']} · "
        f"Features: {len(metadata['feature_names'])}"
    )

if uploaded_file is None:
    data = pd.read_csv(ROOT / "test_data.csv")
    st.info("Showing results on the repository's test_data.csv. Upload another compatible CSV to re-evaluate.")
else:
    try:
        data = pd.read_csv(uploaded_file)
    except Exception as exc:
        st.error(f"The CSV could not be read: {exc}")
        st.stop()

valid, message = validate_upload(data, metadata["feature_names"])
if not valid:
    st.error(message)
    st.stop()

X = data[metadata["feature_names"]]
model = models[selected_model]
predictions = model.predict(X)

st.subheader(selected_model)
st.write(MODEL_NOTES[selected_model])

if TARGET in data.columns:
    y_true = data[TARGET].astype(int)
    y_score = model.predict_proba(X)[:, 1]

    from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score

    current_metrics = {
        "Accuracy": accuracy_score(y_true, predictions),
        "AUC": roc_auc_score(y_true, y_score),
        "Precision": precision_score(y_true, predictions, zero_division=0),
        "Recall": recall_score(y_true, predictions, zero_division=0),
        "F1": f1_score(y_true, predictions, zero_division=0),
        "MCC": matthews_corrcoef(y_true, predictions),
    }

    metric_columns = st.columns(6)
    for column, (name, value) in zip(metric_columns, current_metrics.items()):
        column.metric(name, f"{value:.3f}")

    left, right = st.columns([0.9, 1.1])
    with left:
        st.markdown("#### Confusion matrix")
        matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(5.2, 4.0))
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["Malignant", "Benign"],
            yticklabels=["Malignant", "Benign"],
            ax=ax,
        )
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("Actual class")
        fig.tight_layout()
        st.pyplot(fig)
    with right:
        st.markdown("#### Classification report")
        report = classification_report(
            y_true,
            predictions,
            labels=[0, 1],
            target_names=["malignant", "benign"],
            output_dict=True,
            zero_division=0,
        )
        st.dataframe(pd.DataFrame(report).T.round(3), use_container_width=True)

    st.markdown("#### All-model comparison on the fixed test split")
    comparison = pd.DataFrame(stored_metrics).T
    st.dataframe(
        comparison.style.format("{:.4f}").highlight_max(axis=0, color="#c8f1df"),
        use_container_width=True,
    )
else:
    st.warning("No target column was found, so metrics cannot be computed. Predictions are shown below.")
    output = data.copy()
    output["predicted_target"] = predictions
    output["predicted_label"] = output["predicted_target"].map({0: "malignant", 1: "benign"})
    st.dataframe(output.head(100), use_container_width=True)
    st.download_button(
        "Download predictions",
        output.to_csv(index=False).encode("utf-8"),
        file_name="predictions.csv",
        mime="text/csv",
    )

with st.expander("Dataset and class-label details"):
    st.write(
        "The dataset contains 569 cases and 30 real-valued features computed from digitized "
        "fine-needle aspirate images. The positive class is benign (1); the negative class is malignant (0)."
    )
    st.json(metadata)
