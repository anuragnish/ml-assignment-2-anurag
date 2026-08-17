"""Train and persist all classifiers for the ML Assignment 2 Streamlit app."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
RANDOM_STATE = 42


def build_models() -> dict[str, object]:
    """Return the five mandatory models plus SVM as a sixth model."""
    return {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
                ),
            ]
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "kNN": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", KNeighborsClassifier(n_neighbors=7, weights="distance")),
            ]
        ),
        "Gaussian Naive Bayes": GaussianNB(),
        "Random Forest (Ensemble)": RandomForestClassifier(
            n_estimators=400,
            max_features="sqrt",
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Support Vector Machine (Additional)": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    SVC(kernel="rbf", C=2.0, gamma="scale", probability=True,
                        class_weight="balanced", random_state=RANDOM_STATE),
                ),
            ]
        ),
    }


def evaluate(y_true: pd.Series, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    """Calculate every metric required by the assignment."""
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_score),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    dataset = load_breast_cancer(as_frame=True)
    X = dataset.data.copy()
    y = dataset.target.copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    feature_metadata = {
        "dataset_name": "Breast Cancer Wisconsin (Diagnostic)",
        "source": "UCI Machine Learning Repository / scikit-learn bundled copy",
        "random_state": RANDOM_STATE,
        "test_size": 0.20,
        "positive_class": "benign (1)",
        "negative_class": "malignant (0)",
        "feature_names": list(X.columns),
        "target_names": list(dataset.target_names),
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
    }

    metrics: dict[str, dict[str, float]] = {}
    for model_name, model in build_models().items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]
        metrics[model_name] = evaluate(y_test, predictions, probabilities)
        safe_name = (
            model_name.lower()
            .replace(" (ensemble)", "")
            .replace(" (additional)", "")
            .replace(" ", "_")
        )
        joblib.dump(model, MODEL_DIR / f"{safe_name}.joblib")

    test_data = X_test.copy()
    test_data["target"] = y_test.to_numpy()
    test_data.to_csv(ROOT / "test_data.csv", index=False)

    with (MODEL_DIR / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    with (MODEL_DIR / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(feature_metadata, handle, indent=2)

    print(pd.DataFrame(metrics).T.round(4).to_string())


if __name__ == "__main__":
    main()
