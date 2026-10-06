"""
Generate classification reports and confusion matrices for the
selected Phase 2 baseline models on the held-out test sets.

Selected models
---------------
Email:
    email_linear_svm
    C = 2.0

URL:
    url_random_forest
    n_estimators = 200
    max_features = "sqrt"

This script does NOT perform model selection or hyperparameter tuning.
It only generates diagnostic evaluation artifacts from the frozen
held-out test sets.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from phishing_detection.data.email_loader import EmailDatasetLoader
from phishing_detection.data.url_loader import URLDatasetLoader


# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EMAIL_TEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "email"
    / "test.csv"
)

URL_TEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "url"
    / "uci_features"
    / "test.csv"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "baselines"
    / "models"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "baselines"
    / "outputs"
)

REPORT_PATH = (
    OUTPUT_DIR
    / "classification_reports.json"
)


# ============================================================================
# MODEL ARTIFACT PATHS
# ============================================================================

EMAIL_MODEL_PATH = (
    MODEL_DIR
    / "email_linear_svm.pkl"
)

EMAIL_TFIDF_PATH = (
    MODEL_DIR
    / "email_linear_svm_tfidf_extractor.pkl"
)

URL_MODEL_PATH = (
    MODEL_DIR
    / "url_random_forest.pkl"
)

URL_FEATURE_PATH = (
    MODEL_DIR
    / "url_random_forest_feature_extractor.pkl"
)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def check_required_files() -> None:
    """Verify that all required datasets and model artifacts exist."""

    required_files = [
        EMAIL_TEST_PATH,
        URL_TEST_PATH,
        EMAIL_MODEL_PATH,
        EMAIL_TFIDF_PATH,
        URL_MODEL_PATH,
        URL_FEATURE_PATH,
    ]

    missing_files = [
        str(path)
        for path in required_files
        if not path.exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            "The following required files were not found:\n"
            + "\n".join(
                f"  - {path}"
                for path in missing_files
            )
        )


def save_confusion_matrix(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    model_name: str,
) -> str:
    """
    Generate and save a confusion matrix figure.

    Returns
    -------
    str
        Path to the saved figure.
    """

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=[
            "Legitimate",
            "Phishing",
        ],
    )

    figure, axis = plt.subplots(
        figsize=(7, 6)
    )

    display.plot(
        ax=axis,
        values_format="d",
    )

    axis.set_title(
        f"{model_name} — Held-Out Test Confusion Matrix"
    )

    axis.set_xlabel(
        "Predicted Label"
    )

    axis.set_ylabel(
        "True Label"
    )

    figure.tight_layout()

    output_path = (
        OUTPUT_DIR
        / f"{model_name}_confusion_matrix.png"
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    return str(output_path)


def build_classification_report(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
) -> dict:
    """Generate a JSON-serializable classification report."""

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=[
            "legitimate",
            "phishing",
        ],
        output_dict=True,
        zero_division=0,
    )

    return report


# ============================================================================
# EMAIL REPORT
# ============================================================================

def generate_email_report() -> dict:
    """Generate diagnostics for the selected email model."""

    print()
    print("=" * 70)
    print("EMAIL CLASSIFICATION REPORT")
    print("=" * 70)

    loader = EmailDatasetLoader(
        EMAIL_TEST_PATH
    )

    texts, y_test = (
        loader.get_texts_and_labels()
    )

    print(
        f"Test samples: {len(texts):,}"
    )

    print(
        "Loading email model..."
    )

    model = joblib.load(
        EMAIL_MODEL_PATH
    )

    print(
        "Loading TF-IDF extractor..."
    )

    tfidf_extractor = joblib.load(
        EMAIL_TFIDF_PATH
    )

    print(
        "Transforming test emails..."
    )

    X_test = tfidf_extractor.transform(
        texts
    )

    print(
        "Generating predictions..."
    )

    y_pred = model.predict(
        X_test
    )

    report = build_classification_report(
        y_test,
        y_pred,
    )

    matrix = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1],
    )

    figure_path = save_confusion_matrix(
        y_test,
        y_pred,
        "email_linear_svm",
    )

    print()
    print("Email classification report:")

    print(
        classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=[
                "legitimate",
                "phishing",
            ],
            zero_division=0,
        )
    )

    print(
        "Email confusion matrix:"
    )

    print(matrix)

    print()
    print(
        f"Confusion matrix saved to:"
    )

    print(figure_path)

    return {
        "model_name": "email_linear_svm",
        "modality": "email",
        "parameters": {
            "C": 2.0,
        },
        "test_samples": len(y_test),
        "classification_report": report,
        "confusion_matrix": matrix.astype(
            int
        ).tolist(),
        "confusion_matrix_labels": [
            ["TN", "FP"],
            ["FN", "TP"],
        ],
        "confusion_matrix_figure": figure_path,
    }


# ============================================================================
# URL REPORT
# ============================================================================

def generate_url_report() -> dict:
    """Generate diagnostics for the selected URL model."""

    print()
    print("=" * 70)
    print("URL CLASSIFICATION REPORT")
    print("=" * 70)

    loader = URLDatasetLoader(
        URL_TEST_PATH
    )

    X_test_raw, y_test = (
        loader.get_features_and_labels()
    )

    print(
        f"Test samples: "
        f"{len(X_test_raw):,}"
    )

    print(
        "Loading URL model..."
    )

    model = joblib.load(
        URL_MODEL_PATH
    )

    print(
        "Loading URL feature extractor..."
    )

    feature_extractor = joblib.load(
        URL_FEATURE_PATH
    )

    print(
        "Transforming test URLs..."
    )

    X_test = feature_extractor.transform(
        X_test_raw
    )

    print(
        "Generating predictions..."
    )

    y_pred = model.predict(
        X_test
    )

    report = build_classification_report(
        y_test,
        y_pred,
    )

    matrix = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1],
    )

    figure_path = save_confusion_matrix(
        y_test,
        y_pred,
        "url_random_forest",
    )

    print()
    print("URL classification report:")

    print(
        classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=[
                "legitimate",
                "phishing",
            ],
            zero_division=0,
        )
    )

    print(
        "URL confusion matrix:"
    )

    print(matrix)

    print()
    print(
        "Confusion matrix saved to:"
    )

    print(figure_path)

    return {
        "model_name": "url_random_forest",
        "modality": "url",
        "parameters": {
            "n_estimators": 200,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
        },
        "test_samples": len(y_test),
        "classification_report": report,
        "confusion_matrix": matrix.astype(
            int
        ).tolist(),
        "confusion_matrix_labels": [
            ["TN", "FP"],
            ["FN", "TP"],
        ],
        "confusion_matrix_figure": figure_path,
    }


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    """Generate all selected-model test diagnostics."""

    print()
    print("=" * 70)
    print(
        "PHASE 2 — TEST CLASSIFICATION REPORTS"
    )
    print("=" * 70)

    print()
    print(
        "Checking required files..."
    )

    check_required_files()

    print(
        "All required files found."
    )

    email_report = (
        generate_email_report()
    )

    url_report = (
        generate_url_report()
    )

    results = {
        "experiment": (
            "phase2_test_classification_reports"
        ),
        "test_set_used": True,
        "label_convention": {
            "0": "legitimate",
            "1": "phishing",
        },
        "email": email_report,
        "url": url_report,
    }

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=2,
        )

    print()
    print("=" * 70)
    print(
        "CLASSIFICATION REPORT GENERATION COMPLETE"
    )
    print("=" * 70)

    print()
    print(
        "JSON report saved to:"
    )

    print(REPORT_PATH)

    print()
    print(
        "Generated figures:"
    )

    print(
        OUTPUT_DIR
        / "email_linear_svm_confusion_matrix.png"
    )

    print(
        OUTPUT_DIR
        / "url_random_forest_confusion_matrix.png"
    )


if __name__ == "__main__":
    main()