"""
Evaluate selected Phase 2 baseline models on held-out test sets.

Selected models:

Email:
    email_linear_svm
    C = 2.0

URL:
    url_random_forest
    n_estimators = 200
    min_samples_leaf = 1
    max_features = "sqrt"

Important:
    - Test data is used only for final evaluation.
    - No hyperparameter tuning is performed here.
    - Feature extractors are loaded from artifacts fitted on training data.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib

from phishing_detection.data.email_loader import EmailDatasetLoader
from phishing_detection.data.url_loader import URLDatasetLoader
from phishing_detection.evaluation.metrics import (
    calculate_classification_metrics,
)


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

OUTPUT_PATH = OUTPUT_DIR / "test_metrics.json"


# ============================================================================
# MODEL ARTIFACTS
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
# VALIDATION
# ============================================================================

def check_required_files() -> None:
    """Check that all required datasets and model artifacts exist."""

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


# ============================================================================
# EMAIL EVALUATION
# ============================================================================

def evaluate_email() -> dict:
    """Evaluate the selected email model on the held-out test set."""

    print()
    print("=" * 70)
    print("EMAIL TEST EVALUATION")
    print("=" * 70)

    # ------------------------------------------------------------------------
    # Load test data
    # ------------------------------------------------------------------------

    loader = EmailDatasetLoader(
        EMAIL_TEST_PATH
    )

    texts, y_test = loader.get_texts_and_labels()

    print(
        f"Test samples: {len(texts):,}"
    )

    print(
        "Model: email_linear_svm"
    )

    print(
        "Parameters: C=2.0"
    )

    # ------------------------------------------------------------------------
    # Load model and TF-IDF extractor
    # ------------------------------------------------------------------------

    print()
    print("Loading email model...")

    model = joblib.load(
        EMAIL_MODEL_PATH
    )

    print(
        "Loading TF-IDF extractor..."
    )

    tfidf_extractor = joblib.load(
        EMAIL_TFIDF_PATH
    )

    # ------------------------------------------------------------------------
    # Transform test data
    # ------------------------------------------------------------------------

    print(
        "Transforming test emails..."
    )

    start_transform = time.perf_counter()

    X_test = tfidf_extractor.transform(
        texts
    )

    transform_time = (
        time.perf_counter()
        - start_transform
    )

    print(
        f"Test feature matrix: "
        f"{X_test.shape[0]:,} samples × "
        f"{X_test.shape[1]:,} features"
    )

    print(
        f"Transformation time: "
        f"{transform_time:.4f} seconds"
    )

    # ------------------------------------------------------------------------
    # Predictions
    # ------------------------------------------------------------------------

    print(
        "Generating predictions..."
    )

    start_inference = time.perf_counter()

    y_pred = model.predict(
        X_test
    )

    inference_time = (
        time.perf_counter()
        - start_inference
    )

    print(
        f"Inference time: "
        f"{inference_time:.4f} seconds"
    )

    # ------------------------------------------------------------------------
    # Probability estimates
    # ------------------------------------------------------------------------

    print(
        "Generating probability estimates..."
    )

    start_probability = time.perf_counter()

    y_probability = model.predict_proba(
        X_test
    )[:, 1]

    probability_time = (
        time.perf_counter()
        - start_probability
    )

    print(
        f"Probability estimation time: "
        f"{probability_time:.4f} seconds"
    )

    # ------------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------------

    metrics = calculate_classification_metrics(
        y_true=y_test,
        y_pred=y_pred,
        y_probability=y_probability,
    )

    # ClassificationMetrics already provides to_dict().
    metrics_dict = metrics.to_dict()

    # ------------------------------------------------------------------------
    # Add experiment metadata
    # ------------------------------------------------------------------------

    metrics_dict.update(
        {
            "model_name": "email_linear_svm",
            "modality": "email",
            "parameters": {
                "C": 2.0,
            },
            "test_samples": len(y_test),
            "num_features": int(
                X_test.shape[1]
            ),
            "feature_extractor": "TF-IDF",
            "transformation_time_seconds": float(
                transform_time
            ),
            "inference_time_seconds": float(
                inference_time
            ),
            "probability_estimation_time_seconds": float(
                probability_time
            ),
            "total_prediction_time_seconds": float(
                inference_time
                + probability_time
            ),
            "inference_time_per_sample_ms": float(
                inference_time
                / len(y_test)
                * 1000
            ),
        }
    )

    # ------------------------------------------------------------------------
    # Display metrics
    # ------------------------------------------------------------------------

    print()
    print("Email test metrics:")

    print(
        f"  Accuracy : "
        f"{metrics_dict['accuracy']:.4f}"
    )

    print(
        f"  Precision: "
        f"{metrics_dict['precision']:.4f}"
    )

    print(
        f"  Recall   : "
        f"{metrics_dict['recall']:.4f}"
    )

    print(
        f"  F1       : "
        f"{metrics_dict['f1']:.4f}"
    )

    print(
        f"  ROC-AUC  : "
        f"{metrics_dict['roc_auc']:.4f}"
    )

    print(
        f"  PR-AUC   : "
        f"{metrics_dict['pr_auc']:.4f}"
    )

    print(
        "  Confusion Matrix:"
    )

    for row in metrics_dict[
        "confusion_matrix"
    ]:
        print(
            f"    {row}"
        )

    return metrics_dict


# ============================================================================
# URL EVALUATION
# ============================================================================

def evaluate_url() -> dict:
    """Evaluate the selected URL model on the held-out test set."""

    print()
    print("=" * 70)
    print("URL TEST EVALUATION")
    print("=" * 70)

    # ------------------------------------------------------------------------
    # Load test data
    # ------------------------------------------------------------------------

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
        "Model: url_random_forest"
    )

    print(
        "Parameters: "
        "n_estimators=200, "
        "min_samples_leaf=1, "
        "max_features=sqrt"
    )

    # ------------------------------------------------------------------------
    # Load model and feature extractor
    # ------------------------------------------------------------------------

    print()
    print("Loading URL model...")

    model = joblib.load(
        URL_MODEL_PATH
    )

    print(
        "Loading URL feature extractor..."
    )

    feature_extractor = joblib.load(
        URL_FEATURE_PATH
    )

    # ------------------------------------------------------------------------
    # Transform test data
    # ------------------------------------------------------------------------

    print(
        "Transforming test URLs..."
    )

    start_transform = time.perf_counter()

    X_test = feature_extractor.transform(
        X_test_raw
    )

    transform_time = (
        time.perf_counter()
        - start_transform
    )

    print(
        f"Test feature matrix: "
        f"{X_test.shape[0]:,} samples × "
        f"{X_test.shape[1]:,} features"
    )

    print(
        f"Transformation time: "
        f"{transform_time:.4f} seconds"
    )

    # ------------------------------------------------------------------------
    # Predictions
    # ------------------------------------------------------------------------

    print(
        "Generating predictions..."
    )

    start_inference = time.perf_counter()

    y_pred = model.predict(
        X_test
    )

    inference_time = (
        time.perf_counter()
        - start_inference
    )

    print(
        f"Inference time: "
        f"{inference_time:.4f} seconds"
    )

    # ------------------------------------------------------------------------
    # Probability estimates
    # ------------------------------------------------------------------------

    print(
        "Generating probability estimates..."
    )

    start_probability = time.perf_counter()

    y_probability = model.predict_proba(
        X_test
    )[:, 1]

    probability_time = (
        time.perf_counter()
        - start_probability
    )

    print(
        f"Probability estimation time: "
        f"{probability_time:.4f} seconds"
    )

    # ------------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------------

    metrics = calculate_classification_metrics(
        y_true=y_test,
        y_pred=y_pred,
        y_probability=y_probability,
    )

    metrics_dict = metrics.to_dict()

    # ------------------------------------------------------------------------
    # Add experiment metadata
    # ------------------------------------------------------------------------

    metrics_dict.update(
        {
            "model_name": "url_random_forest",
            "modality": "url",
            "parameters": {
                "n_estimators": 200,
                "min_samples_leaf": 1,
                "max_features": "sqrt",
            },
            "test_samples": len(y_test),
            "num_features": int(
                X_test.shape[1]
            ),
            "feature_extractor": (
                "URLFeatureExtractor"
            ),
            "transformation_time_seconds": float(
                transform_time
            ),
            "inference_time_seconds": float(
                inference_time
            ),
            "probability_estimation_time_seconds": float(
                probability_time
            ),
            "total_prediction_time_seconds": float(
                inference_time
                + probability_time
            ),
            "inference_time_per_sample_ms": float(
                inference_time
                / len(y_test)
                * 1000
            ),
        }
    )

    # ------------------------------------------------------------------------
    # Display metrics
    # ------------------------------------------------------------------------

    print()
    print("URL test metrics:")

    print(
        f"  Accuracy : "
        f"{metrics_dict['accuracy']:.4f}"
    )

    print(
        f"  Precision: "
        f"{metrics_dict['precision']:.4f}"
    )

    print(
        f"  Recall   : "
        f"{metrics_dict['recall']:.4f}"
    )

    print(
        f"  F1       : "
        f"{metrics_dict['f1']:.4f}"
    )

    print(
        f"  ROC-AUC  : "
        f"{metrics_dict['roc_auc']:.4f}"
    )

    print(
        f"  PR-AUC   : "
        f"{metrics_dict['pr_auc']:.4f}"
    )

    print(
        "  Confusion Matrix:"
    )

    for row in metrics_dict[
        "confusion_matrix"
    ]:
        print(
            f"    {row}"
        )

    return metrics_dict


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    """Run final held-out test evaluation."""

    print()
    print("=" * 70)
    print(
        "PHASE 2 — SELECTED BASELINE TEST EVALUATION"
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

    # ------------------------------------------------------------------------
    # Evaluate selected models
    # ------------------------------------------------------------------------

    email_metrics = evaluate_email()

    url_metrics = evaluate_url()

    # ------------------------------------------------------------------------
    # Build final results
    # ------------------------------------------------------------------------

    results = {
        "experiment": (
            "phase2_selected_baseline_test_evaluation"
        ),
        "test_set_used": True,
        "selection_metric": "f1",
        "random_state": 42,
        "email": email_metrics,
        "url": url_metrics,
    }

    # ------------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=2,
        )

    # ------------------------------------------------------------------------
    # Completion message
    # ------------------------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "TEST EVALUATION COMPLETE"
    )
    print("=" * 70)

    print()
    print(
        "Results saved to:"
    )

    print(
        OUTPUT_PATH
    )

    print()
    print(
        "Selected models:"
    )

    print(
        "  Email: email_linear_svm (C=2.0)"
    )

    print(
        "  URL  : url_random_forest "
        "(n_estimators=200, max_features=sqrt)"
    )


if __name__ == "__main__":
    main()