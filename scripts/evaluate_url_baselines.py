"""
Evaluate Phase 2 URL baseline models.

This script evaluates Logistic Regression and Random Forest on the
frozen Phase 1 UCI URL train/validation/test splits.

Important:
    URLFeatureExtractor is fitted ONLY on the training split.

Validation and test data are transformed using the training-fitted
feature extractor to prevent preprocessing leakage.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from phishing_detection.data.url_loader import URLDatasetLoader
from phishing_detection.evaluation.metrics import (
    calculate_classification_metrics,
)
from phishing_detection.features.url_features import URLFeatureExtractor
from phishing_detection.models.traditional_ml.url_logistic_regression import (
    URLLogisticRegressionDetector,
)
from phishing_detection.models.traditional_ml.url_random_forest import (
    URLRandomForestDetector,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "url"
    / "uci_features"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "baselines"
    / "outputs"
)

RESULTS_FILE = (
    OUTPUT_DIR
    / "url_baseline_metrics.json"
)


def load_split(
    filename: str,
):
    """
    Load a URL split using the frozen Phase 1 schema.
    """

    loader = URLDatasetLoader(
        DATA_DIR / filename
    )

    return loader.get_features_and_labels()


def evaluate_model(
    model,
    X,
    y,
):
    """
    Generate predictions and calculate evaluation metrics.
    """

    predictions = model.predict(X)

    probabilities = model.predict_proba(X)

    phishing_probabilities = probabilities[:, 1]

    return calculate_classification_metrics(
        y_true=np.asarray(y),
        y_pred=np.asarray(predictions),
        y_probability=np.asarray(phishing_probabilities),
    )


def main() -> None:
    """
    Train URL baselines and evaluate them on validation and test data.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("PHASE 2 URL BASELINE EVALUATION")
    print("=" * 70)

    print("\nLoading training split...")

    X_train, y_train = load_split(
        "train.csv"
    )

    print(
        f"Training samples: {len(X_train):,}"
    )

    print("\nFitting URL feature extractor on training data...")

    feature_extractor = URLFeatureExtractor()

    X_train_transformed = (
        feature_extractor.fit_transform(
            X_train
        )
    )

    print(
        "Number of URL features: "
        f"{X_train_transformed.shape[1]}"
    )

    print("\nLoading validation split...")

    X_validation, y_validation = load_split(
        "validation.csv"
    )

    X_validation_transformed = (
        feature_extractor.transform(
            X_validation
        )
    )

    print(
        f"Validation samples: {len(X_validation):,}"
    )

    print("\nLoading test split...")

    X_test, y_test = load_split(
        "test.csv"
    )

    X_test_transformed = (
        feature_extractor.transform(
            X_test
        )
    )

    print(
        f"Test samples: {len(X_test):,}"
    )

    results = {
        "dataset": "UCI_Phishing_Websites",
        "feature_count": int(
            X_train_transformed.shape[1]
        ),
        "train_samples": int(
            len(X_train)
        ),
        "validation_samples": int(
            len(X_validation)
        ),
        "test_samples": int(
            len(X_test)
        ),
        "models": {},
    }

    # ---------------------------------------------------------
    # Logistic Regression
    # ---------------------------------------------------------

    print("\n" + "-" * 70)
    print("Training Logistic Regression")
    print("-" * 70)

    logistic_model = URLLogisticRegressionDetector()

    logistic_model.fit(
        X_train_transformed,
        y_train,
    )

    print("Evaluating validation set...")

    logistic_validation_metrics = evaluate_model(
        logistic_model,
        X_validation_transformed,
        y_validation,
    )

    print(
        "Validation F1: "
        f"{logistic_validation_metrics.f1:.4f}"
    )

    print("Evaluating test set...")

    logistic_test_metrics = evaluate_model(
        logistic_model,
        X_test_transformed,
        y_test,
    )

    print(
        "Test F1: "
        f"{logistic_test_metrics.f1:.4f}"
    )

    results["models"]["logistic_regression"] = {
        "validation": logistic_validation_metrics.to_dict(),
        "test": logistic_test_metrics.to_dict(),
    }

    # ---------------------------------------------------------
    # Random Forest
    # ---------------------------------------------------------

    print("\n" + "-" * 70)
    print("Training Random Forest")
    print("-" * 70)

    random_forest_model = URLRandomForestDetector(
        n_estimators=200
    )

    random_forest_model.fit(
        X_train_transformed,
        y_train,
    )

    print("Evaluating validation set...")

    rf_validation_metrics = evaluate_model(
        random_forest_model,
        X_validation_transformed,
        y_validation,
    )

    print(
        "Validation F1: "
        f"{rf_validation_metrics.f1:.4f}"
    )

    print("Evaluating test set...")

    rf_test_metrics = evaluate_model(
        random_forest_model,
        X_test_transformed,
        y_test,
    )

    print(
        "Test F1: "
        f"{rf_test_metrics.f1:.4f}"
    )

    results["models"]["random_forest"] = {
        "validation": rf_validation_metrics.to_dict(),
        "test": rf_test_metrics.to_dict(),
    }

    # ---------------------------------------------------------
    # Save results
    # ---------------------------------------------------------

    with RESULTS_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=2,
        )

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)

    print(
        f"\nResults saved to:\n{RESULTS_FILE}"
    )

    print("\nSummary:")
    print(
        f"Logistic Regression "
        f"Validation F1: "
        f"{logistic_validation_metrics.f1:.4f}"
    )

    print(
        f"Logistic Regression "
        f"Test F1: "
        f"{logistic_test_metrics.f1:.4f}"
    )

    print(
        f"Random Forest "
        f"Validation F1: "
        f"{rf_validation_metrics.f1:.4f}"
    )

    print(
        f"Random Forest "
        f"Test F1: "
        f"{rf_test_metrics.f1:.4f}"
    )


if __name__ == "__main__":
    main()