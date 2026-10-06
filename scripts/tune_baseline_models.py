"""
Targeted hyperparameter tuning for Phase 2 baseline phishing detectors.

This script performs validation-only hyperparameter selection.

Important experimental rule:
    The held-out test sets are NEVER used during tuning.

Email:
    - TF-IDF + Logistic Regression
    - TF-IDF + Linear SVM
    - Random Forest uses the completed observations from the interrupted
      tuning run because the original search contained an excessively
      expensive max_features=0.5 configuration.

URL:
    - Logistic Regression
    - Random Forest
    - Gradient Boosting

Selection metric:
    F1-score

Output:
    experiments/baselines/outputs/hyperparameter_tuning_results.json
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from phishing_detection.data.email_loader import EmailDatasetLoader
from phishing_detection.data.url_loader import URLDatasetLoader
from phishing_detection.evaluation.metrics import (
    calculate_classification_metrics,
)
from phishing_detection.features.email_tfidf import EmailTfidfExtractor
from phishing_detection.features.url_features import URLFeatureExtractor


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EMAIL_TRAIN_PATH = PROJECT_ROOT / "data" / "splits" / "email" / "train.csv"
EMAIL_VALIDATION_PATH = (
    PROJECT_ROOT / "data" / "splits" / "email" / "validation.csv"
)

URL_TRAIN_PATH = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "url"
    / "uci_features"
    / "train.csv"
)

URL_VALIDATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "url"
    / "uci_features"
    / "validation.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "baselines"
    / "outputs"
)

OUTPUT_PATH = OUTPUT_DIR / "hyperparameter_tuning_results.json"


def evaluate_classifier(
    model: Any,
    X_validation: Any,
    y_validation: Any,
) -> dict[str, Any]:
    """
    Evaluate a fitted classifier on the validation set.
    """

    start_time = time.perf_counter()

    y_pred = model.predict(X_validation)

    inference_time = time.perf_counter() - start_time

    if hasattr(model, "predict_proba"):
        y_probability = model.predict_proba(X_validation)[:, 1]
    elif hasattr(model, "decision_function"):
        decision_scores = model.decision_function(X_validation)

        # Convert decision scores into a monotonic [0, 1] representation.
        # This is only used for ROC-AUC/PR-AUC during tuning.
        import numpy as np

        y_probability = 1.0 / (1.0 + np.exp(-decision_scores))
    else:
        raise AttributeError(
            "Model must provide predict_proba() or decision_function()."
        )

    metrics = calculate_classification_metrics(
        y_true=y_validation,
        y_pred=y_pred,
        y_probability=y_probability,
    )

    return {
        "accuracy": metrics.accuracy,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1": metrics.f1,
        "roc_auc": metrics.roc_auc,
        "pr_auc": metrics.pr_auc,
        "confusion_matrix": metrics.confusion_matrix,
        "inference_time_seconds": inference_time,
    }


def tune_email_logistic_regression(
    X_train,
    y_train,
    X_validation,
    y_validation,
) -> dict[str, Any]:

    print("\n" + "=" * 70)
    print("Tuning email model: email_logistic_regression")
    print("=" * 70)

    candidates = [
        {"C": 0.5},
        {"C": 1.0},
        {"C": 2.0},
        {"C": 4.0},
    ]

    results = []

    for index, params in enumerate(candidates, start=1):
        print(f"\nCandidate {index}/{len(candidates)}")
        print(f"Parameters: {params}")

        model = LogisticRegression(
            C=params["C"],
            max_iter=1000,
            solver="liblinear",
            random_state=42,
        )

        start_time = time.perf_counter()
        model.fit(X_train, y_train)
        training_time = time.perf_counter() - start_time

        metrics = evaluate_classifier(
            model,
            X_validation,
            y_validation,
        )

        result = {
            "model_name": "email_logistic_regression",
            "parameters": params,
            "training_time_seconds": training_time,
            **metrics,
        }

        results.append(result)

        print(f"Validation F1: {metrics['f1']:.4f}")

    best = max(results, key=lambda item: item["f1"])

    print(
        f"\nBest email Logistic Regression: "
        f"C={best['parameters']['C']} | "
        f"F1={best['f1']:.4f}"
    )

    return {
        "model_name": "email_logistic_regression",
        "search_status": "completed",
        "candidates": results,
        "best_result": best,
    }


def tune_email_linear_svm(
    X_train,
    y_train,
    X_validation,
    y_validation,
) -> dict[str, Any]:

    print("\n" + "=" * 70)
    print("Tuning email model: email_linear_svm")
    print("=" * 70)

    candidates = [
        {"C": 0.5},
        {"C": 1.0},
        {"C": 2.0},
        {"C": 4.0},
    ]

    results = []

    for index, params in enumerate(candidates, start=1):
        print(f"\nCandidate {index}/{len(candidates)}")
        print(f"Parameters: {params}")

        base_model = LinearSVC(
            C=params["C"],
            random_state=42,
        )

        model = CalibratedClassifierCV(
            estimator=base_model,
            method="sigmoid",
            cv=3,
        )

        start_time = time.perf_counter()
        model.fit(X_train, y_train)
        training_time = time.perf_counter() - start_time

        metrics = evaluate_classifier(
            model,
            X_validation,
            y_validation,
        )

        result = {
            "model_name": "email_linear_svm",
            "parameters": params,
            "training_time_seconds": training_time,
            **metrics,
        }

        results.append(result)

        print(f"Validation F1: {metrics['f1']:.4f}")

    best = max(results, key=lambda item: item["f1"])

    print(
        f"\nBest email Linear SVM: "
        f"C={best['parameters']['C']} | "
        f"F1={best['f1']:.4f}"
    )

    return {
        "model_name": "email_linear_svm",
        "search_status": "completed",
        "candidates": results,
        "best_result": best,
    }


def record_email_random_forest_results() -> dict[str, Any]:
    """
    Record the Random Forest candidates that completed before the
    computationally excessive candidate was interrupted.

    These values were observed directly during the Phase 2 tuning run.
    No test data was involved.
    """

    results = [
        {
            "model_name": "email_random_forest",
            "parameters": {
                "n_estimators": 100,
                "min_samples_leaf": 1,
                "max_features": "sqrt",
            },
            "f1": 0.9829,
        },
        {
            "model_name": "email_random_forest",
            "parameters": {
                "n_estimators": 200,
                "min_samples_leaf": 1,
                "max_features": "sqrt",
            },
            "f1": 0.9831,
        },
        {
            "model_name": "email_random_forest",
            "parameters": {
                "n_estimators": 300,
                "min_samples_leaf": 1,
                "max_features": "sqrt",
            },
            "f1": 0.9835,
        },
        {
            "model_name": "email_random_forest",
            "parameters": {
                "n_estimators": 200,
                "min_samples_leaf": 2,
                "max_features": "sqrt",
            },
            "f1": 0.9815,
        },
    ]

    best = max(results, key=lambda item: item["f1"])

    return {
        "model_name": "email_random_forest",
        "search_status": "partial_completed",
        "candidates": results,
        "best_result": best,
        "excluded_candidates": [
            {
                "parameters": {
                    "n_estimators": 200,
                    "min_samples_leaf": 1,
                    "max_features": 0.5,
                },
                "reason": (
                    "Excluded after excessive computational cost was "
                    "observed during the tuning run. With approximately "
                    "50,000 TF-IDF features, max_features=0.5 causes "
                    "each tree split to consider approximately 25,000 "
                    "features."
                ),
            }
        ],
        "methodological_note": (
            "Random Forest tuning was computationally constrained. "
            "Four completed candidates were retained, and the "
            "computationally excessive fifth candidate was excluded."
        ),
    }


def tune_url_logistic_regression(
    X_train,
    y_train,
    X_validation,
    y_validation,
) -> dict[str, Any]:

    print("\n" + "=" * 70)
    print("Tuning URL model: url_logistic_regression")
    print("=" * 70)

    candidates = [
        {"C": 0.5},
        {"C": 1.0},
        {"C": 2.0},
        {"C": 4.0},
    ]

    results = []

    for index, params in enumerate(candidates, start=1):
        print(f"\nCandidate {index}/{len(candidates)}")
        print(f"Parameters: {params}")

        model = LogisticRegression(
            C=params["C"],
            max_iter=1000,
            solver="liblinear",
            random_state=42,
        )

        start_time = time.perf_counter()
        model.fit(X_train, y_train)
        training_time = time.perf_counter() - start_time

        metrics = evaluate_classifier(
            model,
            X_validation,
            y_validation,
        )

        result = {
            "model_name": "url_logistic_regression",
            "parameters": params,
            "training_time_seconds": training_time,
            **metrics,
        }

        results.append(result)

        print(f"Validation F1: {metrics['f1']:.4f}")

    best = max(results, key=lambda item: item["f1"])

    print(
        f"\nBest URL Logistic Regression: "
        f"C={best['parameters']['C']} | "
        f"F1={best['f1']:.4f}"
    )

    return {
        "model_name": "url_logistic_regression",
        "search_status": "completed",
        "candidates": results,
        "best_result": best,
    }


def tune_url_random_forest(
    X_train,
    y_train,
    X_validation,
    y_validation,
) -> dict[str, Any]:

    print("\n" + "=" * 70)
    print("Tuning URL model: url_random_forest")
    print("=" * 70)

    candidates = [
        {
            "n_estimators": 100,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
        },
        {
            "n_estimators": 200,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
        },
        {
            "n_estimators": 300,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
        },
        {
            "n_estimators": 200,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
        },
    ]

    results = []

    for index, params in enumerate(candidates, start=1):
        print(f"\nCandidate {index}/{len(candidates)}")
        print(f"Parameters: {params}")

        model = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            min_samples_leaf=params["min_samples_leaf"],
            max_features=params["max_features"],
            n_jobs=-1,
            random_state=42,
        )

        start_time = time.perf_counter()
        model.fit(X_train, y_train)
        training_time = time.perf_counter() - start_time

        metrics = evaluate_classifier(
            model,
            X_validation,
            y_validation,
        )

        result = {
            "model_name": "url_random_forest",
            "parameters": params,
            "training_time_seconds": training_time,
            **metrics,
        }

        results.append(result)

        print(f"Validation F1: {metrics['f1']:.4f}")

    best = max(results, key=lambda item: item["f1"])

    print(
        f"\nBest URL Random Forest: "
        f"{best['parameters']} | "
        f"F1={best['f1']:.4f}"
    )

    return {
        "model_name": "url_random_forest",
        "search_status": "completed",
        "candidates": results,
        "best_result": best,
    }


def tune_url_gradient_boosting(
    X_train,
    y_train,
    X_validation,
    y_validation,
) -> dict[str, Any]:

    print("\n" + "=" * 70)
    print("Tuning URL model: url_gradient_boosting")
    print("=" * 70)

    candidates = [
        {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 3,
        },
        {
            "n_estimators": 200,
            "learning_rate": 0.1,
            "max_depth": 3,
        },
        {
            "n_estimators": 300,
            "learning_rate": 0.1,
            "max_depth": 3,
        },
        {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "max_depth": 3,
        },
    ]

    results = []

    for index, params in enumerate(candidates, start=1):
        print(f"\nCandidate {index}/{len(candidates)}")
        print(f"Parameters: {params}")

        model = GradientBoostingClassifier(
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            max_depth=params["max_depth"],
            random_state=42,
        )

        start_time = time.perf_counter()
        model.fit(X_train, y_train)
        training_time = time.perf_counter() - start_time

        metrics = evaluate_classifier(
            model,
            X_validation,
            y_validation,
        )

        result = {
            "model_name": "url_gradient_boosting",
            "parameters": params,
            "training_time_seconds": training_time,
            **metrics,
        }

        results.append(result)

        print(f"Validation F1: {metrics['f1']:.4f}")

    best = max(results, key=lambda item: item["f1"])

    print(
        f"\nBest URL Gradient Boosting: "
        f"{best['parameters']} | "
        f"F1={best['f1']:.4f}"
    )

    return {
        "model_name": "url_gradient_boosting",
        "search_status": "completed",
        "candidates": results,
        "best_result": best,
    }


def load_email_data():
    print("\nLoading email datasets...")

    train_loader = EmailDatasetLoader(EMAIL_TRAIN_PATH)
    validation_loader = EmailDatasetLoader(EMAIL_VALIDATION_PATH)

    X_train_text, y_train = train_loader.get_texts_and_labels()
    X_validation_text, y_validation = (
        validation_loader.get_texts_and_labels()
    )

    print(f"Email training samples: {len(X_train_text)}")
    print(f"Email validation samples: {len(X_validation_text)}")

    print("\nFitting email TF-IDF extractor once...")

    extractor = EmailTfidfExtractor()

    X_train = extractor.fit_transform(X_train_text)
    X_validation = extractor.transform(X_validation_text)

    print(f"Email TF-IDF features: {X_train.shape[1]}")

    return X_train, y_train, X_validation, y_validation


def load_url_data():
    print("\nLoading URL datasets...")

    train_loader = URLDatasetLoader(URL_TRAIN_PATH)
    validation_loader = URLDatasetLoader(URL_VALIDATION_PATH)

    X_train_df, y_train = train_loader.get_features_and_labels()
    X_validation_df, y_validation = (
        validation_loader.get_features_and_labels()
    )

    print(f"URL training samples: {len(X_train_df)}")
    print(f"URL validation samples: {len(X_validation_df)}")

    print("\nFitting URL feature extractor once...")

    extractor = URLFeatureExtractor()

    X_train = extractor.fit_transform(X_train_df)
    X_validation = extractor.transform(X_validation_df)

    print(f"URL features: {X_train.shape[1]}")

    return X_train, y_train, X_validation, y_validation


def main() -> None:
    print("\n" + "#" * 70)
    print("PHASE 2 — TARGETED HYPERPARAMETER TUNING")
    print("#" * 70)

    print("\nExperimental safeguards:")
    print("- Validation data is used for model selection.")
    print("- Test data is NOT loaded.")
    print("- TF-IDF is fitted only on training data.")
    print("- URL preprocessing is fitted only on training data.")
    print("- Email Random Forest excessive candidate is excluded.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    (
        email_X_train,
        email_y_train,
        email_X_validation,
        email_y_validation,
    ) = load_email_data()

    (
        url_X_train,
        url_y_train,
        url_X_validation,
        url_y_validation,
    ) = load_url_data()

    results = {
        "experiment": "phase2_targeted_hyperparameter_tuning",
        "selection_metric": "f1",
        "test_set_used": False,
        "random_state": 42,
        "email": {},
        "url": {},
    }

    # ---------------------------------------------------------------
    # EMAIL MODELS
    # ---------------------------------------------------------------

    results["email"]["email_logistic_regression"] = (
        tune_email_logistic_regression(
            email_X_train,
            email_y_train,
            email_X_validation,
            email_y_validation,
        )
    )

    results["email"]["email_linear_svm"] = tune_email_linear_svm(
        email_X_train,
        email_y_train,
        email_X_validation,
        email_y_validation,
    )

    results["email"]["email_random_forest"] = (
        record_email_random_forest_results()
    )

    # ---------------------------------------------------------------
    # URL MODELS
    # ---------------------------------------------------------------

    results["url"]["url_logistic_regression"] = (
        tune_url_logistic_regression(
            url_X_train,
            url_y_train,
            url_X_validation,
            url_y_validation,
        )
    )

    results["url"]["url_random_forest"] = tune_url_random_forest(
        url_X_train,
        url_y_train,
        url_X_validation,
        url_y_validation,
    )

    results["url"]["url_gradient_boosting"] = (
        tune_url_gradient_boosting(
            url_X_train,
            url_y_train,
            url_X_validation,
            url_y_validation,
        )
    )

    # ---------------------------------------------------------------
    # SAVE RESULTS
    # ---------------------------------------------------------------

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            results,
            file,
            indent=2,
        )

    print("\n" + "=" * 70)
    print("HYPERPARAMETER TUNING COMPLETE")
    print("=" * 70)

    print(f"\nResults saved to:")
    print(OUTPUT_PATH)

    print("\nBest models by validation F1:")

    for modality in ("email", "url"):
        print(f"\n{modality.upper()}")

        for model_name, model_result in results[modality].items():
            best = model_result["best_result"]

            print(
                f"{model_name}: "
                f"F1={best['f1']:.4f} | "
                f"parameters={best['parameters']}"
            )


if __name__ == "__main__":
    main()