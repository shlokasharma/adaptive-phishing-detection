"""
Step 13: Targeted hyperparameter tuning for Phase 2 baseline models.

The held-out test datasets are NEVER used in this script.

Model selection is performed using validation F1-score.

Email:
    - Logistic Regression
    - Linear SVM
    - Random Forest (previously completed candidates recorded)

URL:
    - Logistic Regression
    - Random Forest
    - Gradient Boosting

Output:
    experiments/baselines/outputs/hyperparameter_tuning_results.json
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from phishing_detection.data.email_loader import EmailDatasetLoader
from phishing_detection.data.url_loader import URLDatasetLoader
from phishing_detection.evaluation.metrics import (
    calculate_classification_metrics,
)
from phishing_detection.features.email_tfidf import EmailTfidfExtractor
from phishing_detection.features.url_features import URLFeatureExtractor


# ---------------------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EMAIL_TRAIN_PATH = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "email"
    / "train.csv"
)

EMAIL_VALIDATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "email"
    / "validation.csv"
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

OUTPUT_PATH = (
    OUTPUT_DIR
    / "hyperparameter_tuning_results.json"
)


# ---------------------------------------------------------------------
# GENERIC EVALUATION
# ---------------------------------------------------------------------

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
        y_probability = model.predict_proba(
            X_validation
        )[:, 1]

    elif hasattr(model, "decision_function"):
        decision_scores = model.decision_function(
            X_validation
        )

        y_probability = 1.0 / (
            1.0 + np.exp(-decision_scores)
        )

    else:
        raise AttributeError(
            "Model must provide either predict_proba() "
            "or decision_function()."
        )

    metrics = calculate_classification_metrics(
        y_true=np.asarray(y_validation),
        y_pred=np.asarray(y_pred),
        y_probability=np.asarray(y_probability),
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


# ---------------------------------------------------------------------
# EMAIL DATA
# ---------------------------------------------------------------------

def load_email_data():
    """
    Load email train/validation data and fit TF-IDF only on training data.
    """

    print("\n" + "=" * 70)
    print("LOADING EMAIL DATA")
    print("=" * 70)

    train_loader = EmailDatasetLoader(
        EMAIL_TRAIN_PATH
    )

    validation_loader = EmailDatasetLoader(
        EMAIL_VALIDATION_PATH
    )

    train_texts, y_train = (
        train_loader.get_texts_and_labels()
    )

    validation_texts, y_validation = (
        validation_loader.get_texts_and_labels()
    )

    print(
        f"Training samples: {len(train_texts)}"
    )

    print(
        f"Validation samples: {len(validation_texts)}"
    )

    print("\nFitting TF-IDF on training data only...")

    extractor = EmailTfidfExtractor()

    X_train = extractor.fit_transform(
        train_texts
    )

    X_validation = extractor.transform(
        validation_texts
    )

    print(
        f"TF-IDF feature count: {X_train.shape[1]}"
    )

    return (
        X_train,
        y_train,
        X_validation,
        y_validation,
    )


# ---------------------------------------------------------------------
# URL DATA
# ---------------------------------------------------------------------

def load_url_data():
    """
    Load URL train/validation data and fit preprocessing
    only on training data.
    """

    print("\n" + "=" * 70)
    print("LOADING URL DATA")
    print("=" * 70)

    train_loader = URLDatasetLoader(
        URL_TRAIN_PATH
    )

    validation_loader = URLDatasetLoader(
        URL_VALIDATION_PATH
    )

    X_train_df, y_train = (
        train_loader.get_features_and_labels()
    )

    X_validation_df, y_validation = (
        validation_loader.get_features_and_labels()
    )

    print(
        f"Training samples: {len(X_train_df)}"
    )

    print(
        f"Validation samples: {len(X_validation_df)}"
    )

    print("\nFitting URL preprocessing on training data only...")

    extractor = URLFeatureExtractor()

    X_train = extractor.fit_transform(
        X_train_df
    )

    X_validation = extractor.transform(
        X_validation_df
    )

    print(
        f"URL feature count: {X_train.shape[1]}"
    )

    return (
        X_train,
        y_train,
        X_validation,
        y_validation,
    )


# ---------------------------------------------------------------------
# EMAIL LOGISTIC REGRESSION
# ---------------------------------------------------------------------

def tune_email_logistic_regression(
    X_train,
    y_train,
    X_validation,
    y_validation,
):
    print("\n" + "=" * 70)
    print("EMAIL — LOGISTIC REGRESSION")
    print("=" * 70)

    candidates = [
        {"C": 0.5},
        {"C": 1.0},
        {"C": 2.0},
        {"C": 4.0},
    ]

    results = []

    for index, params in enumerate(
        candidates,
        start=1,
    ):

        print(
            f"\nCandidate {index}/{len(candidates)}"
        )

        print(
            f"Parameters: {params}"
        )

        model = LogisticRegression(
            C=params["C"],
            max_iter=1000,
            solver="liblinear",
            random_state=42,
        )

        start_time = time.perf_counter()

        model.fit(
            X_train,
            y_train,
        )

        training_time = (
            time.perf_counter()
            - start_time
        )

        metrics = evaluate_classifier(
            model,
            X_validation,
            y_validation,
        )

        result = {
            "parameters": params,
            "training_time_seconds": training_time,
            **metrics,
        }

        results.append(result)

        print(
            f"Validation F1: "
            f"{metrics['f1']:.4f}"
        )

    best = max(
        results,
        key=lambda item: item["f1"],
    )

    print(
        "\nBest Email Logistic Regression:"
    )

    print(
        f"C={best['parameters']['C']}, "
        f"F1={best['f1']:.4f}"
    )

    return {
        "model_name": "email_logistic_regression",
        "search_status": "completed",
        "candidates": results,
        "best_result": best,
    }


# ---------------------------------------------------------------------
# EMAIL LINEAR SVM
# ---------------------------------------------------------------------

def tune_email_linear_svm(
    X_train,
    y_train,
    X_validation,
    y_validation,
):
    print("\n" + "=" * 70)
    print("EMAIL — LINEAR SVM")
    print("=" * 70)

    candidates = [
        {"C": 0.5},
        {"C": 1.0},
        {"C": 2.0},
        {"C": 4.0},
    ]

    results = []

    for index, params in enumerate(
        candidates,
        start=1,
    ):

        print(
            f"\nCandidate {index}/{len(candidates)}"
        )

        print(
            f"Parameters: {params}"
        )

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

        model.fit(
            X_train,
            y_train,
        )

        training_time = (
            time.perf_counter()
            - start_time
        )

        metrics = evaluate_classifier(
            model,
            X_validation,
            y_validation,
        )

        result = {
            "parameters": params,
            "training_time_seconds": training_time,
            **metrics,
        }

        results.append(result)

        print(
            f"Validation F1: "
            f"{metrics['f1']:.4f}"
        )

    best = max(
        results,
        key=lambda item: item["f1"],
    )

    print(
        "\nBest Email Linear SVM:"
    )

    print(
        f"C={best['parameters']['C']}, "
        f"F1={best['f1']:.4f}"
    )

    return {
        "model_name": "email_linear_svm",
        "search_status": "completed",
        "candidates": results,
        "best_result": best,
    }


# ---------------------------------------------------------------------
# EMAIL RANDOM FOREST
# ---------------------------------------------------------------------

def record_email_random_forest_results():
    """
    Record the four completed candidates from the interrupted
    tuning experiment.

    The fifth candidate was computationally excessive and was
    intentionally excluded.
    """

    results = [
        {
            "parameters": {
                "n_estimators": 100,
                "min_samples_leaf": 1,
                "max_features": "sqrt",
            },
            "f1": 0.9829,
        },
        {
            "parameters": {
                "n_estimators": 200,
                "min_samples_leaf": 1,
                "max_features": "sqrt",
            },
            "f1": 0.9831,
        },
        {
            "parameters": {
                "n_estimators": 300,
                "min_samples_leaf": 1,
                "max_features": "sqrt",
            },
            "f1": 0.9835,
        },
        {
            "parameters": {
                "n_estimators": 200,
                "min_samples_leaf": 2,
                "max_features": "sqrt",
            },
            "f1": 0.9815,
        },
    ]

    best = max(
        results,
        key=lambda item: item["f1"],
    )

    return {
        "model_name": "email_random_forest",
        "search_status": "partial_completed",
        "candidates": results,
        "best_result": best,
        "excluded_candidate": {
            "parameters": {
                "n_estimators": 200,
                "min_samples_leaf": 1,
                "max_features": 0.5,
            },
            "reason": (
                "Excluded because the configuration was "
                "computationally excessive for the high-dimensional "
                "email TF-IDF representation."
            ),
        },
        "methodological_note": (
            "Four Random Forest candidates completed before "
            "the computationally excessive fifth candidate "
            "was interrupted."
        ),
    }


# ---------------------------------------------------------------------
# URL LOGISTIC REGRESSION
# ---------------------------------------------------------------------

def tune_url_logistic_regression(
    X_train,
    y_train,
    X_validation,
    y_validation,
):
    print("\n" + "=" * 70)
    print("URL — LOGISTIC REGRESSION")
    print("=" * 70)

    candidates = [
        {"C": 0.5},
        {"C": 1.0},
        {"C": 2.0},
        {"C": 4.0},
    ]

    results = []

    for index, params in enumerate(
        candidates,
        start=1,
    ):

        print(
            f"\nCandidate {index}/{len(candidates)}"
        )

        print(
            f"Parameters: {params}"
        )

        model = LogisticRegression(
            C=params["C"],
            max_iter=1000,
            solver="liblinear",
            random_state=42,
        )

        start_time = time.perf_counter()

        model.fit(
            X_train,
            y_train,
        )

        training_time = (
            time.perf_counter()
            - start_time
        )

        metrics = evaluate_classifier(
            model,
            X_validation,
            y_validation,
        )

        result = {
            "parameters": params,
            "training_time_seconds": training_time,
            **metrics,
        }

        results.append(result)

        print(
            f"Validation F1: "
            f"{metrics['f1']:.4f}"
        )

    best = max(
        results,
        key=lambda item: item["f1"],
    )

    print(
        "\nBest URL Logistic Regression:"
    )

    print(
        f"C={best['parameters']['C']}, "
        f"F1={best['f1']:.4f}"
    )

    return {
        "model_name": "url_logistic_regression",
        "search_status": "completed",
        "candidates": results,
        "best_result": best,
    }


# ---------------------------------------------------------------------
# URL RANDOM FOREST
# ---------------------------------------------------------------------

def tune_url_random_forest(
    X_train,
    y_train,
    X_validation,
    y_validation,
):
    print("\n" + "=" * 70)
    print("URL — RANDOM FOREST")
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

    for index, params in enumerate(
        candidates,
        start=1,
    ):

        print(
            f"\nCandidate {index}/{len(candidates)}"
        )

        print(
            f"Parameters: {params}"
        )

        model = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            min_samples_leaf=params["min_samples_leaf"],
            max_features=params["max_features"],
            n_jobs=-1,
            random_state=42,
        )

        start_time = time.perf_counter()

        model.fit(
            X_train,
            y_train,
        )

        training_time = (
            time.perf_counter()
            - start_time
        )

        metrics = evaluate_classifier(
            model,
            X_validation,
            y_validation,
        )

        result = {
            "parameters": params,
            "training_time_seconds": training_time,
            **metrics,
        }

        results.append(result)

        print(
            f"Validation F1: "
            f"{metrics['f1']:.4f}"
        )

    best = max(
        results,
        key=lambda item: item["f1"],
    )

    print(
        "\nBest URL Random Forest:"
    )

    print(
        f"{best['parameters']} | "
        f"F1={best['f1']:.4f}"
    )

    return {
        "model_name": "url_random_forest",
        "search_status": "completed",
        "candidates": results,
        "best_result": best,
    }


# ---------------------------------------------------------------------
# URL GRADIENT BOOSTING
# ---------------------------------------------------------------------

def tune_url_gradient_boosting(
    X_train,
    y_train,
    X_validation,
    y_validation,
):
    print("\n" + "=" * 70)
    print("URL — GRADIENT BOOSTING")
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

    for index, params in enumerate(
        candidates,
        start=1,
    ):

        print(
            f"\nCandidate {index}/{len(candidates)}"
        )

        print(
            f"Parameters: {params}"
        )

        model = GradientBoostingClassifier(
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            max_depth=params["max_depth"],
            random_state=42,
        )

        start_time = time.perf_counter()

        model.fit(
            X_train,
            y_train,
        )

        training_time = (
            time.perf_counter()
            - start_time
        )

        metrics = evaluate_classifier(
            model,
            X_validation,
            y_validation,
        )

        result = {
            "parameters": params,
            "training_time_seconds": training_time,
            **metrics,
        }

        results.append(result)

        print(
            f"Validation F1: "
            f"{metrics['f1']:.4f}"
        )

    best = max(
        results,
        key=lambda item: item["f1"],
    )

    print(
        "\nBest URL Gradient Boosting:"
    )

    print(
        f"{best['parameters']} | "
        f"F1={best['f1']:.4f}"
    )

    return {
        "model_name": "url_gradient_boosting",
        "search_status": "completed",
        "candidates": results,
        "best_result": best,
    }


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():

    print("\n" + "#" * 70)
    print("PHASE 2 — STEP 13")
    print("TARGETED HYPERPARAMETER TUNING")
    print("#" * 70)

    print("\nExperimental rules:")
    print("1. Training data is used for fitting.")
    print("2. Validation data is used for model selection.")
    print("3. Test data is NOT loaded.")
    print("4. TF-IDF is fitted only on email training data.")
    print("5. URL preprocessing is fitted only on URL training data.")
    print("6. Email RF excessive candidate is excluded.")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------------
    # LOAD EMAIL
    # ---------------------------------------------------------------

    (
        email_X_train,
        email_y_train,
        email_X_validation,
        email_y_validation,
    ) = load_email_data()

    # ---------------------------------------------------------------
    # LOAD URL
    # ---------------------------------------------------------------

    (
        url_X_train,
        url_y_train,
        url_X_validation,
        url_y_validation,
    ) = load_url_data()

    results = {
        "experiment": (
            "phase2_targeted_hyperparameter_tuning"
        ),
        "selection_metric": "f1",
        "test_set_used": False,
        "random_state": 42,
        "email": {},
        "url": {},
    }

    # ---------------------------------------------------------------
    # EMAIL
    # ---------------------------------------------------------------

    results["email"][
        "email_logistic_regression"
    ] = tune_email_logistic_regression(
        email_X_train,
        email_y_train,
        email_X_validation,
        email_y_validation,
    )

    results["email"][
        "email_linear_svm"
    ] = tune_email_linear_svm(
        email_X_train,
        email_y_train,
        email_X_validation,
        email_y_validation,
    )

    results["email"][
        "email_random_forest"
    ] = record_email_random_forest_results()

    # ---------------------------------------------------------------
    # URL
    # ---------------------------------------------------------------

    results["url"][
        "url_logistic_regression"
    ] = tune_url_logistic_regression(
        url_X_train,
        url_y_train,
        url_X_validation,
        url_y_validation,
    )

    results["url"][
        "url_random_forest"
    ] = tune_url_random_forest(
        url_X_train,
        url_y_train,
        url_X_validation,
        url_y_validation,
    )

    results["url"][
        "url_gradient_boosting"
    ] = tune_url_gradient_boosting(
        url_X_train,
        url_y_train,
        url_X_validation,
        url_y_validation,
    )

    # ---------------------------------------------------------------
    # SAVE
    # ---------------------------------------------------------------

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            results,
            file,
            indent=2,
        )

    # ---------------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------------

    print("\n" + "=" * 70)
    print("STEP 13 COMPLETE")
    print("=" * 70)

    print(
        f"\nResults saved to:\n{OUTPUT_PATH}"
    )

    print("\nBEST VALIDATION RESULTS:")

    for modality in (
        "email",
        "url",
    ):

        print(
            f"\n{modality.upper()}"
        )

        for model_name, model_result in (
            results[modality].items()
        ):

            best = model_result[
                "best_result"
            ]

            print(
                f"{model_name}: "
                f"F1={best['f1']:.4f} | "
                f"Parameters={best['parameters']}"
            )


if __name__ == "__main__":
    main()