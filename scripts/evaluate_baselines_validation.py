"""
Evaluate all Phase 2 baseline models on validation splits.

Important:
    - Training data is used only to fit preprocessing/model artifacts.
    - Validation data is used only for evaluation.
    - Test data is NOT accessed by this script.

Models evaluated:
    Email:
        - Logistic Regression
        - Linear SVM
        - Random Forest

    URL:
        - Logistic Regression
        - Random Forest
        - Gradient Boosting

Metrics:
    - Accuracy
    - Precision
    - Recall
    - F1
    - ROC-AUC
    - PR-AUC
    - Confusion matrix

Label convention:
    0 = legitimate
    1 = phishing
"""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import numpy as np

from phishing_detection.data.email_loader import (
    EmailDatasetLoader,
)
from phishing_detection.data.url_loader import (
    URLDatasetLoader,
)
from phishing_detection.evaluation.metrics import (
    calculate_classification_metrics,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

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

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

EMAIL_VALIDATION_FILE = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "email"
    / "validation.csv"
)

URL_VALIDATION_FILE = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "url"
    / "uci_features"
    / "validation.csv"
)


def load_pickle(path: Path):
    """Load a serialized Python object."""

    if not path.exists():
        raise FileNotFoundError(
            f"Artifact not found: {path}"
        )

    with path.open("rb") as file:
        return pickle.load(file)


def evaluate_model(
    model,
    X,
    y,
) -> dict[str, object]:
    """
    Generate predictions and calculate classification metrics.
    """

    start_time = time.perf_counter()

    predictions = model.predict(X)

    probabilities = model.predict_proba(X)

    elapsed = time.perf_counter() - start_time

    phishing_probabilities = probabilities[:, 1]

    metrics = calculate_classification_metrics(
        y_true=np.asarray(y),
        y_pred=np.asarray(predictions),
        y_probability=np.asarray(
            phishing_probabilities
        ),
    )

    result = metrics.to_dict()

    result["inference_time_seconds"] = float(
        elapsed
    )

    result["samples_evaluated"] = int(
        len(y)
    )

    result["inference_time_per_sample_ms"] = float(
        (elapsed / len(y)) * 1000
    )

    return result


def evaluate_email_model(
    model_name: str,
    extractor_name: str,
    X_text,
    y,
) -> dict[str, object]:

    print("=" * 80)
    print(f"Evaluating email model: {model_name}")
    print("=" * 80)

    model_path = MODEL_DIR / f"{model_name}.pkl"
    extractor_path = (
        MODEL_DIR
        / f"{extractor_name}.pkl"
    )

    model = load_pickle(model_path)
    extractor = load_pickle(extractor_path)

    print("Transforming validation emails...")

    X_transformed = extractor.transform(
        X_text
    )

    print(
        f"Validation matrix shape: "
        f"{X_transformed.shape}"
    )

    result = evaluate_model(
        model=model,
        X=X_transformed,
        y=y,
    )

    result["model_name"] = model_name
    result["modality"] = "email"
    result["dataset"] = "email_combined"
    result["split"] = "validation"

    print_metrics(result)

    return result


def evaluate_url_model(
    model_name: str,
    extractor_name: str,
    X_features,
    y,
) -> dict[str, object]:

    print("=" * 80)
    print(f"Evaluating URL model: {model_name}")
    print("=" * 80)

    model_path = MODEL_DIR / f"{model_name}.pkl"
    extractor_path = (
        MODEL_DIR
        / f"{extractor_name}.pkl"
    )

    model = load_pickle(model_path)
    extractor = load_pickle(extractor_path)

    print("Transforming validation URLs...")

    X_transformed = extractor.transform(
        X_features
    )

    print(
        f"Validation matrix shape: "
        f"{X_transformed.shape}"
    )

    result = evaluate_model(
        model=model,
        X=X_transformed,
        y=y,
    )

    result["model_name"] = model_name
    result["modality"] = "url"
    result["dataset"] = "uci_phishing_websites"
    result["split"] = "validation"

    print_metrics(result)

    return result


def print_metrics(
    result: dict[str, object],
) -> None:

    print(
        f"Accuracy : "
        f"{result['accuracy']:.4f}"
    )

    print(
        f"Precision: "
        f"{result['precision']:.4f}"
    )

    print(
        f"Recall   : "
        f"{result['recall']:.4f}"
    )

    print(
        f"F1       : "
        f"{result['f1']:.4f}"
    )

    print(
        f"ROC-AUC  : "
        f"{result['roc_auc']:.4f}"
    )

    print(
        f"PR-AUC   : "
        f"{result['pr_auc']:.4f}"
    )

    print(
        "Confusion matrix:"
    )

    print(
        result["confusion_matrix"]
    )

    print(
        f"Inference: "
        f"{result['inference_time_seconds']:.4f} s"
    )

    print()


def save_results(
    results: list[dict[str, object]],
) -> Path:

    output_path = (
        OUTPUT_DIR
        / "validation_metrics.json"
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            {
                "phase": "phase_2",
                "evaluation": "validation",
                "label_convention": {
                    "0": "legitimate",
                    "1": "phishing",
                },
                "results": results,
            },
            file,
            indent=2,
        )

    return output_path


def main() -> None:

    results: list[
        dict[str, object]
    ] = []

    # ------------------------------------------------------------------
    # Email validation data
    # ------------------------------------------------------------------

    print(
        "Loading email validation split..."
    )

    email_loader = EmailDatasetLoader(
        EMAIL_VALIDATION_FILE
    )

    email_texts, email_labels = (
        email_loader.get_texts_and_labels()
    )

    print(
        f"Email validation samples: "
        f"{len(email_texts)}"
    )

    email_models = [
        (
            "email_logistic_regression",
            "email_logistic_regression_tfidf_extractor",
        ),
        (
            "email_linear_svm",
            "email_linear_svm_tfidf_extractor",
        ),
        (
            "email_random_forest",
            "email_random_forest_tfidf_extractor",
        ),
    ]

    for (
        model_name,
        extractor_name,
    ) in email_models:

        result = evaluate_email_model(
            model_name=model_name,
            extractor_name=extractor_name,
            X_text=email_texts,
            y=email_labels,
        )

        results.append(result)

    # ------------------------------------------------------------------
    # URL validation data
    # ------------------------------------------------------------------

    print(
        "Loading UCI URL validation split..."
    )

    url_loader = URLDatasetLoader(
        URL_VALIDATION_FILE
    )

    url_features, url_labels = (
        url_loader.get_features_and_labels()
    )

    print(
        f"URL validation samples: "
        f"{len(url_features)}"
    )

    url_models = [
        (
            "url_logistic_regression",
            "url_logistic_regression_feature_extractor",
        ),
        (
            "url_random_forest",
            "url_random_forest_feature_extractor",
        ),
        (
            "url_gradient_boosting",
            "url_gradient_boosting_feature_extractor",
        ),
    ]

    for (
        model_name,
        extractor_name,
    ) in url_models:

        result = evaluate_url_model(
            model_name=model_name,
            extractor_name=extractor_name,
            X_features=url_features,
            y=url_labels,
        )

        results.append(result)

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------

    output_path = save_results(
        results
    )

    print("=" * 80)
    print(
        "Validation evaluation completed."
    )
    print("=" * 80)

    print(
        f"Saved results to:\n{output_path}"
    )

    print(
        f"Total models evaluated: "
        f"{len(results)}"
    )


if __name__ == "__main__":
    main()