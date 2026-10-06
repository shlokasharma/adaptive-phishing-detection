"""
Phase 2 cross-dataset evaluation for the selected email model.

The selected email model is trained only on the frozen Phase 1
training split and evaluated on external email datasets.

No retraining, fitting, or hyperparameter tuning is performed on
the external datasets.

Selected model:
    email_linear_svm

External datasets:
    Enron
    Ling
    Nazario
    Nazario_5
    Nigerian_5
    Nigerian_Fraud
    SpamAssasin
    TREC_07
"""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import pandas as pd

from phishing_detection.evaluation.metrics import (
    calculate_classification_metrics,
)
from phishing_detection.models.base import BasePhishingDetector


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_FILE = (
    PROJECT_ROOT
    / "experiments"
    / "baselines"
    / "models"
    / "email_linear_svm.pkl"
)

TFIDF_FILE = (
    PROJECT_ROOT
    / "experiments"
    / "baselines"
    / "models"
    / "email_linear_svm_tfidf_extractor.pkl"
)

EMAIL_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "email"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "baselines"
    / "outputs"
)

OUTPUT_JSON = OUTPUT_DIR / "email_cross_dataset_metrics.json"
OUTPUT_CSV = OUTPUT_DIR / "email_cross_dataset_metrics.csv"


EXTERNAL_DATASETS = [
    "Enron_cleaned.csv",
    "Ling_cleaned.csv",
    "Nazario_cleaned.csv",
    "Nazario_5_cleaned.csv",
    "Nigerian_5_cleaned.csv",
    "Nigerian_Fraud_cleaned.csv",
    "SpamAssasin_cleaned.csv",
    "TREC_07_cleaned.csv",
]


def load_pickle(path: Path):
    """Load a serialized model or feature extractor."""

    if not path.exists():
        raise FileNotFoundError(
            f"Required artifact not found:\n{path}"
        )

    with path.open("rb") as file:
        return pickle.load(file)


def validate_dataset(df: pd.DataFrame, dataset_name: str) -> None:
    """Validate the external email dataset schema."""

    required_columns = {
        "sample_id",
        "source_dataset",
        "label",
        "clean_text",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"{dataset_name} is missing required columns: "
            f"{sorted(missing)}"
        )


def evaluate_dataset(
    model: BasePhishingDetector,
    extractor,
    dataset_path: Path,
) -> dict:
    """Evaluate the selected model on one external dataset."""

    dataset_name = dataset_path.stem

    df = pd.read_csv(dataset_path)

    validate_dataset(df, dataset_name)

    texts = (
        df["clean_text"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    y_true = df["label"].astype(int).to_numpy()

    start_time = time.perf_counter()

    X = extractor.transform(texts)

    transform_time = time.perf_counter() - start_time

    start_time = time.perf_counter()

    y_pred = model.predict(X)

    inference_time = time.perf_counter() - start_time

    start_time = time.perf_counter()

    y_probability = model.predict_proba(X)[:, 1]

    probability_time = time.perf_counter() - start_time

    metrics = calculate_classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_probability=y_probability,
    )

    result = metrics.to_dict()

    result.update(
        {
            "dataset": dataset_name,
            "num_samples": int(len(df)),
            "transform_time_seconds": float(transform_time),
            "inference_time_seconds": float(inference_time),
            "probability_time_seconds": float(probability_time),
            "inference_time_per_sample_ms": float(
                inference_time / len(df) * 1000
            ),
        }
    )

    return result


def main() -> None:
    """Run email cross-dataset evaluation."""

    print("=" * 70)
    print("PHASE 2 — EMAIL CROSS-DATASET GENERALIZATION")
    print("=" * 70)

    print("\nLoading selected model...")
    model = load_pickle(MODEL_FILE)

    print("Loading TF-IDF extractor...")
    extractor = load_pickle(TFIDF_FILE)

    print("\nSelected model:")
    print(f"  {model.get_model_name()}")

    print("\nExternal datasets:")
    for dataset in EXTERNAL_DATASETS:
        print(f"  - {dataset}")

    results = []

    for filename in EXTERNAL_DATASETS:

        dataset_path = EMAIL_DATA_DIR / filename

        if not dataset_path.exists():
            print(
                f"\nWARNING: dataset not found, skipping: "
                f"{dataset_path}"
            )
            continue

        print(f"\nEvaluating: {filename}")

        result = evaluate_dataset(
            model=model,
            extractor=extractor,
            dataset_path=dataset_path,
        )

        results.append(result)

        print(
            f"  Samples : {result['num_samples']}"
        )
        print(
            f"  Accuracy: {result['accuracy']:.4f}"
        )
        print(
            f"  F1      : {result['f1']:.4f}"
        )
        print(
            f"  ROC-AUC : {result['roc_auc']:.4f}"
        )
        print(
            f"  PR-AUC  : {result['pr_auc']:.4f}"
        )

    if not results:
        raise RuntimeError(
            "No external email datasets were available."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = {
        "experiment": "phase2_email_cross_dataset_generalization",
        "model": "email_linear_svm",
        "training_data": (
            "data/splits/email/train.csv"
        ),
        "external_retraining": False,
        "external_hyperparameter_tuning": False,
        "datasets": results,
    }

    with OUTPUT_JSON.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            indent=2,
        )

    csv_rows = []

    for result in results:
        row = {
            "dataset": result["dataset"],
            "num_samples": result["num_samples"],
            "accuracy": result["accuracy"],
            "precision": result["precision"],
            "recall": result["recall"],
            "f1": result["f1"],
            "roc_auc": result["roc_auc"],
            "pr_auc": result["pr_auc"],
            "inference_time_seconds": result[
                "inference_time_seconds"
            ],
            "inference_time_per_sample_ms": result[
                "inference_time_per_sample_ms"
            ],
        }

        csv_rows.append(row)

    pd.DataFrame(csv_rows).to_csv(
        OUTPUT_CSV,
        index=False,
    )

    print("\n" + "=" * 70)
    print("CROSS-DATASET EVALUATION COMPLETE")
    print("=" * 70)

    print(f"\nJSON: {OUTPUT_JSON}")
    print(f"CSV : {OUTPUT_CSV}")


if __name__ == "__main__":
    main()