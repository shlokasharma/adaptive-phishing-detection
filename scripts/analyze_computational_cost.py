"""
Phase 2 computational-cost and inference-time analysis.

Selected models:

    Email -> Linear SVM
    URL   -> Random Forest

The analysis measures:

    - Feature transformation time
    - Prediction time
    - Probability estimation time
    - Total inference time
    - Time per sample
    - Throughput
    - Number of samples
    - Number of model features

No model retraining is performed.
"""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import pandas as pd


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

EMAIL_MODEL = MODEL_DIR / "email_linear_svm.pkl"

EMAIL_EXTRACTOR = (
    MODEL_DIR
    / "email_linear_svm_tfidf_extractor.pkl"
)

EMAIL_TEST = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "email"
    / "test.csv"
)

URL_MODEL = MODEL_DIR / "url_random_forest.pkl"

URL_EXTRACTOR = (
    MODEL_DIR
    / "url_random_forest_feature_extractor.pkl"
)

URL_TEST = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "url"
    / "uci_features"
    / "test.csv"
)


def load_pickle(path: Path):
    """Load a serialized Python artifact."""

    if not path.exists():
        raise FileNotFoundError(
            f"Required artifact not found:\n{path}"
        )

    with path.open("rb") as file:
        return pickle.load(file)


def calculate_throughput(
    num_samples: int,
    seconds: float,
) -> float:
    """Calculate samples processed per second."""

    if seconds <= 0:
        return 0.0

    return float(num_samples / seconds)


def evaluate_email() -> dict:
    """Measure computational cost for the email model."""

    print("\n" + "-" * 70)
    print("EMAIL — LINEAR SVM")
    print("-" * 70)

    model = load_pickle(EMAIL_MODEL)
    extractor = load_pickle(EMAIL_EXTRACTOR)

    df = pd.read_csv(EMAIL_TEST)

    texts = (
        df["clean_text"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    num_samples = len(texts)

    print(f"Samples: {num_samples:,}")

    # ---------------------------------------------------------------
    # Feature transformation
    # ---------------------------------------------------------------

    start = time.perf_counter()

    X = extractor.transform(texts)

    transform_time = time.perf_counter() - start

    # ---------------------------------------------------------------
    # Class prediction
    # ---------------------------------------------------------------

    start = time.perf_counter()

    y_pred = model.predict(X)

    prediction_time = time.perf_counter() - start

    # ---------------------------------------------------------------
    # Probability estimation
    # ---------------------------------------------------------------

    start = time.perf_counter()

    y_probability = model.predict_proba(X)[:, 1]

    probability_time = time.perf_counter() - start

    total_time = (
        transform_time
        + prediction_time
        + probability_time
    )

    result = {
        "modality": "email",
        "model": "email_linear_svm",
        "num_samples": int(num_samples),
        "num_features": int(X.shape[1]),
        "feature_transformation_time_seconds": float(
            transform_time
        ),
        "prediction_time_seconds": float(
            prediction_time
        ),
        "probability_time_seconds": float(
            probability_time
        ),
        "total_inference_time_seconds": float(
            total_time
        ),
        "transformation_time_per_sample_ms": float(
            transform_time / num_samples * 1000
        ),
        "prediction_time_per_sample_ms": float(
            prediction_time / num_samples * 1000
        ),
        "probability_time_per_sample_ms": float(
            probability_time / num_samples * 1000
        ),
        "total_time_per_sample_ms": float(
            total_time / num_samples * 1000
        ),
        "throughput_samples_per_second": calculate_throughput(
            num_samples,
            total_time,
        ),
    }

    print(
        f"Feature transformation : "
        f"{transform_time:.4f} s"
    )
    print(
        f"Prediction              : "
        f"{prediction_time:.4f} s"
    )
    print(
        f"Probability estimation  : "
        f"{probability_time:.4f} s"
    )
    print(
        f"Total                   : "
        f"{total_time:.4f} s"
    )
    print(
        f"Per sample              : "
        f"{result['total_time_per_sample_ms']:.4f} ms"
    )
    print(
        f"Throughput              : "
        f"{result['throughput_samples_per_second']:.2f} "
        f"samples/s"
    )
    print(
        f"Features                : "
        f"{X.shape[1]:,}"
    )

    # Avoid unused-variable warnings in some environments.
    _ = y_pred
    _ = y_probability

    return result


def evaluate_url() -> dict:
    """Measure computational cost for the URL model."""

    print("\n" + "-" * 70)
    print("URL — RANDOM FOREST")
    print("-" * 70)

    model = load_pickle(URL_MODEL)
    extractor = load_pickle(URL_EXTRACTOR)

    df = pd.read_csv(URL_TEST)

    feature_columns = [
        column
        for column in df.columns
        if column not in {
            "sample_id",
            "source_dataset",
            "label",
        }
    ]

    X_raw = df[feature_columns].copy()

    num_samples = len(X_raw)

    print(f"Samples: {num_samples:,}")

    # ---------------------------------------------------------------
    # Feature transformation
    # ---------------------------------------------------------------

    start = time.perf_counter()

    X = extractor.transform(X_raw)

    transform_time = time.perf_counter() - start

    # ---------------------------------------------------------------
    # Class prediction
    # ---------------------------------------------------------------

    start = time.perf_counter()

    y_pred = model.predict(X)

    prediction_time = time.perf_counter() - start

    # ---------------------------------------------------------------
    # Probability estimation
    # ---------------------------------------------------------------

    start = time.perf_counter()

    y_probability = model.predict_proba(X)[:, 1]

    probability_time = time.perf_counter() - start

    total_time = (
        transform_time
        + prediction_time
        + probability_time
    )

    result = {
        "modality": "url",
        "model": "url_random_forest",
        "num_samples": int(num_samples),
        "num_features": int(X.shape[1]),
        "feature_transformation_time_seconds": float(
            transform_time
        ),
        "prediction_time_seconds": float(
            prediction_time
        ),
        "probability_time_seconds": float(
            probability_time
        ),
        "total_inference_time_seconds": float(
            total_time
        ),
        "transformation_time_per_sample_ms": float(
            transform_time / num_samples * 1000
        ),
        "prediction_time_per_sample_ms": float(
            prediction_time / num_samples * 1000
        ),
        "probability_time_per_sample_ms": float(
            probability_time / num_samples * 1000
        ),
        "total_time_per_sample_ms": float(
            total_time / num_samples * 1000
        ),
        "throughput_samples_per_second": calculate_throughput(
            num_samples,
            total_time,
        ),
    }

    print(
        f"Feature transformation : "
        f"{transform_time:.4f} s"
    )
    print(
        f"Prediction              : "
        f"{prediction_time:.4f} s"
    )
    print(
        f"Probability estimation  : "
        f"{probability_time:.4f} s"
    )
    print(
        f"Total                   : "
        f"{total_time:.4f} s"
    )
    print(
        f"Per sample              : "
        f"{result['total_time_per_sample_ms']:.4f} ms"
    )
    print(
        f"Throughput              : "
        f"{result['throughput_samples_per_second']:.2f} "
        f"samples/s"
    )
    print(
        f"Features                : "
        f"{X.shape[1]:,}"
    )

    _ = y_pred
    _ = y_probability

    return result


def main() -> None:
    """Run computational-cost analysis."""

    print("=" * 70)
    print("PHASE 2 — COMPUTATIONAL COST ANALYSIS")
    print("=" * 70)

    email_result = evaluate_email()
    url_result = evaluate_url()

    results = [
        email_result,
        url_result,
    ]

    output = {
        "experiment": (
            "phase2_computational_cost_analysis"
        ),
        "measurement_scope": (
            "held-out test-set inference"
        ),
        "retraining_performed": False,
        "models": results,
    }

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_file = (
        OUTPUT_DIR
        / "computational_cost_analysis.json"
    )

    with json_file.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            indent=2,
        )

    summary_rows = []

    for result in results:
        summary_rows.append(
            {
                "Modality": result["modality"],
                "Model": result["model"],
                "Samples": result["num_samples"],
                "Features": result["num_features"],
                "Transform Time (s)": result[
                    "feature_transformation_time_seconds"
                ],
                "Prediction Time (s)": result[
                    "prediction_time_seconds"
                ],
                "Probability Time (s)": result[
                    "probability_time_seconds"
                ],
                "Total Time (s)": result[
                    "total_inference_time_seconds"
                ],
                "Time per Sample (ms)": result[
                    "total_time_per_sample_ms"
                ],
                "Throughput (samples/s)": result[
                    "throughput_samples_per_second"
                ],
            }
        )

    summary = pd.DataFrame(summary_rows)

    csv_file = (
        OUTPUT_DIR
        / "computational_cost_summary.csv"
    )

    summary.to_csv(
        csv_file,
        index=False,
    )

    print("\n" + "=" * 70)
    print("COMPUTATIONAL COST SUMMARY")
    print("=" * 70)

    print(
        summary.to_string(
            index=False
        )
    )

    print("\nOUTPUT FILES")
    print("-" * 70)
    print(json_file)
    print(csv_file)

    print("\nSTEP 21 ANALYSIS COMPLETE")


if __name__ == "__main__":
    main()