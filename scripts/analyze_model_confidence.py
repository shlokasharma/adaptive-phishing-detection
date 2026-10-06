"""
Phase 2 confidence/probability distribution analysis.

Selected models:

    Email -> Linear SVM
    URL   -> Random Forest

The analysis studies prediction confidence before adaptive
evidence orchestration is introduced.

Confidence is defined as:

    max(P(legitimate), P(phishing))

Descriptive confidence bands:

    Low:
        confidence < 0.60

    Medium:
        0.60 <= confidence < 0.80

    High:
        confidence >= 0.80

These thresholds are descriptive and are not treated as learned
decision thresholds.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "baselines"
    / "outputs"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "baselines"
    / "models"
)

EMAIL_TEST = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "email"
    / "test.csv"
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
    """Load a serialized model artifact."""

    if not path.exists():
        raise FileNotFoundError(
            f"Required artifact not found:\n{path}"
        )

    with path.open("rb") as file:
        return pickle.load(file)


def summarize_probabilities(
    probabilities: np.ndarray,
) -> dict:
    """Calculate probability-distribution statistics."""

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    quantiles = {
        "q01": float(np.quantile(probabilities, 0.01)),
        "q05": float(np.quantile(probabilities, 0.05)),
        "q10": float(np.quantile(probabilities, 0.10)),
        "q25": float(np.quantile(probabilities, 0.25)),
        "q50": float(np.quantile(probabilities, 0.50)),
        "q75": float(np.quantile(probabilities, 0.75)),
        "q90": float(np.quantile(probabilities, 0.90)),
        "q95": float(np.quantile(probabilities, 0.95)),
        "q99": float(np.quantile(probabilities, 0.99)),
    }

    return {
        "mean": float(np.mean(probabilities)),
        "median": float(np.median(probabilities)),
        "std": float(np.std(probabilities)),
        "minimum": float(np.min(probabilities)),
        "maximum": float(np.max(probabilities)),
        "quantiles": quantiles,
    }


def confidence_band_counts(
    confidence: np.ndarray,
) -> dict:
    """Calculate low, medium, and high confidence counts."""

    confidence = np.asarray(
        confidence,
        dtype=float,
    )

    low = confidence < 0.60

    medium = (
        (confidence >= 0.60)
        & (confidence < 0.80)
    )

    high = confidence >= 0.80

    total = len(confidence)

    return {
        "low": {
            "count": int(low.sum()),
            "rate": float(low.mean()) if total else 0.0,
        },
        "medium": {
            "count": int(medium.sum()),
            "rate": float(medium.mean()) if total else 0.0,
        },
        "high": {
            "count": int(high.sum()),
            "rate": float(high.mean()) if total else 0.0,
        },
    }


def analyze_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probability: np.ndarray,
) -> dict:
    """Analyze confidence and probability behavior."""

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_probability = np.asarray(
        y_probability,
        dtype=float,
    )

    confidence = np.maximum(
        y_probability,
        1.0 - y_probability,
    )

    correct = y_true == y_pred
    incorrect = ~correct

    result = {
        "probability_distribution": summarize_probabilities(
            y_probability
        ),
        "confidence_distribution": summarize_probabilities(
            confidence
        ),
        "confidence_bands": confidence_band_counts(
            confidence
        ),
        "mean_confidence_correct": (
            float(np.mean(confidence[correct]))
            if np.any(correct)
            else None
        ),
        "mean_confidence_incorrect": (
            float(np.mean(confidence[incorrect]))
            if np.any(incorrect)
            else None
        ),
        "correct_prediction_count": int(correct.sum()),
        "incorrect_prediction_count": int(incorrect.sum()),
    }

    # Confusion-group confidence.
    groups = {
        "true_negative": (
            (y_true == 0)
            & (y_pred == 0)
        ),
        "false_positive": (
            (y_true == 0)
            & (y_pred == 1)
        ),
        "false_negative": (
            (y_true == 1)
            & (y_pred == 0)
        ),
        "true_positive": (
            (y_true == 1)
            & (y_pred == 1)
        ),
    }

    group_results = {}

    for name, mask in groups.items():

        group_confidence = confidence[mask]

        group_results[name] = {
            "count": int(mask.sum()),
            "mean_confidence": (
                float(np.mean(group_confidence))
                if len(group_confidence)
                else None
            ),
            "median_confidence": (
                float(np.median(group_confidence))
                if len(group_confidence)
                else None
            ),
        }

    result["confusion_group_confidence"] = group_results

    return result


def plot_confidence_distribution(
    confidence: np.ndarray,
    model_name: str,
    filename: str,
) -> None:
    """Save a confidence histogram."""

    plt.figure(figsize=(10, 6))

    plt.hist(
        confidence,
        bins=20,
        edgecolor="black",
    )

    plt.axvline(
        0.60,
        linestyle="--",
        linewidth=1.5,
        label="Low/Medium boundary",
    )

    plt.axvline(
        0.80,
        linestyle="--",
        linewidth=1.5,
        label="Medium/High boundary",
    )

    plt.xlabel("Prediction Confidence")
    plt.ylabel("Number of Samples")

    plt.title(
        f"Prediction Confidence Distribution — "
        f"{model_name}"
    )

    plt.legend()

    plt.tight_layout()

    output_path = OUTPUT_DIR / filename

    plt.savefig(
        output_path,
        dpi=200,
    )

    plt.close()

    print(f"Saved figure: {output_path}")


def evaluate_email() -> dict:
    """Evaluate confidence for the selected email model."""

    model = load_pickle(
        MODEL_DIR / "email_linear_svm.pkl"
    )

    extractor = load_pickle(
        MODEL_DIR
        / "email_linear_svm_tfidf_extractor.pkl"
    )

    df = pd.read_csv(EMAIL_TEST)

    texts = (
        df["clean_text"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    y_true = df["label"].astype(int).to_numpy()

    X = extractor.transform(texts)

    y_pred = model.predict(X)

    y_probability = model.predict_proba(X)[:, 1]

    confidence = np.maximum(
        y_probability,
        1.0 - y_probability,
    )

    result = analyze_predictions(
        y_true=y_true,
        y_pred=y_pred,
        y_probability=y_probability,
    )

    result.update(
        {
            "model": "email_linear_svm",
            "modality": "email",
            "test_samples": int(len(df)),
        }
    )

    plot_confidence_distribution(
        confidence,
        "Email Linear SVM",
        "email_confidence_distribution.png",
    )

    return result


def evaluate_url() -> dict:
    """Evaluate confidence for the selected URL model."""

    model = load_pickle(
        MODEL_DIR / "url_random_forest.pkl"
    )

    extractor = load_pickle(
        MODEL_DIR
        / "url_random_forest_feature_extractor.pkl"
    )

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

    y_true = df["label"].astype(int).to_numpy()

    X = extractor.transform(X_raw)

    y_pred = model.predict(X)

    y_probability = model.predict_proba(X)[:, 1]

    confidence = np.maximum(
        y_probability,
        1.0 - y_probability,
    )

    result = analyze_predictions(
        y_true=y_true,
        y_pred=y_pred,
        y_probability=y_probability,
    )

    result.update(
        {
            "model": "url_random_forest",
            "modality": "url",
            "test_samples": int(len(df)),
        }
    )

    plot_confidence_distribution(
        confidence,
        "URL Random Forest",
        "url_confidence_distribution.png",
    )

    return result


def main() -> None:
    """Run confidence analysis for both selected models."""

    print("=" * 70)
    print("PHASE 2 — MODEL CONFIDENCE ANALYSIS")
    print("=" * 70)

    email_result = evaluate_email()

    print("\nEmail confidence:")
    print(
        json.dumps(
            email_result["confidence_bands"],
            indent=2,
        )
    )

    url_result = evaluate_url()

    print("\nURL confidence:")
    print(
        json.dumps(
            url_result["confidence_bands"],
            indent=2,
        )
    )

    result = {
        "experiment": "phase2_model_confidence_analysis",
        "confidence_definition": (
            "max(P(legitimate), P(phishing))"
        ),
        "confidence_thresholds": {
            "low": "< 0.60",
            "medium": "0.60 <= confidence < 0.80",
            "high": ">= 0.80",
        },
        "thresholds_are_descriptive": True,
        "models": {
            "email": email_result,
            "url": url_result,
        },
    }

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        OUTPUT_DIR
        / "model_confidence_analysis.json"
    )

    with output_file.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result,
            file,
            indent=2,
        )

    # Create a compact CSV summary.
    rows = []

    for result_item in [
        email_result,
        url_result,
    ]:

        bands = result_item["confidence_bands"]

        rows.append(
            {
                "Modality": result_item["modality"],
                "Model": result_item["model"],
                "Test Samples": result_item[
                    "test_samples"
                ],
                "Mean Confidence": result_item[
                    "confidence_distribution"
                ]["mean"],
                "Median Confidence": result_item[
                    "confidence_distribution"
                ]["median"],
                "Std Confidence": result_item[
                    "confidence_distribution"
                ]["std"],
                "Mean Confidence Correct": result_item[
                    "mean_confidence_correct"
                ],
                "Mean Confidence Incorrect": result_item[
                    "mean_confidence_incorrect"
                ],
                "Low Confidence Rate": bands[
                    "low"
                ]["rate"],
                "Medium Confidence Rate": bands[
                    "medium"
                ]["rate"],
                "High Confidence Rate": bands[
                    "high"
                ]["rate"],
            }
        )

    summary_file = (
        OUTPUT_DIR
        / "model_confidence_summary.csv"
    )

    pd.DataFrame(rows).to_csv(
        summary_file,
        index=False,
    )

    print("\n" + "=" * 70)
    print("CONFIDENCE ANALYSIS COMPLETE")
    print("=" * 70)

    print(f"\nJSON: {output_file}")
    print(f"CSV : {summary_file}")


if __name__ == "__main__":
    main()