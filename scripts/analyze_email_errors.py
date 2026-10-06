"""
Phase 2 error analysis for the selected email baseline.

Selected model:
    email_linear_svm

The analysis identifies:
    - False positives
    - False negatives
    - Error rates
    - Confidence/probability distributions
    - Representative error examples

No model retraining is performed.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd


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

TEST_FILE = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "email"
    / "test.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "baselines"
    / "outputs"
)

FP_FILE = OUTPUT_DIR / "email_false_positives.csv"
FN_FILE = OUTPUT_DIR / "email_false_negatives.csv"
JSON_FILE = OUTPUT_DIR / "email_error_analysis.json"


def load_pickle(path: Path):
    """Load a serialized Python artifact."""

    if not path.exists():
        raise FileNotFoundError(
            f"Required artifact not found:\n{path}"
        )

    with path.open("rb") as file:
        return pickle.load(file)


def main() -> None:
    """Run email false-positive/false-negative analysis."""

    print("=" * 70)
    print("PHASE 2 — EMAIL FALSE POSITIVE / FALSE NEGATIVE ANALYSIS")
    print("=" * 70)

    model = load_pickle(MODEL_FILE)
    extractor = load_pickle(TFIDF_FILE)

    if not TEST_FILE.exists():
        raise FileNotFoundError(
            f"Email test set not found:\n{TEST_FILE}"
        )

    df = pd.read_csv(TEST_FILE)

    required_columns = {
        "sample_id",
        "source_dataset",
        "label",
        "clean_text",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Email test set is missing columns: {sorted(missing)}"
        )

    texts = (
        df["clean_text"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    y_true = df["label"].astype(int).to_numpy()

    print(f"\nTest samples: {len(df):,}")
    print("Generating predictions...")

    X = extractor.transform(texts)

    y_pred = model.predict(X)
    y_probability = model.predict_proba(X)[:, 1]

    df["prediction"] = y_pred.astype(int)
    df["phishing_probability"] = y_probability.astype(float)

    df["prediction_name"] = df["prediction"].map(
        {
            0: "legitimate",
            1: "phishing",
        }
    )

    # False positive:
    # Actual legitimate (0), predicted phishing (1)
    fp_mask = (
        (df["label"] == 0)
        & (df["prediction"] == 1)
    )

    # False negative:
    # Actual phishing (1), predicted legitimate (0)
    fn_mask = (
        (df["label"] == 1)
        & (df["prediction"] == 0)
    )

    false_positives = df.loc[fp_mask].copy()
    false_negatives = df.loc[fn_mask].copy()

    # Confidence of the predicted class.
    false_positives["prediction_confidence"] = (
        false_positives["phishing_probability"]
    )

    false_negatives["prediction_confidence"] = (
        1.0 - false_negatives["phishing_probability"]
    )

    # Keep the most useful fields for manual research inspection.
    output_columns = [
        "sample_id",
        "source_dataset",
        "label",
        "prediction",
        "prediction_name",
        "phishing_probability",
        "prediction_confidence",
        "clean_text",
    ]

    false_positives = false_positives[
        output_columns
    ]

    false_negatives = false_negatives[
        output_columns
    ]

    # Sort by confidence so that the strongest mistakes appear first.
    false_positives = false_positives.sort_values(
        "prediction_confidence",
        ascending=False,
    )

    false_negatives = false_negatives.sort_values(
        "prediction_confidence",
        ascending=False,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    false_positives.to_csv(
        FP_FILE,
        index=False,
    )

    false_negatives.to_csv(
        FN_FILE,
        index=False,
    )

    fp_count = len(false_positives)
    fn_count = len(false_negatives)

    total_legitimate = int(
        (df["label"] == 0).sum()
    )

    total_phishing = int(
        (df["label"] == 1).sum()
    )

    false_positive_rate = (
        fp_count / total_legitimate
        if total_legitimate
        else 0.0
    )

    false_negative_rate = (
        fn_count / total_phishing
        if total_phishing
        else 0.0
    )

    # Probability distributions for the errors.
    fp_mean_probability = (
        float(false_positives["phishing_probability"].mean())
        if fp_count
        else None
    )

    fn_mean_probability = (
        float(false_negatives["phishing_probability"].mean())
        if fn_count
        else None
    )

    # Highest-confidence mistakes.
    representative_fp = []

    for _, row in false_positives.head(10).iterrows():
        representative_fp.append(
            {
                "sample_id": row["sample_id"],
                "source_dataset": row["source_dataset"],
                "phishing_probability": float(
                    row["phishing_probability"]
                ),
                "prediction_confidence": float(
                    row["prediction_confidence"]
                ),
                "text_preview": str(
                    row["clean_text"]
                )[:500],
            }
        )

    representative_fn = []

    for _, row in false_negatives.head(10).iterrows():
        representative_fn.append(
            {
                "sample_id": row["sample_id"],
                "source_dataset": row["source_dataset"],
                "phishing_probability": float(
                    row["phishing_probability"]
                ),
                "prediction_confidence": float(
                    row["prediction_confidence"]
                ),
                "text_preview": str(
                    row["clean_text"]
                )[:500],
            }
        )

    result = {
        "experiment": "phase2_email_error_analysis",
        "model": "email_linear_svm",
        "test_samples": int(len(df)),
        "legitimate_samples": total_legitimate,
        "phishing_samples": total_phishing,
        "false_positive_count": fp_count,
        "false_negative_count": fn_count,
        "false_positive_rate": float(
            false_positive_rate
        ),
        "false_negative_rate": float(
            false_negative_rate
        ),
        "false_positive_mean_phishing_probability": (
            fp_mean_probability
        ),
        "false_negative_mean_phishing_probability": (
            fn_mean_probability
        ),
        "representative_false_positives": representative_fp,
        "representative_false_negatives": representative_fn,
    }

    with JSON_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result,
            file,
            indent=2,
        )

    print("\nERROR SUMMARY")
    print("-" * 70)
    print(f"False positives : {fp_count:,}")
    print(f"False negatives : {fn_count:,}")
    print(
        f"False positive rate: "
        f"{false_positive_rate:.4f}"
    )
    print(
        f"False negative rate: "
        f"{false_negative_rate:.4f}"
    )

    print("\nOUTPUTS")
    print("-" * 70)
    print(FP_FILE)
    print(FN_FILE)
    print(JSON_FILE)


if __name__ == "__main__":
    main()