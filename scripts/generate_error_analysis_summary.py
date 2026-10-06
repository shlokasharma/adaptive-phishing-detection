"""
Generate a combined Phase 2 false-positive/false-negative summary.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "baselines"
    / "outputs"
)

EMAIL_FILE = OUTPUT_DIR / "email_error_analysis.json"
URL_FILE = OUTPUT_DIR / "url_error_analysis.json"

OUTPUT_FILE = (
    OUTPUT_DIR / "error_analysis_summary.csv"
)


def load_json(path: Path) -> dict:
    """Load a JSON file."""

    if not path.exists():
        raise FileNotFoundError(
            f"Required error-analysis file not found:\n{path}"
        )

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def main() -> None:
    """Generate combined error-analysis table."""

    print("=" * 70)
    print("PHASE 2 — ERROR ANALYSIS SUMMARY")
    print("=" * 70)

    email = load_json(EMAIL_FILE)
    url = load_json(URL_FILE)

    rows = []

    for result in [email, url]:
        rows.append(
            {
                "Modality": (
                    "Email"
                    if result["experiment"]
                    == "phase2_email_error_analysis"
                    else "URL"
                ),
                "Model": result["model"],
                "Test Samples": result["test_samples"],
                "False Positives": result[
                    "false_positive_count"
                ],
                "False Negatives": result[
                    "false_negative_count"
                ],
                "False Positive Rate": result[
                    "false_positive_rate"
                ],
                "False Negative Rate": result[
                    "false_negative_rate"
                ],
                "FP Mean Phishing Probability": result[
                    "false_positive_mean_phishing_probability"
                ],
                "FN Mean Phishing Probability": result[
                    "false_negative_mean_phishing_probability"
                ],
            }
        )

    df = pd.DataFrame(rows)

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print(df.to_string(index=False))

    print()
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()