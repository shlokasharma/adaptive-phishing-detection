from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from phishing_detection.agents.base import AgentContext
from phishing_detection.agents.result import AgentResult
from phishing_detection.explainability.url_xai import (
    URLLIMEExplainer,
    URLSHAPExplainer,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "baselines"
    / "models"
    / "url_random_forest.pkl"
)

FEATURE_EXTRACTOR_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "baselines"
    / "models"
    / "url_random_forest_feature_extractor.pkl"
)

BACKGROUND_PATH = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "url"
    / "uci_features"
    / "train.csv"
)


def load_verification_sample():
    """Load one frozen UCI URL test sample."""

    test_path = (
        PROJECT_ROOT
        / "data"
        / "splits"
        / "url"
        / "uci_features"
        / "test.csv"
    )

    if not test_path.exists():
        raise FileNotFoundError(
            f"URL test split not found: {test_path}"
        )

    frame = pd.read_csv(test_path)

    excluded = {
        "sample_id",
        "source_dataset",
        "label",
    }

    feature_columns = [
        column
        for column in frame.columns
        if column not in excluded
    ]

    row = frame.iloc[0]

    features = {
        column: row[column]
        for column in feature_columns
    }

    url = (
        row["url"]
        if "url" in row.index
        else "https://example.com"
    )

    return url, features


def build_result():
    """Create a verification AgentResult."""

    return AgentResult.from_probability(
        agent_name="url_analysis_agent",
        modality="url",
        phishing_probability=0.50,
        evidence=[],
        execution_metadata=None,
    )


def print_explanation(name, explanation):
    """Print a readable XAI explanation."""

    print("\n" + "=" * 80)
    print(f"{name} EXPLANATION")
    print("=" * 80)

    print(f"Explainer       : {explanation.explainer_name}")
    print(f"Method          : {explanation.method}")
    print(f"Framework       : {explanation.framework}")
    print(f"Scope           : {explanation.scope}")
    print(f"Model           : {explanation.model_name}")
    print(f"Target label    : {explanation.target.label_name}")
    print(
        f"Probability     : "
        f"{explanation.target.probability:.6f}"
    )
    print(f"Factor count    : {explanation.factor_count}")

    print("\nTop explanation factors:")

    for factor in explanation.factors[:10]:
        print(
            f"  {factor.feature!r:45s}"
            f" contribution={factor.contribution:+.6f}"
            f" importance={factor.importance:.6f}"
            f" direction={factor.direction.value}"
        )

    print("\nSummary:")
    print(f"  {explanation.summary}")


def main() -> int:

    print("=" * 80)
    print("PHASE 4.4 — REAL URL SHAP + LIME VERIFICATION")
    print("=" * 80)

    print("\nProject root:")
    print(f"  {PROJECT_ROOT}")

    print("\nChecking frozen Phase 2 artifacts...")

    required_artifacts = [
        (
            MODEL_PATH,
            "url_random_forest.pkl",
        ),
        (
            FEATURE_EXTRACTOR_PATH,
            "url_random_forest_feature_extractor.pkl",
        ),
        (
            BACKGROUND_PATH,
            "uci_features/train.csv",
        ),
    ]

    for path, name in required_artifacts:

        if not path.exists():

            print(
                f"\n[FAIL] {name} not found:"
            )

            print(
                f"  {path}"
            )

            return 1

        print(
            f"  [PASS] {name}"
        )

    print(
        "\nLoading frozen URL verification sample..."
    )

    url, url_features = load_verification_sample()

    print(
        f"  URL: {url}"
    )

    print(
        f"  Feature count: {len(url_features)}"
    )

    context = AgentContext(
        input_text=url,
        url=url,
        metadata={
            "url_features": url_features,
            "source": "phase4_real_xai_verification",
        },
    )

    result = build_result()

    # ------------------------------------------------------------------
    # SHAP
    # ------------------------------------------------------------------

    print(
        "\nInitializing SHAP explainer..."
    )

    print(
        "Running SHAP explanation..."
    )

    shap_explainer = URLSHAPExplainer(
        model_path=MODEL_PATH,
        feature_extractor_path=FEATURE_EXTRACTOR_PATH,
        max_display=15,
    )

    shap_explanation = shap_explainer.explain(
        result=result,
        context=context,
    )

    print_explanation(
        "SHAP",
        shap_explanation,
    )

    # ------------------------------------------------------------------
    # LIME
    # ------------------------------------------------------------------

    print(
        "\nInitializing LIME explainer..."
    )

    print(
        "Running LIME explanation..."
    )

    lime_explainer = URLLIMEExplainer(
        model_path=MODEL_PATH,
        feature_extractor_path=FEATURE_EXTRACTOR_PATH,
        background_path=BACKGROUND_PATH,
        num_features=15,
        num_samples=1000,
        background_size=100,
    )

    lime_explanation = lime_explainer.explain(
        result=result,
        context=context,
    )

    print_explanation(
        "LIME",
        lime_explanation,
    )

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    verification_checks = {
        "model_artifact_exists": MODEL_PATH.exists(),
        "feature_extractor_exists": (
            FEATURE_EXTRACTOR_PATH.exists()
        ),
        "background_exists": BACKGROUND_PATH.exists(),
        "shap_factor_count_positive": (
            shap_explanation.factor_count > 0
        ),
        "lime_factor_count_positive": (
            lime_explanation.factor_count > 0
        ),
        "shap_model_matches": (
            shap_explanation.model_name
            == "url_random_forest"
        ),
        "lime_model_matches": (
            lime_explanation.model_name
            == "url_random_forest"
        ),
        "shap_method_correct": (
            shap_explanation.method
            == "shap"
        ),
        "lime_method_correct": (
            lime_explanation.method
            == "lime"
        ),
    }

    failed_checks = [
        name
        for name, passed in verification_checks.items()
        if not passed
    ]

    status = (
        "PASS"
        if not failed_checks
        else "FAIL"
    )

    output = {
        "phase": "4.4",
        "status": status,
        "model_artifact": str(
            MODEL_PATH.relative_to(
                PROJECT_ROOT
            )
        ),
        "feature_extractor_artifact": str(
            FEATURE_EXTRACTOR_PATH.relative_to(
                PROJECT_ROOT
            )
        ),
        "background_dataset": str(
            BACKGROUND_PATH.relative_to(
                PROJECT_ROOT
            )
        ),
        "sample_type": (
            "frozen_uci_url_test_sample"
        ),
        "verification_checks": (
            verification_checks
        ),
        "shap": (
            shap_explanation.to_dict()
        ),
        "lime": (
            lime_explanation.to_dict()
        ),
    }

    output_dir = (
        PROJECT_ROOT
        / "experiments"
        / "baselines"
        / "outputs"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / "phase4_4_url_xai_real_verification.json"
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False,
        )

    # ------------------------------------------------------------------
    # Final result
    # ------------------------------------------------------------------

    print(
        "\n" + "=" * 80
    )

    print(
        "PHASE 4.4 REAL VERIFICATION RESULT"
    )

    print(
        "=" * 80
    )

    if failed_checks:

        print(
            "\n[FAIL] Verification checks failed:"
        )

        for check in failed_checks:
            print(
                f"  - {check}"
            )

        print(
            "\nVerification output:"
        )

        print(
            f"  {output_path}"
        )

        return 1

    print(
        "\n[PASS] Real SHAP explanation generated."
    )

    print(
        "[PASS] Real LIME explanation generated."
    )

    print(
        "[PASS] Frozen URL Random Forest was used."
    )

    print(
        "[PASS] Frozen URL feature extractor was used."
    )

    print(
        "[PASS] SHAP produced URL feature attributions."
    )

    print(
        "[PASS] LIME produced URL feature attributions."
    )

    print(
        "[PASS] Unified explanation objects were generated."
    )

    print(
        "[PASS] Verification JSON was written successfully."
    )

    print(
        "\nVerification output:"
    )

    print(
        f"  {output_path}"
    )

    print(
        "\n" + "=" * 80
    )

    print(
        "PHASE 4.4 REAL VERIFICATION: PASS"
    )

    print(
        "=" * 80
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())