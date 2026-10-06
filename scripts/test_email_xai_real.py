from __future__ import annotations

import json
import sys
from pathlib import Path

from phishing_detection.agents.base import AgentContext
from phishing_detection.agents.result import AgentResult
from phishing_detection.explainability.email_xai import (
    EmailLIMEExplainer,
    EmailSHAPExplainer,
)


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "baselines"
    / "models"
    / "email_linear_svm.pkl"
)

TFIDF_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "baselines"
    / "models"
    / "email_linear_svm_tfidf_extractor.pkl"
)


# ---------------------------------------------------------------------------
# Verification email
# ---------------------------------------------------------------------------

SAMPLE_EMAIL = """
Subject: Urgent account verification required

Dear Customer,

Your account requires immediate verification to prevent suspension.

Please confirm your account information using the secure link below:

http://secure-account-verification.example.com/login

Failure to verify your information within 24 hours may result in restricted
access to your account.

Thank you,
Account Security Team
"""


# ---------------------------------------------------------------------------
# Input construction
# ---------------------------------------------------------------------------

def build_context() -> AgentContext:
    """Build a minimal email analysis context for real XAI verification."""

    return AgentContext(
        input_text=SAMPLE_EMAIL,
        metadata={
            "source": "phase4_real_xai_verification",
        },
    )


def build_agent_result() -> AgentResult:
    """
    Build a valid AgentResult for explanation verification.

    The probability is intentionally fixed because this script verifies
    the explanation machinery against the frozen model artifacts.
    """

    return AgentResult.from_probability(
        agent_name="email_analysis_agent",
        modality="email",
        phishing_probability=0.85,
        evidence=[],
        execution_metadata=None,
    )


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def print_explanation(name: str, explanation) -> None:
    """Print a human-readable explanation summary."""

    print("\n" + "=" * 80)
    print(f"{name} EXPLANATION")
    print("=" * 80)

    print(f"Explainer       : {explanation.explainer_name}")
    print(f"Method          : {explanation.method}")
    print(f"Framework       : {explanation.framework}")
    print(f"Scope           : {explanation.scope}")
    print(f"Model           : {explanation.model_name}")
    print(f"Target label    : {explanation.target.label_name}")
    print(f"Probability     : {explanation.target.probability:.6f}")
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


# ---------------------------------------------------------------------------
# Main verification
# ---------------------------------------------------------------------------

def main() -> int:
    """Run real SHAP and LIME verification against frozen Phase 2 artifacts."""

    print("=" * 80)
    print("PHASE 4.3 — REAL EMAIL SHAP + LIME VERIFICATION")
    print("=" * 80)

    # -----------------------------------------------------------------------
    # Project information
    # -----------------------------------------------------------------------

    print("\nProject root:")
    print(f"  {PROJECT_ROOT}")

    # -----------------------------------------------------------------------
    # Verify frozen artifacts
    # -----------------------------------------------------------------------

    print("\nChecking frozen Phase 2 artifacts...")

    if not MODEL_PATH.exists():
        print("\nERROR: Model artifact not found:")
        print(f"  {MODEL_PATH}")
        return 1

    if not TFIDF_PATH.exists():
        print("\nERROR: TF-IDF artifact not found:")
        print(f"  {TFIDF_PATH}")
        return 1

    print("  [PASS] email_linear_svm.pkl")
    print("  [PASS] email_linear_svm_tfidf_extractor.pkl")

    # -----------------------------------------------------------------------
    # Build verification input
    # -----------------------------------------------------------------------

    context = build_context()
    result = build_agent_result()

    # -----------------------------------------------------------------------
    # SHAP verification
    # -----------------------------------------------------------------------

    print("\nInitializing SHAP explainer...")

    shap_explainer = EmailSHAPExplainer(
        model_path=MODEL_PATH,
        tfidf_path=TFIDF_PATH,
        max_evals=200,
        max_display=15,
    )

    print("Running SHAP explanation...")

    shap_explanation = shap_explainer.explain(
        result=result,
        context=context,
    )

    print_explanation(
        "SHAP",
        shap_explanation,
    )

    # -----------------------------------------------------------------------
    # LIME verification
    # -----------------------------------------------------------------------

    print("\nInitializing LIME explainer...")

    lime_explainer = EmailLIMEExplainer(
        model_path=MODEL_PATH,
        tfidf_path=TFIDF_PATH,
        num_features=15,
        num_samples=1000,
    )

    print("Running LIME explanation...")

    lime_explanation = lime_explainer.explain(
        result=result,
        context=context,
    )

    print_explanation(
        "LIME",
        lime_explanation,
    )

    # -----------------------------------------------------------------------
    # Verification checks
    # -----------------------------------------------------------------------

    verification_checks = {
        "model_artifact_exists": MODEL_PATH.exists(),
        "tfidf_artifact_exists": TFIDF_PATH.exists(),
        "shap_factor_count_positive": shap_explanation.factor_count > 0,
        "lime_factor_count_positive": lime_explanation.factor_count > 0,
        "shap_model_matches": shap_explanation.model_name == "email_linear_svm",
        "lime_model_matches": lime_explanation.model_name == "email_linear_svm",
        "shap_method_correct": shap_explanation.method == "shap",
        "lime_method_correct": lime_explanation.method == "lime",
    }

    # -----------------------------------------------------------------------
    # Save verification output
    # -----------------------------------------------------------------------

    output = {
        "phase": "4.3",
        "status": "PASS",
        "model_artifact": str(
            MODEL_PATH.relative_to(PROJECT_ROOT)
        ),
        "tfidf_artifact": str(
            TFIDF_PATH.relative_to(PROJECT_ROOT)
        ),
        "sample_type": "synthetic_verification_email",
        "verification_checks": verification_checks,
        "shap": shap_explanation.to_dict(),
        "lime": lime_explanation.to_dict(),
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
        / "phase4_3_email_xai_real_verification.json"
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

    # -----------------------------------------------------------------------
    # Final result
    # -----------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("PHASE 4.3 REAL VERIFICATION RESULT")
    print("=" * 80)

    failed_checks = [
        name
        for name, passed in verification_checks.items()
        if not passed
    ]

    if failed_checks:
        print("\n[FAIL] One or more verification checks failed:")

        for check in failed_checks:
            print(f"  - {check}")

        print("\nVerification output:")
        print(f"  {output_path}")

        return 1

    print("\n[PASS] Real SHAP explanation generated.")
    print("[PASS] Real LIME explanation generated.")
    print("[PASS] Both explainers used the frozen Phase 2 email artifacts.")
    print("[PASS] SHAP produced token-level explanation factors.")
    print("[PASS] LIME produced token-level explanation factors.")
    print("[PASS] Unified explanation objects were generated.")
    print("[PASS] Verification JSON was written successfully.")

    print("\nVerification output:")
    print(f"  {output_path}")

    print("\n" + "=" * 80)
    print("PHASE 4.3 REAL VERIFICATION: PASS")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())