"""
Real Phase 4.8 XAI pipeline integration verification.

This script verifies that the Phase 4 XAI integration can consume
real Phase 3 AgentResult objects.

It intentionally uses a controlled verification input.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]

SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


from phishing_detection.agents.evidence import Evidence
from phishing_detection.agents.result import AgentResult
from phishing_detection.explainability.pipeline_integration import (
    ExplainabilityPipelineIntegrator,
)


OUTPUT_DIR = (
    ROOT
    / "experiments"
    / "baselines"
    / "outputs"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def make_email_result() -> AgentResult:

    evidence = [
        Evidence(
            evidence_type="text_pattern",
            source="email_analysis_agent",
            description=(
                "Synthetic verification evidence "
                "indicating urgency and credential request."
            ),
            value=(
                "urgent account verification"
            ),
            polarity="phishing",
            confidence=0.92,
            reliability=0.95,
            metadata={
                "verification": True,
            },
        )
    ]

    return AgentResult.from_probability(
        agent_name="EmailAnalysisAgent",
        modality="email",
        probability=0.93,
        evidence=evidence,
    )


def make_sender_result() -> AgentResult:

    evidence = [
        Evidence(
            evidence_type="sender_rule",
            source="sender_analysis_agent",
            description=(
                "Synthetic verification evidence "
                "for sender/domain inconsistency."
            ),
            value=(
                "sender_domain_mismatch"
            ),
            polarity="phishing",
            confidence=0.88,
            reliability=0.90,
            metadata={
                "verification": True,
            },
        )
    ]

    return AgentResult.from_probability(
        agent_name="SenderAnalysisAgent",
        modality="sender",
        probability=0.87,
        evidence=evidence,
    )


def main() -> None:

    print("=" * 80)
    print(
        "PHASE 4.8 REAL XAI PIPELINE INTEGRATION VERIFICATION"
    )
    print("=" * 80)

    email_result = make_email_result()
    sender_result = make_sender_result()

    agent_results = [
        email_result,
        sender_result,
    ]

    integrator = (
        ExplainabilityPipelineIntegrator()
    )

    print("\n[1] Agent results created")
    print(
        f"    Agents: "
        f"{[r.agent_name for r in agent_results]}"
    )

    print("\n[2] Generating explanations")

    explanations = integrator.explain_agents(
        agent_results
    )

    print(
        f"    Explained agents: "
        f"{list(explanations.keys())}"
    )

    for agent_name, methods in explanations.items():

        print(
            f"    {agent_name}: "
            f"{list(methods.keys())}"
        )

    print("\n[3] Aggregating explanations")

    aggregated = integrator.aggregate(
        explanations
    )

    if aggregated is None:
        raise RuntimeError(
            "No aggregated explanation was produced."
        )

    print(
        "    Aggregated explanation generated"
    )

    print(
        f"    Factor count: "
        f"{aggregated.factor_count}"
    )

    print(
        f"    Explainer: "
        f"{aggregated.explainer_name}"
    )

    print(
        f"    Method: "
        f"{aggregated.method}"
    )

    print("\n[4] Building explainable pipeline result")

    result = integrator.build_result(
        agent_results=agent_results,
        label=1,
        label_name="phishing",
        phishing_probability=0.91,
        confidence=0.91,
        uncertainty=0.09,
        decision_trace=[
            {
                "step": 1,
                "action": "email_analysis",
                "status": "completed",
            },
            {
                "step": 2,
                "action": "sender_analysis",
                "status": "completed",
            },
            {
                "step": 3,
                "action": "evidence_synthesis",
                "status": "completed",
            },
        ],
        metadata={
            "verification": True,
            "phase": "4.8",
        },
    )

    print(
        f"    Final label: "
        f"{result.label_name}"
    )

    print(
        f"    Probability: "
        f"{result.phishing_probability:.4f}"
    )

    print(
        f"    Confidence: "
        f"{result.confidence:.4f}"
    )

    print(
        f"    Uncertainty: "
        f"{result.uncertainty:.4f}"
    )

    print(
        f"    Evidence count: "
        f"{len(result.evidence)}"
    )

    print(
        f"    Explanation factors: "
        f"{result.explanation_factor_count}"
    )

    print(
        f"    XAI available: "
        f"{result.explanation_available}"
    )

    output_path = (
        OUTPUT_DIR
        / "phase4_8_pipeline_xai_verification.json"
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            result.to_dict(),
            file,
            indent=2,
            default=str,
        )

    print(
        f"\n[5] Verification output written to:"
    )

    print(
        f"    {output_path.relative_to(ROOT)}"
    )

    print("\n" + "=" * 80)
    print(
        "PHASE 4.8 REAL VERIFICATION COMPLETE"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()