"""
Tests for Phase 4.8 pipeline XAI integration.
"""

from __future__ import annotations

import time

from phishing_detection.agents.evidence import Evidence
from phishing_detection.agents.result import AgentResult
from phishing_detection.explainability.pipeline_integration import (
    ExplainabilityPipelineIntegrator,
    ExplainablePipelineResult,
)
from phishing_detection.explainability.schema import (
    ExplanationDirection,
    ExplanationFactor,
    ExplanationScope,
    ExplanationTarget,
    UnifiedExplanation,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def make_evidence(
    source: str = "EmailAnalysisAgent",
    polarity: str = "supports_phishing",
) -> Evidence:
    return Evidence(
        evidence_type="test",
        source=source,
        description="Test evidence",
        value="test",
        polarity=polarity,
        confidence=0.9,
        reliability=0.9,
        metadata={},
    )


def make_agent_result(
    agent_name: str = "EmailAnalysisAgent",
    probability: float = 0.9,
) -> AgentResult:
    evidence = make_evidence(
        source=agent_name,
        polarity="supports_phishing",
    )

    return AgentResult.from_probability(
        agent_name=agent_name,
        modality="test",
        phishing_probability=probability,
        evidence=[evidence],
        metadata={
            "test": True,
            "agent_name": agent_name,
        },
    )


def make_explanation(
    explainer_name: str = "test",
    method: str = "test",
    feature: str = "test_feature",
    direction: ExplanationDirection = (
        ExplanationDirection.SUPPORTS_PHISHING
    ),
    importance: float = 0.8,
    probability: float = 0.9,
) -> UnifiedExplanation:

    factor = ExplanationFactor(
        feature=feature,
        value="test_value",
        importance=importance,
        direction=direction,
        rank=1,
        contribution=importance,
        metadata={},
    )

    target = ExplanationTarget(
        label=1,
        label_name="phishing",
        probability=probability,
    )

    return UnifiedExplanation(
        summary=f"Test explanation for {feature}.",
        explainer_name=explainer_name,
        method=method,
        framework="test",
        scope=ExplanationScope.LOCAL,
        target=target,
        factors=[factor],
        model_name="test_model",
        execution_time_ms=1.0,
        metadata={
            "test": True,
        },
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_integrator_can_be_constructed():
    integrator = ExplainabilityPipelineIntegrator()

    assert integrator is not None
    assert integrator.email_explainer is not None
    assert integrator.url_explainer is not None
    assert integrator.sender_explainer is not None
    assert integrator.aggregator is not None


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def test_normalize_single_explanation():
    explanation = make_explanation(
        explainer_name="test",
        method="test",
        feature="url_length",
        direction=ExplanationDirection.SUPPORTS_PHISHING,
        importance=0.8,
    )

    result = (
        ExplainabilityPipelineIntegrator
        ._normalize_explanation_result(explanation)
    )

    assert isinstance(result, dict)
    assert len(result) == 1
    assert all(
        isinstance(value, UnifiedExplanation)
        for value in result.values()
    )


def test_normalize_dictionary():
    shap = make_explanation(
        explainer_name="shap",
        method="shap",
        feature="urgency",
        direction=ExplanationDirection.SUPPORTS_PHISHING,
        importance=0.8,
    )

    lime = make_explanation(
        explainer_name="lime",
        method="lime",
        feature="credential",
        direction=ExplanationDirection.SUPPORTS_PHISHING,
        importance=0.7,
    )

    result = (
        ExplainabilityPipelineIntegrator
        ._normalize_explanation_result(
            {
                "shap": shap,
                "lime": lime,
            }
        )
    )

    assert set(result.keys()) == {"shap", "lime"}
    assert result["shap"] is shap
    assert result["lime"] is lime


def test_normalize_none():
    result = (
        ExplainabilityPipelineIntegrator
        ._normalize_explanation_result(None)
    )

    assert result == {}


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def test_aggregate_explanations():
    integrator = ExplainabilityPipelineIntegrator()

    shap = make_explanation(
        explainer_name="email_shap",
        method="shap",
        feature="urgency",
        direction=ExplanationDirection.SUPPORTS_PHISHING,
        importance=0.8,
    )

    lime = make_explanation(
        explainer_name="email_lime",
        method="lime",
        feature="urgency",
        direction=ExplanationDirection.SUPPORTS_PHISHING,
        importance=0.6,
    )

    aggregated = integrator.aggregate(
        {
            "EmailAnalysisAgent": {
                "shap": shap,
                "lime": lime,
            }
        }
    )

    assert aggregated is not None
    assert isinstance(
        aggregated,
        UnifiedExplanation,
    )

    assert aggregated.explainer_name == (
        "multi_agent_explanation_aggregator"
    )

    assert aggregated.factor_count >= 1


# ---------------------------------------------------------------------------
# Result construction
# ---------------------------------------------------------------------------


def test_build_result():
    integrator = ExplainabilityPipelineIntegrator()

    agent_result = make_agent_result(
        "EmailAnalysisAgent",
        probability=0.91,
    )

    explanation = make_explanation(
        explainer_name="email_shap",
        method="shap",
        feature="urgency",
        probability=0.91,
    )

    result = integrator.build_result(
        agent_results=[agent_result],
        explanations={
            "EmailAnalysisAgent": {
                "shap": explanation,
            }
        },
    )

    assert isinstance(
        result,
        ExplainablePipelineResult,
    )

    assert result.final_agent_result is agent_result
    assert result.phishing_probability == 0.91
    assert result.aggregated_explanation is not None
    assert result.decision is not None


def test_build_result_contains_agent_evidence():
    integrator = ExplainabilityPipelineIntegrator()

    agent_result = make_agent_result(
        "EmailAnalysisAgent"
    )

    result = integrator.build_result(
        agent_results=[agent_result]
    )

    assert len(result.agent_results) == 1

    stored_result = result.agent_results[0]

    assert len(stored_result.evidence) == 1
    assert stored_result.evidence[0].source == (
        "EmailAnalysisAgent"
    )


def test_result_to_dict():
    integrator = ExplainabilityPipelineIntegrator()

    agent_result = make_agent_result(
        "EmailAnalysisAgent"
    )

    explanation = make_explanation(
        explainer_name="email_shap",
        method="shap",
    )

    result = integrator.build_result(
        agent_results=[agent_result],
        explanations={
            "EmailAnalysisAgent": {
                "shap": explanation,
            }
        },
        decision_trace=[
            {
                "step": 1,
                "agent": "EmailAnalysisAgent",
            }
        ],
    )

    data = result.to_dict()

    assert isinstance(data, dict)

    assert "agent_results" in data
    assert "explanations" in data
    assert "aggregated_explanation" in data
    assert "final_agent_result" in data
    assert "decision" in data
    assert "confidence" in data
    assert "phishing_probability" in data
    assert "decision_trace" in data
    assert "metadata" in data


# ---------------------------------------------------------------------------
# Explanation availability
# ---------------------------------------------------------------------------


def test_result_explanation_available():
    integrator = ExplainabilityPipelineIntegrator()

    agent_result = make_agent_result(
        "EmailAnalysisAgent"
    )

    explanation = make_explanation(
        explainer_name="email_shap",
        method="shap",
    )

    result = integrator.build_result(
        agent_results=[agent_result],
        explanations={
            "EmailAnalysisAgent": {
                "shap": explanation,
            }
        },
    )

    assert result.explanation is not None
    assert result.final_explanation is not None
    assert result.is_explainable is True


def test_result_without_explanation():
    integrator = ExplainabilityPipelineIntegrator()

    agent_result = make_agent_result(
        "EmailAnalysisAgent"
    )

    result = integrator.build_result(
        agent_results=[agent_result]
    )

    assert result.explanation is None
    assert result.final_explanation is None
    assert result.is_explainable is False


# ---------------------------------------------------------------------------
# Unknown agents
# ---------------------------------------------------------------------------


def test_unknown_agent_is_ignored():
    integrator = ExplainabilityPipelineIntegrator()

    unknown = make_agent_result(
        "UnknownAgent"
    )

    explanations = integrator.explain_agents(
        [unknown]
    )

    assert explanations == {}


# ---------------------------------------------------------------------------
# Multiple agents
# ---------------------------------------------------------------------------


def test_multiple_agents_are_supported():
    integrator = ExplainabilityPipelineIntegrator()

    email = make_agent_result(
        "EmailAnalysisAgent",
        0.90,
    )

    url = make_agent_result(
        "URLAnalysisAgent",
        0.80,
    )

    sender = make_agent_result(
        "SenderAnalysisAgent",
        0.70,
    )

    result = integrator.build_result(
        agent_results=[
            email,
            url,
            sender,
        ]
    )

    assert len(result.agent_results) == 3

    assert result.final_agent_result is sender


# ---------------------------------------------------------------------------
# Execution metadata
# ---------------------------------------------------------------------------


def test_execution_time_is_recorded():
    integrator = ExplainabilityPipelineIntegrator()

    agent_result = make_agent_result(
        "EmailAnalysisAgent"
    )

    start = time.perf_counter()

    result = integrator.build_result(
        agent_results=[agent_result]
    )

    elapsed = (
        time.perf_counter() - start
    )

    result.metadata[
        "test_elapsed_seconds"
    ] = elapsed

    assert elapsed >= 0.0
    assert "phase" in result.metadata
    assert result.metadata["phase"] == "4.8"


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def test_metadata_preserved():
    integrator = ExplainabilityPipelineIntegrator()

    agent_result = make_agent_result(
        "EmailAnalysisAgent"
    )

    result = integrator.build_result(
        agent_results=[agent_result],
        metadata={
            "experiment": "phase4_8_test",
            "custom_value": 123,
        },
    )

    assert result.metadata["experiment"] == (
        "phase4_8_test"
    )

    assert result.metadata["custom_value"] == 123

    assert result.metadata["phase"] == "4.8"


# ---------------------------------------------------------------------------
# Decision trace
# ---------------------------------------------------------------------------


def test_decision_trace_is_preserved():
    integrator = ExplainabilityPipelineIntegrator()

    agent_result = make_agent_result(
        "EmailAnalysisAgent"
    )

    trace = [
        {
            "step": 1,
            "agent": "EmailAnalysisAgent",
            "reason": "initial analysis",
        },
        {
            "step": 2,
            "agent": "SenderAnalysisAgent",
            "reason": "additional evidence",
        },
    ]

    result = integrator.build_result(
        agent_results=[agent_result],
        decision_trace=trace,
    )

    assert result.decision_trace == trace
    assert result.decision_trace[0]["step"] == 1
    assert result.decision_trace[1]["step"] == 2


# ---------------------------------------------------------------------------
# Agent-name normalization
# ---------------------------------------------------------------------------


def test_agent_name_normalization():
    assert (
        ExplainabilityPipelineIntegrator
        ._normalize_agent_name(
            "EmailAnalysisAgent"
        )
        == "emailanalysisagent"
    )

    assert (
        ExplainabilityPipelineIntegrator
        ._normalize_agent_name(
            "email_analysis_agent"
        )
        == "email_analysis_agent"
    )


def test_agent_modality_detection():
    assert (
        ExplainabilityPipelineIntegrator
        ._agent_modality(
            "EmailAnalysisAgent"
        )
        == "email"
    )

    assert (
        ExplainabilityPipelineIntegrator
        ._agent_modality(
            "URLAnalysisAgent"
        )
        == "url"
    )

    assert (
        ExplainabilityPipelineIntegrator
        ._agent_modality(
            "SenderAnalysisAgent"
        )
        == "sender"
    )

    assert (
        ExplainabilityPipelineIntegrator
        ._agent_modality(
            "UnknownAgent"
        )
        is None
    )