"""
Phase 5: Comprehensive tests for AdaptiveEvidencePolicy.

These tests use lightweight mock AgentResult objects so that
no real machine-learning models are required.
"""

from __future__ import annotations

import json
import pytest
import numpy as np
import joblib

from phishing_detection.agents.base import AgentContext
from phishing_detection.agents.email_agent import EmailAnalysisAgent
from phishing_detection.agents.evidence import Evidence
from phishing_detection.agents.result import AgentResult
from phishing_detection.agents.sender_agent import SenderAnalysisAgent
from phishing_detection.agents.url_agent import URLAnalysisAgent
from phishing_detection.orchestration.adaptive_policy import (
    AdaptiveEvidencePolicy,
    EvidencePolicyAssessment,
)
from phishing_detection.orchestration.orchestrator import (
    OrchestratorAgent,
)


# ============================================================
# MODULE-LEVEL MOCK CLASSES FOR JOBLIB SERIALIZATION
# ============================================================


class FakeExtractor:
    def transform(self, texts):
        return np.array([[1.0, 2.0]])


class HighConfModel:
    def predict(self, X):
        return np.array([1])

    def predict_proba(self, X):
        return np.array([[0.05, 0.95]])


class LowConfModel:
    def predict(self, X):
        return np.array([1])

    def predict_proba(self, X):
        return np.array([[0.45, 0.55]])


class MedConfModel:
    def predict(self, X):
        return np.array([1])

    def predict_proba(self, X):
        return np.array([[0.30, 0.70]])


class ConfidentLegitimateModel:
    def predict(self, X):
        return np.array([0])

    def predict_proba(self, X):
        return np.array([[0.95, 0.05]])


class CrashingModel:
    def predict(self, X):
        raise RuntimeError("Simulated model failure")

    def predict_proba(self, X):
        raise RuntimeError("Simulated model failure")


class FakePreprocessor:
    def transform(self, X):
        return X


# ============================================================
# HELPERS
# ============================================================


def make_evidence(
    polarity: str = "supports_phishing",
    confidence: float = 0.8,
    reliability: float = 0.9,
) -> Evidence:
    """Create a minimal Evidence object."""

    return Evidence(
        evidence_type="test_evidence",
        source="test_agent",
        description="Test evidence item.",
        polarity=polarity,
        confidence=confidence,
        reliability=reliability,
    )


def make_result(
    phishing_probability: float,
    evidence: list[Evidence] | None = None,
    agent_name: str = "test_agent",
) -> AgentResult:
    """Create an AgentResult from a probability."""

    return AgentResult.from_probability(
        agent_name=agent_name,
        modality="test",
        phishing_probability=phishing_probability,
        evidence=evidence or [],
    )


# ============================================================
# CONSTRUCTOR VALIDATION
# ============================================================


def test_policy_default_construction():
    """Default policy constructs without error."""

    policy = AdaptiveEvidencePolicy()

    assert policy.confidence_threshold == 0.90
    assert policy.min_evidence_count == 1


def test_policy_custom_threshold():
    """Custom threshold is accepted."""

    policy = AdaptiveEvidencePolicy(
        confidence_threshold=0.75,
        min_evidence_count=2,
    )

    assert policy.confidence_threshold == 0.75
    assert policy.min_evidence_count == 2


def test_invalid_threshold_raises():
    """Threshold above 1.0 raises ValueError."""

    with pytest.raises(ValueError):
        AdaptiveEvidencePolicy(confidence_threshold=1.5)


def test_invalid_threshold_negative_raises():
    """Threshold below 0.0 raises ValueError."""

    with pytest.raises(ValueError):
        AdaptiveEvidencePolicy(confidence_threshold=-0.1)


def test_invalid_min_evidence_negative_raises():
    """Negative min_evidence_count raises ValueError."""

    with pytest.raises(ValueError):
        AdaptiveEvidencePolicy(min_evidence_count=-1)


# ============================================================
# EMPTY INPUT
# ============================================================


def test_empty_results_returns_safe_assessment():
    """No agent results produces a safe, non-crashing assessment."""

    policy = AdaptiveEvidencePolicy()
    assessment = policy.assess([])

    assert isinstance(assessment, EvidencePolicyAssessment)
    assert assessment.phishing_probability == 0.5
    assert assessment.confidence == 0.5
    assert assessment.uncertainty == 0.5
    assert assessment.evidence_strength == 0.0
    assert assessment.evidence_count == 0
    assert assessment.agent_count == 0
    assert assessment.sufficient is False
    assert isinstance(assessment.reason, str)
    assert len(assessment.reason) > 0


# ============================================================
# COMBINED PROBABILITY — ARITHMETIC MEAN
# ============================================================


def test_single_result_probability_is_unchanged():
    """Combined probability for a single result equals that result's probability."""

    policy = AdaptiveEvidencePolicy(min_evidence_count=0)
    result = make_result(0.80)
    assessment = policy.assess([result])

    assert assessment.phishing_probability == pytest.approx(0.80)
    assert assessment.agent_count == 1


def test_two_results_probability_is_mean():
    """Combined probability is the arithmetic mean of two results."""

    policy = AdaptiveEvidencePolicy(min_evidence_count=0)
    r1 = make_result(0.80)
    r2 = make_result(0.60)

    assessment = policy.assess([r1, r2])

    expected = (0.80 + 0.60) / 2
    assert assessment.phishing_probability == pytest.approx(expected)


def test_three_results_probability_is_mean():
    """Combined probability is the arithmetic mean of three results."""

    policy = AdaptiveEvidencePolicy(min_evidence_count=0)
    r1 = make_result(0.90)
    r2 = make_result(0.50)
    r3 = make_result(0.70)

    assessment = policy.assess([r1, r2, r3])

    expected = (0.90 + 0.50 + 0.70) / 3
    assert assessment.phishing_probability == pytest.approx(expected)


# ============================================================
# CONFIDENCE AND UNCERTAINTY
# ============================================================


def test_high_probability_gives_high_confidence():
    """High phishing probability -> confidence = phishing_probability."""

    policy = AdaptiveEvidencePolicy(min_evidence_count=0)
    result = make_result(0.95)
    assessment = policy.assess([result])

    assert assessment.confidence == pytest.approx(0.95)
    assert assessment.uncertainty == pytest.approx(1.0 - 0.95)


def test_low_probability_gives_high_confidence():
    """Low phishing probability -> confidence = 1 - phishing_probability."""

    policy = AdaptiveEvidencePolicy(min_evidence_count=0)
    result = make_result(0.05)
    assessment = policy.assess([result])

    assert assessment.confidence == pytest.approx(0.95)
    assert assessment.uncertainty == pytest.approx(0.05)


def test_uncertain_probability_gives_low_confidence():
    """Probability near 0.5 -> low confidence."""

    policy = AdaptiveEvidencePolicy(min_evidence_count=0)
    result = make_result(0.55)
    assessment = policy.assess([result])

    assert assessment.confidence == pytest.approx(0.55)
    assert assessment.uncertainty == pytest.approx(0.45, abs=1e-9)


# ============================================================
# EVIDENCE COUNT AND STRENGTH
# ============================================================


def test_evidence_count_aggregated_across_results():
    """Evidence count spans all evidence items from all results."""

    policy = AdaptiveEvidencePolicy()
    ev1 = make_evidence()
    ev2 = make_evidence()
    r1 = make_result(0.90, evidence=[ev1])
    r2 = make_result(0.80, evidence=[ev2])

    assessment = policy.assess([r1, r2])

    assert assessment.evidence_count == 2


def test_zero_evidence_across_results():
    """Evidence count is zero when agents produce no evidence."""

    policy = AdaptiveEvidencePolicy(min_evidence_count=0)
    r1 = make_result(0.90, evidence=[])
    assessment = policy.assess([r1])

    assert assessment.evidence_count == 0
    assert assessment.evidence_strength == 0.0


def test_evidence_strength_nonzero_when_evidence_exists():
    """Evidence strength > 0 when evidence has confidence and reliability."""

    policy = AdaptiveEvidencePolicy(min_evidence_count=0)
    ev = make_evidence(confidence=0.8, reliability=0.9)
    r = make_result(0.90, evidence=[ev])
    assessment = policy.assess([r])

    assert assessment.evidence_strength > 0.0


# ============================================================
# SUFFICIENCY DECISIONS & EDGE CASES
# ============================================================


def test_sufficient_when_confidence_and_evidence_meet_thresholds():
    """Assessment is sufficient when both criteria are satisfied."""

    policy = AdaptiveEvidencePolicy(
        confidence_threshold=0.80,
        min_evidence_count=1,
    )
    ev = make_evidence()
    result = make_result(0.95, evidence=[ev])

    assessment = policy.assess([result])

    assert assessment.sufficient is True


def test_insufficient_when_confidence_below_threshold():
    """Assessment is insufficient when confidence is below threshold."""

    policy = AdaptiveEvidencePolicy(
        confidence_threshold=0.90,
        min_evidence_count=0,
    )
    result = make_result(0.60)

    assessment = policy.assess([result])

    assert assessment.sufficient is False
    assert "below" in assessment.reason.lower() or "threshold" in assessment.reason.lower()


def test_insufficient_when_evidence_count_below_minimum():
    """Assessment is insufficient when evidence count is too low."""

    policy = AdaptiveEvidencePolicy(
        confidence_threshold=0.80,
        min_evidence_count=2,
    )
    ev = make_evidence()
    result = make_result(0.95, evidence=[ev])

    assessment = policy.assess([result])

    assert assessment.sufficient is False
    assert "minimum" in assessment.reason.lower() or "evidence" in assessment.reason.lower()


def test_sufficient_with_zero_min_evidence_and_high_confidence():
    """Policy is satisfied with 0 min_evidence_count and high confidence."""

    policy = AdaptiveEvidencePolicy(
        confidence_threshold=0.80,
        min_evidence_count=0,
    )
    result = make_result(0.95, evidence=[])

    assessment = policy.assess([result])

    assert assessment.sufficient is True


def test_multiple_results_combined_sufficient():
    """Combined moderate-confidence results become sufficient together."""

    policy = AdaptiveEvidencePolicy(
        confidence_threshold=0.65,
        min_evidence_count=0,
    )
    r1 = make_result(0.70)
    r2 = make_result(0.70)

    assessment = policy.assess([r1, r2])

    assert assessment.sufficient is True


def test_contradictory_predictions_insufficient():
    """Contradictory confident predictions must not declare evidence sufficient."""

    policy = AdaptiveEvidencePolicy(
        confidence_threshold=0.90,
        min_evidence_count=0,
    )
    # One confident legitimate (0.05 -> conf 0.95), one confident phishing (0.95 -> conf 0.95)
    r1 = make_result(0.05)
    r2 = make_result(0.95)

    assessment = policy.assess([r1, r2])

    assert assessment.sufficient is False
    assert "contradict" in assessment.reason.lower() or "conflict" in assessment.reason.lower()


def test_decisive_second_agent_when_first_was_uncertain():
    """Decisive second agent provides sufficient evidence if no contradiction."""

    policy = AdaptiveEvidencePolicy(
        confidence_threshold=0.90,
        min_evidence_count=1,
    )
    ev = make_evidence()
    r1 = make_result(0.55, evidence=[])
    r2 = make_result(0.95, evidence=[ev])

    assessment = policy.assess([r1, r2])

    assert assessment.sufficient is True


# ============================================================
# INVALID INPUT
# ============================================================


def test_invalid_probability_raises():
    """AgentResult with probability > 1.0 raises ValueError."""

    policy = AdaptiveEvidencePolicy()

    class BadResult:
        phishing_probability = 1.5
        evidence = []

    with pytest.raises(ValueError, match="between 0 and 1"):
        policy.assess([BadResult()])


# ============================================================
# SERIALIZATION
# ============================================================


def test_to_dict_contains_expected_keys():
    """to_dict() returns all required keys."""

    policy = AdaptiveEvidencePolicy(min_evidence_count=0)
    ev = make_evidence()
    result = make_result(0.90, evidence=[ev])
    assessment = policy.assess([result])

    d = assessment.to_dict()

    expected_keys = {
        "phishing_probability",
        "confidence",
        "uncertainty",
        "evidence_strength",
        "evidence_count",
        "agent_count",
        "sufficient",
        "reason",
    }

    assert expected_keys.issubset(d.keys())


def test_to_dict_values_are_json_serializable():
    """to_dict() values can be serialized to JSON."""

    policy = AdaptiveEvidencePolicy(min_evidence_count=0)
    ev = make_evidence()
    result = make_result(0.90, evidence=[ev])
    assessment = policy.assess([result])

    d = assessment.to_dict()

    json_str = json.dumps(d)
    assert isinstance(json_str, str)


def test_to_dict_sufficient_is_bool():
    """to_dict() returns a real Python bool, not a numpy bool."""

    policy = AdaptiveEvidencePolicy(
        confidence_threshold=0.80,
        min_evidence_count=0,
    )
    result = make_result(0.95)
    assessment = policy.assess([result])

    d = assessment.to_dict()

    assert isinstance(d["sufficient"], bool)


def test_to_dict_empty_assessment():
    """to_dict() works for an empty (no results) assessment."""

    policy = AdaptiveEvidencePolicy()
    assessment = policy.assess([])

    d = assessment.to_dict()

    json_str = json.dumps(d)
    assert isinstance(json_str, str)
    assert d["sufficient"] is False


# ============================================================
# PACKAGE EXPORT
# ============================================================


def test_policy_importable_from_package():
    """AdaptiveEvidencePolicy is importable from the orchestration package."""

    from phishing_detection.orchestration import (
        AdaptiveEvidencePolicy,
        EvidencePolicyAssessment,
    )

    assert AdaptiveEvidencePolicy is not None
    assert EvidencePolicyAssessment is not None


# ============================================================
# ORCHESTRATOR INTEGRATION — PHASE 5
# ============================================================


def test_orchestrator_metadata_contains_policy_assessment(
    tmp_path,
):
    """Final metadata includes policy_assessment dict from Phase 5 policy."""

    model_path = tmp_path / "em.pkl"
    ext_path = tmp_path / "fe.pkl"
    joblib.dump(HighConfModel(), model_path)
    joblib.dump(FakeExtractor(), ext_path)

    email_agent = EmailAnalysisAgent(
        model_path=model_path,
        extractor_path=ext_path,
    )

    url_model_path = tmp_path / "um.pkl"
    url_pre_path = tmp_path / "up.pkl"
    joblib.dump(HighConfModel(), url_model_path)
    joblib.dump(FakePreprocessor(), url_pre_path)

    url_agent = URLAnalysisAgent(
        model_path=url_model_path,
        preprocessor_path=url_pre_path,
    )

    orchestrator = OrchestratorAgent(
        email_agent=email_agent,
        url_agent=url_agent,
        sender_agent=SenderAnalysisAgent(),
        confidence_threshold=0.90,
    )

    context = AgentContext(
        input_text="Verify your account now.",
        metadata={
            "url_features": {"f1": 1, "f2": 2}
        },
    )

    result = orchestrator.analyze(context)

    assert "policy_assessment" in result.metadata
    pa = result.metadata["policy_assessment"]
    assert isinstance(pa, dict)

    for key in [
        "phishing_probability",
        "confidence",
        "uncertainty",
        "evidence_count",
        "agent_count",
        "sufficient",
        "reason",
    ]:
        assert key in pa, f"Missing key: {key}"


def test_orchestrator_metadata_is_json_serializable(tmp_path):
    """Full result.to_dict() including policy_assessment serializes to JSON."""

    model_path = tmp_path / "em.pkl"
    ext_path = tmp_path / "fe.pkl"
    url_model_path = tmp_path / "um.pkl"
    url_pre_path = tmp_path / "up.pkl"

    joblib.dump(HighConfModel(), model_path)
    joblib.dump(FakeExtractor(), ext_path)
    joblib.dump(HighConfModel(), url_model_path)
    joblib.dump(FakePreprocessor(), url_pre_path)

    orchestrator = OrchestratorAgent(
        email_agent=EmailAnalysisAgent(
            model_path=model_path,
            extractor_path=ext_path,
        ),
        url_agent=URLAnalysisAgent(
            model_path=url_model_path,
            preprocessor_path=url_pre_path,
        ),
        sender_agent=SenderAnalysisAgent(),
    )

    context = AgentContext(
        input_text="Verify your account now.",
        metadata={"url_features": {"f1": 1}},
    )

    result = orchestrator.analyze(context)
    d = result.to_dict()

    json_str = json.dumps(d)
    assert isinstance(json_str, str)


def test_orchestrator_phase5_policy_metadata_keys(tmp_path):
    """Metadata contains all Phase 5 policy tracking keys."""

    model_path = tmp_path / "em.pkl"
    ext_path = tmp_path / "fe.pkl"
    url_model_path = tmp_path / "um.pkl"
    url_pre_path = tmp_path / "up.pkl"

    joblib.dump(HighConfModel(), model_path)
    joblib.dump(FakeExtractor(), ext_path)
    joblib.dump(HighConfModel(), url_model_path)
    joblib.dump(FakePreprocessor(), url_pre_path)

    orchestrator = OrchestratorAgent(
        email_agent=EmailAnalysisAgent(
            model_path=model_path,
            extractor_path=ext_path,
        ),
        url_agent=URLAnalysisAgent(
            model_path=url_model_path,
            preprocessor_path=url_pre_path,
        ),
        sender_agent=SenderAnalysisAgent(),
        confidence_threshold=0.90,
        max_agents=3,
    )

    context = AgentContext(
        input_text="Click here to verify.",
        metadata={"url_features": {"f1": 1}},
    )

    result = orchestrator.analyze(context)
    meta = result.metadata

    assert meta["policy"] == "adaptive_evidence_policy"
    assert "confidence_threshold" in meta
    assert "min_evidence_count" in meta
    assert "max_agents" in meta
    assert "agents_executed" in meta
    assert "evidence_count" in meta
    assert "total_execution_time_ms" in meta
    assert "policy_assessment" in meta


def test_orchestrator_decision_trace_has_policy_reason(tmp_path):
    """Decision trace stop/continue steps mention policy reason."""

    model_path = tmp_path / "em.pkl"
    ext_path = tmp_path / "fe.pkl"
    url_model_path = tmp_path / "um.pkl"
    url_pre_path = tmp_path / "up.pkl"

    joblib.dump(HighConfModel(), model_path)
    joblib.dump(FakeExtractor(), ext_path)
    joblib.dump(HighConfModel(), url_model_path)
    joblib.dump(FakePreprocessor(), url_pre_path)

    orchestrator = OrchestratorAgent(
        email_agent=EmailAnalysisAgent(
            model_path=model_path,
            extractor_path=ext_path,
        ),
        url_agent=URLAnalysisAgent(
            model_path=url_model_path,
            preprocessor_path=url_pre_path,
        ),
        sender_agent=SenderAnalysisAgent(),
        confidence_threshold=0.90,
    )

    context = AgentContext(
        input_text="Click here to verify.",
        metadata={"url_features": {"f1": 1}},
    )

    result = orchestrator.analyze(context)

    policy_steps = [
        step for step in result.decision_trace
        if step.action in {"stop", "continue"}
    ]
    assert len(policy_steps) >= 1

    for step in policy_steps:
        assert "Policy:" in step.reason


def test_orchestrator_max_agents_one_stops_after_first(tmp_path):
    """With max_agents=1, orchestration stops after the first agent."""

    model_path = tmp_path / "em.pkl"
    ext_path = tmp_path / "fe.pkl"
    url_model_path = tmp_path / "um.pkl"
    url_pre_path = tmp_path / "up.pkl"

    joblib.dump(LowConfModel(), model_path)
    joblib.dump(FakeExtractor(), ext_path)
    joblib.dump(LowConfModel(), url_model_path)
    joblib.dump(FakePreprocessor(), url_pre_path)

    orchestrator = OrchestratorAgent(
        email_agent=EmailAnalysisAgent(
            model_path=model_path,
            extractor_path=ext_path,
        ),
        url_agent=URLAnalysisAgent(
            model_path=url_model_path,
            preprocessor_path=url_pre_path,
        ),
        sender_agent=SenderAnalysisAgent(),
        confidence_threshold=0.99,
        max_agents=1,
    )

    context = AgentContext(
        input_text="Verify your account.",
        url="https://example.com",
        sender="a@b.com",
        headers={"spf": "pass"},
        metadata={"url_features": {"f1": 1}},
    )

    result = orchestrator.analyze(context)

    assert result.agent_count == 1


def test_orchestrator_agent_exception_recorded_in_trace(tmp_path):
    """Agent that raises an exception records an error step and continues."""

    crash_path = tmp_path / "crash.pkl"
    ext_path = tmp_path / "fe.pkl"
    url_model_path = tmp_path / "um.pkl"
    url_pre_path = tmp_path / "up.pkl"

    joblib.dump(CrashingModel(), crash_path)
    joblib.dump(FakeExtractor(), ext_path)
    joblib.dump(HighConfModel(), url_model_path)
    joblib.dump(FakePreprocessor(), url_pre_path)

    orchestrator = OrchestratorAgent(
        email_agent=EmailAnalysisAgent(
            model_path=crash_path,
            extractor_path=ext_path,
        ),
        url_agent=URLAnalysisAgent(
            model_path=url_model_path,
            preprocessor_path=url_pre_path,
        ),
        sender_agent=SenderAnalysisAgent(),
        confidence_threshold=0.90,
    )

    context = AgentContext(
        input_text="Verify your account.",
        metadata={"url_features": {"f1": 1}},
    )

    result = orchestrator.analyze(context)

    error_steps = [
        step for step in result.decision_trace
        if step.action == "error"
    ]
    assert len(error_steps) >= 1

    assert result.label in {0, 1}
    assert 0.0 <= result.phishing_probability <= 1.0


def test_orchestrator_custom_policy_injected(tmp_path):
    """Supplying an explicit policy overrides default threshold."""

    model_path = tmp_path / "em.pkl"
    ext_path = tmp_path / "fe.pkl"
    url_model_path = tmp_path / "um.pkl"
    url_pre_path = tmp_path / "up.pkl"

    joblib.dump(MedConfModel(), model_path)
    joblib.dump(FakeExtractor(), ext_path)
    joblib.dump(MedConfModel(), url_model_path)
    joblib.dump(FakePreprocessor(), url_pre_path)

    custom_policy = AdaptiveEvidencePolicy(
        confidence_threshold=0.60,
        min_evidence_count=0,
    )

    orchestrator = OrchestratorAgent(
        email_agent=EmailAnalysisAgent(
            model_path=model_path,
            extractor_path=ext_path,
        ),
        url_agent=URLAnalysisAgent(
            model_path=url_model_path,
            preprocessor_path=url_pre_path,
        ),
        sender_agent=SenderAnalysisAgent(),
        confidence_threshold=0.99,
        policy=custom_policy,
    )

    context = AgentContext(
        input_text="Verify your account now.",
        metadata={"url_features": {"f1": 1}},
    )

    result = orchestrator.analyze(context)

    assert result.agent_count == 1


def test_orchestrator_contradictory_predictions_continues_to_next_agent(tmp_path):
    """Contradictory predictions prevent early stopping and continue analysis."""

    model_path = tmp_path / "em.pkl"
    ext_path = tmp_path / "fe.pkl"
    url_model_path = tmp_path / "um.pkl"
    url_pre_path = tmp_path / "up.pkl"

    # Email says confident legitimate (p=0.05)
    joblib.dump(ConfidentLegitimateModel(), model_path)
    joblib.dump(FakeExtractor(), ext_path)

    # URL says confident phishing (p=0.95)
    joblib.dump(HighConfModel(), url_model_path)
    joblib.dump(FakePreprocessor(), url_pre_path)

    # With min_evidence_count=2, Email cannot stop alone; URL will run.
    # When URL runs, Email (legitimate) and URL (phishing) will contradict,
    # forcing the orchestrator to continue to Sender rather than stopping at URL.
    policy = AdaptiveEvidencePolicy(
        confidence_threshold=0.90,
        min_evidence_count=2,
    )

    orchestrator = OrchestratorAgent(
        email_agent=EmailAnalysisAgent(
            model_path=model_path,
            extractor_path=ext_path,
        ),
        url_agent=URLAnalysisAgent(
            model_path=url_model_path,
            preprocessor_path=url_pre_path,
        ),
        sender_agent=SenderAnalysisAgent(),
        policy=policy,
    )

    context = AgentContext(
        input_text="Legitimate looking email.",
        url="https://evil-phish.com",
        sender="admin@example.com",
        headers={"spf": "pass"},
        metadata={"url_features": {"f1": 1}},
    )

    result = orchestrator.analyze(context)

    # Because Email (legitimate) and URL (phishing) contradict each other,
    # the orchestrator must not stop after URL; it must continue to Sender agent (3 agents)
    assert result.agent_count == 3
