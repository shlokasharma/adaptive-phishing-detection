from __future__ import annotations

import numpy as np

from phishing_detection.agents.base import AgentContext
from phishing_detection.agents.email_agent import EmailAnalysisAgent
from phishing_detection.agents.sender_agent import SenderAnalysisAgent
from phishing_detection.agents.url_agent import URLAnalysisAgent
from phishing_detection.orchestration.orchestrator import (
    OrchestrationResult,
    OrchestratorAgent,
)


class FakeExtractor:
    def transform(self, texts):
        return np.array([[1.0, 2.0]])


class HighConfidenceEmailModel:
    def predict(self, features):
        return np.array([1])

    def predict_proba(self, features):
        return np.array([[0.05, 0.95]])


class UncertainEmailModel:
    def predict(self, features):
        return np.array([1])

    def predict_proba(self, features):
        return np.array([[0.45, 0.55]])


class HighConfidenceURLModel:
    def predict(self, features):
        return np.array([1])

    def predict_proba(self, features):
        return np.array([[0.10, 0.90]])


class UncertainURLModel:
    def predict(self, features):
        return np.array([1])

    def predict_proba(self, features):
        return np.array([[0.40, 0.60]])


class FakePreprocessor:
    def transform(self, features):
        return features


def create_email_agent(
    tmp_path,
    model,
):
    import joblib

    model_path = tmp_path / "email_model.pkl"
    extractor_path = tmp_path / "email_extractor.pkl"

    joblib.dump(model, model_path)
    joblib.dump(FakeExtractor(), extractor_path)

    return EmailAnalysisAgent(
        model_path=model_path,
        extractor_path=extractor_path,
    )


def create_url_agent(
    tmp_path,
    model,
):
    import joblib

    model_path = tmp_path / "url_model.pkl"
    preprocessor_path = tmp_path / "url_preprocessor.pkl"

    joblib.dump(model, model_path)
    joblib.dump(
        FakePreprocessor(),
        preprocessor_path,
    )

    return URLAnalysisAgent(
        model_path=model_path,
        preprocessor_path=preprocessor_path,
    )


def create_context(
    *,
    include_url=True,
    include_sender=True,
):
    return AgentContext(
        input_text="Urgent account verification required.",
        url=(
            "https://example.com/login"
            if include_url
            else None
        ),
        sender=(
            "security@example.com"
            if include_sender
            else None
        ),
        headers={
            "spf": "pass",
        }
        if include_sender
        else {},
        metadata={
            "url_features": {
                "feature_1": 1,
                "feature_2": 2,
            }
        }
        if include_url
        else {},
    )


def create_orchestrator(
    tmp_path,
    email_model,
    url_model,
    confidence_threshold=0.90,
):
    email_agent = create_email_agent(
        tmp_path,
        email_model,
    )

    url_agent = create_url_agent(
        tmp_path,
        url_model,
    )

    sender_agent = SenderAnalysisAgent()

    return OrchestratorAgent(
        email_agent=email_agent,
        url_agent=url_agent,
        sender_agent=sender_agent,
        confidence_threshold=confidence_threshold,
    )


def test_orchestrator_returns_result(tmp_path):
    orchestrator = create_orchestrator(
        tmp_path,
        HighConfidenceEmailModel(),
        HighConfidenceURLModel(),
    )

    result = orchestrator.analyze(
        create_context()
    )

    assert isinstance(
        result,
        OrchestrationResult,
    )


def test_high_confidence_email_stops_early(tmp_path):
    orchestrator = create_orchestrator(
        tmp_path,
        HighConfidenceEmailModel(),
        HighConfidenceURLModel(),
    )

    result = orchestrator.analyze(
        create_context()
    )

    assert result.agent_count == 1

    assert (
        result.agent_results[0].agent_name
        == "email_analysis_agent"
    )

    assert any(
        step.action == "stop"
        and step.agent_name
        == "email_analysis_agent"
        for step in result.decision_trace
    )


def test_uncertain_email_triggers_url_analysis(
    tmp_path,
):
    orchestrator = create_orchestrator(
        tmp_path,
        UncertainEmailModel(),
        HighConfidenceURLModel(),
    )

    result = orchestrator.analyze(
        create_context()
    )

    assert result.agent_count == 2

    agent_names = [
        item.agent_name
        for item in result.agent_results
    ]

    assert "email_analysis_agent" in agent_names
    assert "url_analysis_agent" in agent_names


def test_uncertain_email_and_url_triggers_sender(
    tmp_path,
):
    orchestrator = create_orchestrator(
        tmp_path,
        UncertainEmailModel(),
        UncertainURLModel(),
    )

    result = orchestrator.analyze(
        create_context()
    )

    assert result.agent_count == 3

    agent_names = [
        item.agent_name
        for item in result.agent_results
    ]

    assert (
        "sender_analysis_agent"
        in agent_names
    )


def test_email_only_context(tmp_path):
    orchestrator = create_orchestrator(
        tmp_path,
        UncertainEmailModel(),
        HighConfidenceURLModel(),
    )

    result = orchestrator.analyze(
        create_context(
            include_url=False,
            include_sender=False,
        )
    )

    assert result.agent_count == 1

    assert (
        result.agent_results[0].agent_name
        == "email_analysis_agent"
    )


def test_url_is_skipped_when_unavailable(
    tmp_path,
):
    orchestrator = create_orchestrator(
        tmp_path,
        UncertainEmailModel(),
        HighConfidenceURLModel(),
    )

    result = orchestrator.analyze(
        create_context(
            include_url=False,
            include_sender=True,
        )
    )

    assert result.agent_count == 2

    agent_names = [
        item.agent_name
        for item in result.agent_results
    ]

    assert (
        "url_analysis_agent"
        not in agent_names
    )

    assert (
        "sender_analysis_agent"
        in agent_names
    )


def test_sender_is_skipped_when_unavailable(
    tmp_path,
):
    orchestrator = create_orchestrator(
        tmp_path,
        UncertainEmailModel(),
        UncertainURLModel(),
    )

    result = orchestrator.analyze(
        create_context(
            include_url=True,
            include_sender=False,
        )
    )

    assert result.agent_count == 2

    agent_names = [
        item.agent_name
        for item in result.agent_results
    ]

    assert (
        "sender_analysis_agent"
        not in agent_names
    )


def test_evidence_is_combined(tmp_path):
    orchestrator = create_orchestrator(
        tmp_path,
        UncertainEmailModel(),
        HighConfidenceURLModel(),
    )

    result = orchestrator.analyze(
        create_context()
    )

    assert result.evidence_count >= 2


def test_decision_trace_is_generated(tmp_path):
    orchestrator = create_orchestrator(
        tmp_path,
        UncertainEmailModel(),
        HighConfidenceURLModel(),
    )

    result = orchestrator.analyze(
        create_context()
    )

    assert len(result.decision_trace) >= 3

    step_numbers = [
        step.step_number
        for step in result.decision_trace
    ]

    assert step_numbers == list(
        range(1, len(step_numbers) + 1)
    )


def test_final_probability_is_valid(tmp_path):
    orchestrator = create_orchestrator(
        tmp_path,
        UncertainEmailModel(),
        HighConfidenceURLModel(),
    )

    result = orchestrator.analyze(
        create_context()
    )

    assert (
        0.0
        <= result.phishing_probability
        <= 1.0
    )

    assert (
        0.0
        <= result.confidence
        <= 1.0
    )

    assert (
        0.0
        <= result.uncertainty
        <= 1.0
    )


def test_final_label_matches_probability(
    tmp_path,
):
    orchestrator = create_orchestrator(
        tmp_path,
        UncertainEmailModel(),
        HighConfidenceURLModel(),
    )

    result = orchestrator.analyze(
        create_context()
    )

    expected_label = int(
        result.phishing_probability >= 0.5
    )

    assert result.label == expected_label


def test_to_dict_is_serializable(tmp_path):
    orchestrator = create_orchestrator(
        tmp_path,
        UncertainEmailModel(),
        HighConfidenceURLModel(),
    )

    result = orchestrator.analyze(
        create_context()
    )

    output = result.to_dict()

    assert isinstance(output, dict)
    assert "agent_results" in output
    assert "decision_trace" in output
    assert "evidence" in output
    assert "metadata" in output


def test_max_agents_is_respected(tmp_path):
    email_agent = create_email_agent(
        tmp_path,
        UncertainEmailModel(),
    )

    url_agent = create_url_agent(
        tmp_path,
        UncertainURLModel(),
    )

    sender_agent = SenderAnalysisAgent()

    orchestrator = OrchestratorAgent(
        email_agent=email_agent,
        url_agent=url_agent,
        sender_agent=sender_agent,
        confidence_threshold=0.99,
        max_agents=2,
    )

    result = orchestrator.analyze(
        create_context()
    )

    assert result.agent_count == 2


def test_invalid_threshold_raises(tmp_path):
    email_agent = create_email_agent(
        tmp_path,
        HighConfidenceEmailModel(),
    )

    url_agent = create_url_agent(
        tmp_path,
        HighConfidenceURLModel(),
    )

    sender_agent = SenderAnalysisAgent()

    try:
        OrchestratorAgent(
            email_agent=email_agent,
            url_agent=url_agent,
            sender_agent=sender_agent,
            confidence_threshold=1.5,
        )
    except ValueError:
        return

    raise AssertionError(
        "Expected ValueError for invalid threshold."
    )


def test_invalid_max_agents_raises(tmp_path):
    email_agent = create_email_agent(
        tmp_path,
        HighConfidenceEmailModel(),
    )

    url_agent = create_url_agent(
        tmp_path,
        HighConfidenceURLModel(),
    )

    sender_agent = SenderAnalysisAgent()

    try:
        OrchestratorAgent(
            email_agent=email_agent,
            url_agent=url_agent,
            sender_agent=sender_agent,
            max_agents=0,
        )
    except ValueError:
        return

    raise AssertionError(
        "Expected ValueError for invalid max_agents."
    )