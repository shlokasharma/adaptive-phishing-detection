from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from phishing_detection.agents.base import AgentContext
from phishing_detection.agents.email_agent import EmailAnalysisAgent
from phishing_detection.agents.result import AgentResult


class FakeExtractor:
    """Small fake extractor used for unit testing."""

    def transform(self, texts):
        assert len(texts) == 1
        return np.array([[1.0, 2.0]])


class FakeModel:
    """Small fake classifier used for unit testing."""

    def predict(self, features):
        assert features.shape == (1, 2)
        return np.array([1])

    def predict_proba(self, features):
        assert features.shape == (1, 2)
        return np.array([[0.15, 0.85]])


def create_agent(tmp_path: Path) -> EmailAnalysisAgent:
    """Create an EmailAnalysisAgent with temporary fake artifacts."""

    import joblib

    model_path = tmp_path / "model.pkl"
    extractor_path = tmp_path / "extractor.pkl"

    joblib.dump(FakeModel(), model_path)
    joblib.dump(FakeExtractor(), extractor_path)

    return EmailAnalysisAgent(
        model_path=model_path,
        extractor_path=extractor_path,
    )


def test_agent_metadata(tmp_path):
    agent = create_agent(tmp_path)

    assert agent.get_agent_name() == "email_analysis_agent"
    assert agent.get_modality() == "email"

    capabilities = agent.get_capabilities()

    assert "email_text_analysis" in capabilities
    assert "tfidf_feature_extraction" in capabilities
    assert "linear_svm_classification" in capabilities


def test_can_analyze_with_email_text(tmp_path):
    agent = create_agent(tmp_path)

    context = AgentContext(
        input_text="Please verify your account immediately."
    )

    assert agent.can_analyze(context) is True


def test_cannot_analyze_empty_email(tmp_path):
    agent = create_agent(tmp_path)

    context = AgentContext(
        input_text=""
    )

    assert agent.can_analyze(context) is False


def test_cannot_analyze_whitespace_email(tmp_path):
    agent = create_agent(tmp_path)

    context = AgentContext(
        input_text="   "
    )

    assert agent.can_analyze(context) is False


def test_analyze_returns_agent_result(tmp_path):
    agent = create_agent(tmp_path)

    context = AgentContext(
        input_text=(
            "URGENT: Your account will be suspended. "
            "Click the link immediately."
        )
    )

    result = agent.analyze(context)

    assert isinstance(result, AgentResult)


def test_prediction_is_phishing(tmp_path):
    agent = create_agent(tmp_path)

    context = AgentContext(
        input_text="Click this urgent link to verify your account."
    )

    result = agent.analyze(context)

    assert result.label == 1
    assert result.label_name == "phishing"
    assert result.is_phishing is True
    assert result.is_legitimate is False


def test_probability_and_confidence(tmp_path):
    agent = create_agent(tmp_path)

    context = AgentContext(
        input_text="Click this urgent link to verify your account."
    )

    result = agent.analyze(context)

    assert result.phishing_probability == pytest.approx(0.85)
    assert result.confidence == pytest.approx(0.85)
    assert result.uncertainty == pytest.approx(0.15)


def test_evidence_is_generated(tmp_path):
    agent = create_agent(tmp_path)

    context = AgentContext(
        input_text="Click this urgent link immediately."
    )

    result = agent.analyze(context)

    assert result.evidence_count == 1

    evidence = result.evidence[0]

    assert evidence.evidence_type == "model_prediction"
    assert evidence.source == "email_analysis_agent"
    assert evidence.polarity == "supports_phishing"
    assert evidence.confidence == pytest.approx(0.85)
    assert evidence.reliability == pytest.approx(1.0)


def test_execution_metadata_is_generated(tmp_path):
    agent = create_agent(tmp_path)

    context = AgentContext(
        input_text="Your account requires verification."
    )

    result = agent.analyze(context)

    assert result.execution_metadata is not None
    assert result.execution_metadata.model_name == "email_linear_svm"
    assert result.execution_metadata.execution_time_ms >= 0.0


def test_missing_model_artifact_raises_error(tmp_path):
    agent = EmailAnalysisAgent(
        model_path=tmp_path / "missing_model.pkl",
        extractor_path=tmp_path / "extractor.pkl",
    )

    context = AgentContext(
        input_text="Test email"
    )

    with pytest.raises(FileNotFoundError, match="Email model artifact"):
        agent.analyze(context)


def test_missing_extractor_artifact_raises_error(tmp_path):
    import joblib

    model_path = tmp_path / "model.pkl"
    extractor_path = tmp_path / "missing_extractor.pkl"

    joblib.dump(FakeModel(), model_path)

    agent = EmailAnalysisAgent(
        model_path=model_path,
        extractor_path=extractor_path,
    )

    context = AgentContext(
        input_text="Test email"
    )

    with pytest.raises(
        FileNotFoundError,
        match="Email TF-IDF extractor",
    ):
        agent.analyze(context)


def test_analyze_requires_input_text(tmp_path):
    agent = create_agent(tmp_path)

    context = AgentContext(
        input_text=""
    )

    with pytest.raises(ValueError, match="non-empty input_text"):
        agent.analyze(context)