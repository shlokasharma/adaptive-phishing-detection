from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pytest

from phishing_detection.agents.base import AgentContext
from phishing_detection.agents.result import AgentResult
from phishing_detection.agents.url_agent import URLAnalysisAgent


class FakePreprocessor:
    """Fake URL preprocessor for unit testing."""

    def transform(self, features):
        assert features.shape == (1, 3)
        return features


class FakeModel:
    """Fake URL classifier for unit testing."""

    def predict(self, features):
        assert features.shape == (1, 3)
        return np.array([1])

    def predict_proba(self, features):
        assert features.shape == (1, 3)
        return np.array([[0.10, 0.90]])


class FakeLegitimateModel:
    """Fake classifier returning a legitimate prediction."""

    def predict(self, features):
        assert features.shape == (1, 3)
        return np.array([0])

    def predict_proba(self, features):
        assert features.shape == (1, 3)
        return np.array([[0.80, 0.20]])


def create_agent(
    tmp_path: Path,
    model=None,
) -> URLAnalysisAgent:
    """Create an URLAnalysisAgent with fake artifacts."""

    model_path = tmp_path / "model.pkl"
    preprocessor_path = tmp_path / "preprocessor.pkl"

    joblib.dump(
        model or FakeModel(),
        model_path,
    )

    joblib.dump(
        FakePreprocessor(),
        preprocessor_path,
    )

    return URLAnalysisAgent(
        model_path=model_path,
        preprocessor_path=preprocessor_path,
    )


def create_context() -> AgentContext:
    """Create a valid URL analysis context."""

    return AgentContext(
        url="http://example-phishing.test/login",
        metadata={
            "url_features": {
                "feature_1": 1,
                "feature_2": 2,
                "feature_3": 3,
            }
        },
    )


def test_agent_metadata(tmp_path):
    agent = create_agent(tmp_path)

    assert agent.get_agent_name() == "url_analysis_agent"
    assert agent.get_modality() == "url"

    capabilities = agent.get_capabilities()

    assert "url_analysis" in capabilities
    assert "engineered_url_feature_analysis" in capabilities
    assert "random_forest_classification" in capabilities


def test_can_analyze_valid_context(tmp_path):
    agent = create_agent(tmp_path)

    context = create_context()

    assert agent.can_analyze(context) is True


def test_cannot_analyze_without_url(tmp_path):
    agent = create_agent(tmp_path)

    context = AgentContext(
        url="",
        metadata={
            "url_features": {
                "feature_1": 1,
            }
        },
    )

    assert agent.can_analyze(context) is False


def test_cannot_analyze_without_features(tmp_path):
    agent = create_agent(tmp_path)

    context = AgentContext(
        url="https://example.com",
        metadata={},
    )

    assert agent.can_analyze(context) is False


def test_analyze_returns_agent_result(tmp_path):
    agent = create_agent(tmp_path)

    result = agent.analyze(
        create_context()
    )

    assert isinstance(result, AgentResult)


def test_prediction_is_phishing(tmp_path):
    agent = create_agent(tmp_path)

    result = agent.analyze(
        create_context()
    )

    assert result.label == 1
    assert result.label_name == "phishing"
    assert result.is_phishing is True
    assert result.is_legitimate is False


def test_probability_and_confidence(tmp_path):
    agent = create_agent(tmp_path)

    result = agent.analyze(
        create_context()
    )

    assert result.phishing_probability == pytest.approx(0.90)
    assert result.confidence == pytest.approx(0.90)
    assert result.uncertainty == pytest.approx(0.10)


def test_evidence_is_generated(tmp_path):
    agent = create_agent(tmp_path)

    result = agent.analyze(
        create_context()
    )

    assert result.evidence_count == 1

    evidence = result.evidence[0]

    assert evidence.evidence_type == "model_prediction"
    assert evidence.source == "url_analysis_agent"
    assert evidence.polarity == "supports_phishing"
    assert evidence.confidence == pytest.approx(0.90)
    assert evidence.reliability == pytest.approx(1.0)

    assert (
        evidence.metadata["model"]
        == "url_random_forest"
    )


def test_execution_metadata_is_generated(tmp_path):
    agent = create_agent(tmp_path)

    result = agent.analyze(
        create_context()
    )

    assert result.execution_metadata is not None

    assert (
        result.execution_metadata.model_name
        == "url_random_forest"
    )

    assert (
        result.execution_metadata.execution_time_ms
        >= 0.0
    )


def test_legitimate_prediction(tmp_path):
    agent = create_agent(
        tmp_path,
        model=FakeLegitimateModel(),
    )

    result = agent.analyze(
        create_context()
    )

    assert result.label == 0
    assert result.label_name == "legitimate"
    assert result.is_legitimate is True
    assert result.is_phishing is False

    assert (
        result.phishing_probability
        == pytest.approx(0.20)
    )

    assert (
        result.confidence
        == pytest.approx(0.80)
    )


def test_list_features_are_supported(tmp_path):
    agent = create_agent(tmp_path)

    context = AgentContext(
        url="https://example.com",
        metadata={
            "url_features": [1, 2, 3],
        },
    )

    result = agent.analyze(context)

    assert result.label == 1


def test_numpy_features_are_supported(tmp_path):
    agent = create_agent(tmp_path)

    context = AgentContext(
        url="https://example.com",
        metadata={
            "url_features": np.array(
                [1, 2, 3]
            ),
        },
    )

    result = agent.analyze(context)

    assert result.label == 1


def test_empty_features_raise_error(tmp_path):
    agent = create_agent(tmp_path)

    context = AgentContext(
        url="https://example.com",
        metadata={
            "url_features": [],
        },
    )

    with pytest.raises(
        ValueError,
        match="at least one feature",
    ):
        agent.analyze(context)


def test_non_numeric_features_raise_error(tmp_path):
    agent = create_agent(tmp_path)

    context = AgentContext(
        url="https://example.com",
        metadata={
            "url_features": [
                1,
                "invalid",
                3,
            ],
        },
    )

    with pytest.raises(
        ValueError,
        match="numeric",
    ):
        agent.analyze(context)


def test_missing_model_artifact_raises_error(tmp_path):
    agent = URLAnalysisAgent(
        model_path=tmp_path / "missing_model.pkl",
        preprocessor_path=tmp_path / "preprocessor.pkl",
    )

    context = create_context()

    with pytest.raises(
        FileNotFoundError,
        match="URL model artifact",
    ):
        agent.analyze(context)


def test_missing_preprocessor_artifact_raises_error(tmp_path):
    model_path = tmp_path / "model.pkl"

    joblib.dump(
        FakeModel(),
        model_path,
    )

    agent = URLAnalysisAgent(
        model_path=model_path,
        preprocessor_path=tmp_path / "missing_preprocessor.pkl",
    )

    context = create_context()

    with pytest.raises(
        FileNotFoundError,
        match="URL preprocessor artifact",
    ):
        agent.analyze(context)


def test_analyze_requires_url_and_features(tmp_path):
    agent = create_agent(tmp_path)

    context = AgentContext(
        url="",
        metadata={},
    )

    with pytest.raises(
        ValueError,
        match="non-empty URL",
    ):
        agent.analyze(context)