import pytest

from phishing_detection.models.base import (
    BasePhishingDetector,
    PredictionResult,
)


class DummyDetector(BasePhishingDetector):
    """Minimal detector used to test the common interface."""

    def fit(self, X, y):
        return self

    def predict(self, X):
        return [1]

    def predict_proba(self, X):
        return [[0.2, 0.8]]

    def get_model_name(self):
        return "dummy_detector"

    def get_modality(self):
        return "email"


def test_prediction_result_for_phishing():
    detector = DummyDetector()

    result = detector.build_prediction_result(
        label=1,
        phishing_probability=0.9,
    )

    assert isinstance(result, PredictionResult)
    assert result.label == 1
    assert result.label_name == "phishing"
    assert result.confidence == pytest.approx(0.9)
    assert result.phishing_probability == pytest.approx(0.9)
    assert result.model_name == "dummy_detector"
    assert result.modality == "email"


def test_prediction_result_for_legitimate():
    detector = DummyDetector()

    result = detector.build_prediction_result(
        label=0,
        phishing_probability=0.1,
    )

    assert result.label == 0
    assert result.label_name == "legitimate"
    assert result.confidence == pytest.approx(0.9)
    assert result.phishing_probability == pytest.approx(0.1)


def test_invalid_label_is_rejected():
    detector = DummyDetector()

    with pytest.raises(ValueError):
        detector.build_prediction_result(
            label=2,
            phishing_probability=0.5,
        )


def test_invalid_probability_is_rejected():
    detector = DummyDetector()

    with pytest.raises(ValueError):
        detector.build_prediction_result(
            label=1,
            phishing_probability=1.5,
        )