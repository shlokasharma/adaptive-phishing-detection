from scipy.sparse import csr_matrix

import pytest

from phishing_detection.models.base import PredictionResult
from phishing_detection.models.traditional_ml.email_random_forest import (
    EmailRandomForestDetector,
)


def create_toy_dataset():
    """
    Create a small synthetic dataset for Random Forest testing.
    """

    X = csr_matrix(
        [
            [3.0, 0.0],
            [2.5, 0.0],
            [0.0, 3.0],
            [0.0, 2.5],
            [2.8, 0.1],
            [0.1, 2.8],
            [2.6, 0.2],
            [0.2, 2.6],
        ]
    )

    y = [0, 0, 1, 1, 0, 1, 0, 1]

    return X, y


def test_detector_can_fit_and_predict():
    X, y = create_toy_dataset()

    detector = EmailRandomForestDetector(
        n_estimators=20,
        n_jobs=-1,
        random_state=42,
    )

    detector.fit(X, y)

    predictions = detector.predict(X)

    assert len(predictions) == len(y)

    assert set(predictions).issubset({0, 1})


def test_detector_returns_probabilities():
    X, y = create_toy_dataset()

    detector = EmailRandomForestDetector(
        n_estimators=20,
        n_jobs=-1,
        random_state=42,
    )

    detector.fit(X, y)

    probabilities = detector.predict_proba(X)

    assert probabilities.shape == (len(y), 2)

    assert all(
        0.0 <= probability <= 1.0
        for probability in probabilities.flatten()
    )

    row_sums = probabilities.sum(axis=1)

    assert all(
        abs(row_sum - 1.0) < 1e-6
        for row_sum in row_sums
    )


def test_predict_one_returns_prediction_result():
    X, y = create_toy_dataset()

    detector = EmailRandomForestDetector(
        n_estimators=20,
        n_jobs=-1,
        random_state=42,
    )

    detector.fit(X, y)

    result = detector.predict_one(X[0])

    assert isinstance(
        result,
        PredictionResult,
    )

    assert result.label in (0, 1)

    assert result.label_name in {
        "legitimate",
        "phishing",
    }

    assert 0.0 <= result.confidence <= 1.0

    assert result.phishing_probability is not None

    assert 0.0 <= result.phishing_probability <= 1.0

    assert result.model_name == (
        "email_tfidf_random_forest"
    )

    assert result.modality == "email"


def test_prediction_before_training_is_rejected():
    detector = EmailRandomForestDetector()

    X, _ = create_toy_dataset()

    with pytest.raises(RuntimeError):
        detector.predict(X)


def test_invalid_prediction_label_is_rejected():
    detector = EmailRandomForestDetector()

    with pytest.raises(ValueError):
        detector.build_prediction_result(
            label=2,
            phishing_probability=0.5,
        )