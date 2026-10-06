from scipy.sparse import csr_matrix

import pytest

from phishing_detection.models.base import PredictionResult
from phishing_detection.models.traditional_ml.email_logistic_regression import (
    EmailLogisticRegressionDetector,
)


def create_toy_dataset():
    """
    Create a tiny synthetic feature matrix for unit testing.

    Feature 0 is associated with legitimate examples.
    Feature 1 is associated with phishing examples.
    """

    X = csr_matrix(
        [
            [3.0, 0.0],
            [2.5, 0.0],
            [0.0, 3.0],
            [0.0, 2.5],
        ]
    )

    y = [0, 0, 1, 1]

    return X, y


def test_detector_can_fit_and_predict():
    X, y = create_toy_dataset()

    detector = EmailLogisticRegressionDetector()

    detector.fit(X, y)

    predictions = detector.predict(X)

    assert len(predictions) == 4
    assert set(predictions).issubset({0, 1})


def test_detector_returns_probabilities():
    X, y = create_toy_dataset()

    detector = EmailLogisticRegressionDetector()

    detector.fit(X, y)

    probabilities = detector.predict_proba(X)

    assert probabilities.shape == (4, 2)

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

    detector = EmailLogisticRegressionDetector()

    detector.fit(X, y)

    result = detector.predict_one(X[0])

    assert isinstance(result, PredictionResult)

    assert result.label in (0, 1)

    assert result.label_name in {
        "legitimate",
        "phishing",
    }

    assert 0.0 <= result.confidence <= 1.0

    assert result.phishing_probability is not None

    assert 0.0 <= result.phishing_probability <= 1.0

    assert result.model_name == (
        "email_tfidf_logistic_regression"
    )

    assert result.modality == "email"


def test_prediction_before_training_is_rejected():
    detector = EmailLogisticRegressionDetector()

    X, _ = create_toy_dataset()

    with pytest.raises(RuntimeError):
        detector.predict(X)


def test_invalid_prediction_label_is_rejected():
    detector = EmailLogisticRegressionDetector()

    with pytest.raises(ValueError):
        detector.build_prediction_result(
            label=2,
            phishing_probability=0.5,
        )