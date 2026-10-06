"""
Unit tests for the URL Gradient Boosting detector.
"""

import numpy as np
import pytest

from phishing_detection.models.traditional_ml.url_gradient_boosting import (
    URLGradientBoostingDetector,
)


@pytest.fixture
def training_data():
    X = np.array(
        [
            [0.1, 0.2, 0.1],
            [0.2, 0.1, 0.2],
            [0.8, 0.9, 0.8],
            [0.9, 0.8, 0.9],
            [0.15, 0.25, 0.15],
            [0.85, 0.95, 0.85],
        ]
    )

    y = np.array(
        [0, 0, 1, 1, 0, 1]
    )

    return X, y


def test_model_can_fit(training_data):
    X, y = training_data

    model = URLGradientBoostingDetector(
        n_estimators=20
    )

    model.fit(X, y)

    assert model._is_fitted is True


def test_predict_returns_binary_labels(training_data):
    X, y = training_data

    model = URLGradientBoostingDetector(
        n_estimators=20
    )

    model.fit(X, y)

    predictions = model.predict(X)

    assert len(predictions) == len(y)
    assert set(predictions).issubset({0, 1})


def test_predict_proba_returns_probabilities(training_data):
    X, y = training_data

    model = URLGradientBoostingDetector(
        n_estimators=20
    )

    model.fit(X, y)

    probabilities = model.predict_proba(X)

    assert probabilities.shape == (len(X), 2)
    assert np.all(probabilities >= 0.0)
    assert np.all(probabilities <= 1.0)


def test_predict_one_returns_prediction_result(training_data):
    X, y = training_data

    model = URLGradientBoostingDetector(
        n_estimators=20
    )

    model.fit(X, y)

    result = model.predict_one(X[:1])

    assert result.label in {0, 1}

    assert result.label_name in {
        "legitimate",
        "phishing",
    }

    assert 0.0 <= result.confidence <= 1.0

    assert result.phishing_probability is not None

    assert 0.0 <= result.phishing_probability <= 1.0

    assert result.model_name == "url_gradient_boosting"

    assert result.modality == "url"


def test_prediction_before_fit_is_rejected(training_data):
    X, _ = training_data

    model = URLGradientBoostingDetector(
        n_estimators=20
    )

    with pytest.raises(RuntimeError):
        model.predict(X)