import numpy as np

from phishing_detection.features.url_features import URLFeatureExtractor
from phishing_detection.models.traditional_ml.url_gradient_boosting import (
    URLGradientBoostingDetector,
)


def test_url_gradient_boosting_pipeline():
    """
    Verify the complete URL feature extraction -> Gradient Boosting
    pipeline using a small synthetic dataset.

    Full UCI training is performed by the dedicated experiment scripts,
    not by the automated test suite.
    """

    X = np.array(
        [
            [1, 10, 0, 0, 0],
            [0, 20, 1, 1, 1],
            [1, 15, 0, 0, 0],
            [0, 30, 1, 1, 1],
            [1, 12, 0, 0, 0],
            [0, 25, 1, 1, 1],
            [1, 18, 0, 0, 0],
            [0, 35, 1, 1, 1],
            [1, 11, 0, 0, 0],
            [0, 28, 1, 1, 1],
            [1, 14, 0, 0, 0],
            [0, 32, 1, 1, 1],
        ],
        dtype=float,
    )

    y = np.array(
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        dtype=int,
    )

    feature_extractor = URLFeatureExtractor()

    X_transformed = feature_extractor.fit_transform(
        __import__("pandas").DataFrame(
            X,
            columns=[
                "feature_1",
                "feature_2",
                "feature_3",
                "feature_4",
                "feature_5",
            ],
        )
    )

    model = URLGradientBoostingDetector(
        n_estimators=10,
    )

    model.fit(
        X_transformed,
        y,
    )

    predictions = model.predict(
        X_transformed,
    )

    probabilities = model.predict_proba(
        X_transformed,
    )

    assert len(predictions) == len(y)

    assert probabilities.shape == (
        len(y),
        2,
    )

    assert set(predictions).issubset({0, 1})