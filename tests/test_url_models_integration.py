import numpy as np
import pandas as pd

from phishing_detection.features.url_features import URLFeatureExtractor
from phishing_detection.models.traditional_ml.url_logistic_regression import (
    URLLogisticRegressionDetector,
)
from phishing_detection.models.traditional_ml.url_random_forest import (
    URLRandomForestDetector,
)


def test_url_models_pipeline():
    """
    Verify that the URL feature pipeline can be consumed by both
    Logistic Regression and Random Forest detectors.

    A small synthetic dataset is used so that the test remains fast.
    Full UCI experiments are handled by dedicated training scripts.
    """

    X = pd.DataFrame(
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
        columns=[
            "feature_1",
            "feature_2",
            "feature_3",
            "feature_4",
            "feature_5",
        ],
        dtype=float,
    )

    y = pd.Series(
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        dtype=int,
    )

    extractor = URLFeatureExtractor()

    X_transformed = extractor.fit_transform(X)

    logistic_model = URLLogisticRegressionDetector()

    logistic_model.fit(
        X_transformed,
        y,
    )

    logistic_predictions = logistic_model.predict(
        X_transformed,
    )

    assert len(logistic_predictions) == len(y)
    assert set(logistic_predictions).issubset({0, 1})

    random_forest_model = URLRandomForestDetector(
        n_estimators=10,
    )

    random_forest_model.fit(
        X_transformed,
        y,
    )

    random_forest_predictions = random_forest_model.predict(
        X_transformed,
    )

    assert len(random_forest_predictions) == len(y)
    assert set(random_forest_predictions).issubset({0, 1})