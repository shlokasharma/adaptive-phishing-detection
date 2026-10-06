import pandas as pd

from phishing_detection.features.url_features import URLFeatureExtractor


def test_url_training_pipeline():
    """
    Verify the complete URL feature extraction pipeline using
    a small synthetic dataset.

    Full UCI training data is intentionally excluded from the
    automated test suite for performance and reproducibility.
    """

    X = pd.DataFrame(
        [
            [1, 10, 0, 0, 0],
            [0, 20, 1, 1, 1],
            [1, 15, 0, 0, 0],
            [0, 30, 1, 1, 1],
            [1, 12, 0, 0, 0],
            [0, 25, 1, 1, 1],
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
        [0, 1, 0, 1, 0, 1],
        dtype=int,
    )

    assert len(X) > 0
    assert len(X) == len(y)
    assert set(y.unique()).issubset({0, 1})

    extractor = URLFeatureExtractor()

    transformed = extractor.fit_transform(X)

    assert transformed.shape[0] == len(X)
    assert transformed.shape[1] > 0
    assert extractor.get_num_features() == transformed.shape[1]