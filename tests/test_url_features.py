import pandas as pd
import pytest

from phishing_detection.features.url_features import (
    URLFeatureExtractor,
)


def create_test_features():
    return pd.DataFrame(
        {
            "feature_a": [
                1.0,
                2.0,
                3.0,
                4.0,
            ],
            "feature_b": [
                10.0,
                20.0,
                30.0,
                40.0,
            ],
        }
    )


def test_fit_transform_returns_expected_shape():
    X = create_test_features()

    extractor = URLFeatureExtractor()

    transformed = (
        extractor.fit_transform(X)
    )

    assert transformed.shape == (
        4,
        2,
    )


def test_transform_after_fit_works():
    X = create_test_features()

    extractor = URLFeatureExtractor()

    extractor.fit(X)

    transformed = (
        extractor.transform(X)
    )

    assert transformed.shape == (
        4,
        2,
    )


def test_transform_before_fit_is_rejected():
    X = create_test_features()

    extractor = URLFeatureExtractor()

    with pytest.raises(
        RuntimeError
    ):
        extractor.transform(X)


def test_feature_names_are_available():
    X = create_test_features()

    extractor = URLFeatureExtractor()

    extractor.fit(X)

    assert extractor.get_feature_names() == [
        "feature_a",
        "feature_b",
    ]

    assert (
        extractor.get_num_features()
        == 2
    )


def test_column_mismatch_is_rejected():
    X = create_test_features()

    extractor = URLFeatureExtractor()

    extractor.fit(X)

    invalid_X = pd.DataFrame(
        {
            "feature_a": [
                1.0
            ],
            "feature_c": [
                2.0
            ],
        }
    )

    with pytest.raises(
        ValueError
    ):
        extractor.transform(
            invalid_X
        )