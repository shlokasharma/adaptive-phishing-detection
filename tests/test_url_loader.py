from pathlib import Path

import pandas as pd
import pytest

from phishing_detection.data.url_loader import (
    URLDatasetLoader,
)


def create_test_dataset():
    return pd.DataFrame(
        {
            "sample_id": [
                "url_1",
                "url_2",
                "url_3",
            ],
            "source_dataset": [
                "test",
                "test",
                "test",
            ],
            "label": [
                0,
                1,
                1,
            ],
            "feature_a": [
                1,
                2,
                3,
            ],
            "feature_b": [
                0.1,
                0.2,
                0.3,
            ],
        }
    )


def test_url_loader_reads_dataset(
    tmp_path: Path,
):
    dataset = create_test_dataset()

    file_path = (
        tmp_path
        / "url.csv"
    )

    dataset.to_csv(
        file_path,
        index=False,
    )

    loader = URLDatasetLoader(
        file_path
    )

    loaded = loader.load()

    assert len(loaded) == 3

    assert set(
        [
            "sample_id",
            "source_dataset",
            "label",
            "feature_a",
            "feature_b",
        ]
    ).issubset(
        loaded.columns
    )


def test_url_loader_extracts_features_and_labels(
    tmp_path: Path,
):
    dataset = create_test_dataset()

    file_path = (
        tmp_path
        / "url.csv"
    )

    dataset.to_csv(
        file_path,
        index=False,
    )

    loader = URLDatasetLoader(
        file_path
    )

    X, y = (
        loader.get_features_and_labels()
    )

    assert list(
        X.columns
    ) == [
        "feature_a",
        "feature_b",
    ]

    assert list(y) == [
        0,
        1,
        1,
    ]


def test_url_loader_returns_feature_names(
    tmp_path: Path,
):
    dataset = create_test_dataset()

    file_path = (
        tmp_path
        / "url.csv"
    )

    dataset.to_csv(
        file_path,
        index=False,
    )

    loader = URLDatasetLoader(
        file_path
    )

    feature_columns = (
        loader.get_feature_columns()
    )

    assert feature_columns == [
        "feature_a",
        "feature_b",
    ]


def test_missing_url_dataset_is_rejected(
    tmp_path: Path,
):
    file_path = (
        tmp_path
        / "missing.csv"
    )

    loader = URLDatasetLoader(
        file_path
    )

    with pytest.raises(
        FileNotFoundError
    ):
        loader.load()


def test_missing_required_column_is_rejected(
    tmp_path: Path,
):
    dataset = pd.DataFrame(
        {
            "sample_id": [
                "url_1"
            ],
            "label": [
                0
            ],
        }
    )

    file_path = (
        tmp_path
        / "invalid.csv"
    )

    dataset.to_csv(
        file_path,
        index=False,
    )

    loader = URLDatasetLoader(
        file_path
    )

    with pytest.raises(
        ValueError
    ):
        loader.load()