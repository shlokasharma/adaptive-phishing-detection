"""
Utilities for loading Phase 1 URL datasets.

The URL datasets use numerical/security-related website features.
This loader provides a common interface for loading URL train,
validation, and test splits while preserving the frozen Phase 1
data boundary.

Label convention:

    0 = legitimate
    1 = phishing
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


COMMON_REQUIRED_COLUMNS = {
    "sample_id",
    "source_dataset",
    "label",
}


class URLDatasetLoader:
    """
    Load and validate a Phase 1 URL dataset or split.

    Parameters
    ----------
    file_path:
        Path to the URL CSV file.
    """

    def __init__(
        self,
        file_path: str | Path,
    ) -> None:
        self.file_path = Path(file_path)

    def load(self) -> pd.DataFrame:
        """
        Load and validate the URL dataset.
        """

        if not self.file_path.exists():
            raise FileNotFoundError(
                f"URL dataset not found: {self.file_path}"
            )

        df = pd.read_csv(
            self.file_path
        )

        missing_columns = (
            COMMON_REQUIRED_COLUMNS
            - set(df.columns)
        )

        if missing_columns:
            raise ValueError(
                "URL dataset is missing required columns: "
                f"{sorted(missing_columns)}"
            )

        return df

    def get_feature_columns(
        self,
    ) -> list[str]:
        """
        Return numerical feature columns.

        Metadata columns are excluded:

            sample_id
            source_dataset
            label

        Any remaining numeric columns are considered candidate
        model features.
        """

        df = self.load()

        excluded_columns = {
            "sample_id",
            "source_dataset",
            "label",
        }

        feature_columns = [
            column
            for column in df.columns
            if column not in excluded_columns
            and pd.api.types.is_numeric_dtype(
                df[column]
            )
        ]

        if not feature_columns:
            raise ValueError(
                "No numerical URL feature columns were found."
            )

        return feature_columns

    def get_features_and_labels(
        self,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Return numerical URL features and labels.

        Returns
        -------
        X:
            DataFrame containing numerical URL features.

        y:
            Series containing labels.

        Label convention:

            0 = legitimate
            1 = phishing
        """

        df = self.load()

        feature_columns = (
            self.get_feature_columns()
        )

        X = df[
            feature_columns
        ].copy()

        y = (
            df["label"]
            .astype(int)
        )

        return X, y

    def get_sample_metadata(
        self,
    ) -> pd.DataFrame:
        """
        Return sample identifiers and dataset metadata.
        """

        df = self.load()

        return df[
            [
                "sample_id",
                "source_dataset",
                "label",
            ]
        ].copy()