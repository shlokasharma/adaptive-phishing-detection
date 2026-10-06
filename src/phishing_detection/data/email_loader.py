"""
Utilities for loading Phase 1 email datasets.

The Phase 1 email split contains the following canonical columns:

    sample_id
    source_dataset
    label
    clean_text

The clean_text field is used as the input representation for
Phase 2 email-text baseline models.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "sample_id",
    "source_dataset",
    "label",
    "clean_text",
}


class EmailDatasetLoader:
    """
    Load a Phase 1 email dataset or split.
    """

    def __init__(self, file_path: str | Path) -> None:
        self.file_path = Path(file_path)

    def load(self) -> pd.DataFrame:
        """
        Load and validate the Phase 1 email dataset.
        """

        if not self.file_path.exists():
            raise FileNotFoundError(
                f"Email dataset not found: {self.file_path}"
            )

        df = pd.read_csv(self.file_path)

        missing_columns = REQUIRED_COLUMNS - set(df.columns)

        if missing_columns:
            raise ValueError(
                "Email dataset is missing required columns: "
                f"{sorted(missing_columns)}"
            )

        return df

    def get_texts_and_labels(
        self,
    ) -> tuple[list[str], list[int]]:
        """
        Return clean email text and phishing labels.

        Label convention:
            0 = legitimate
            1 = phishing
        """

        df = self.load()

        texts = (
            df["clean_text"]
            .fillna("")
            .astype(str)
            .tolist()
        )

        labels = (
            df["label"]
            .astype(int)
            .tolist()
        )

        return texts, labels