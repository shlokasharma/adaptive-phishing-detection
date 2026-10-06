from pathlib import Path

import pandas as pd
import pytest

from phishing_detection.data.email_loader import EmailDatasetLoader


def test_email_loader_reads_phase1_dataset(tmp_path: Path):
    dataset = pd.DataFrame(
        {
            "sample_id": ["sample_1", "sample_2"],
            "source_dataset": ["test", "test"],
            "label": [0, 1],
            "clean_text": [
                "Meeting scheduled for tomorrow",
                "Click this suspicious link immediately",
            ],
        }
    )

    file_path = tmp_path / "email.csv"
    dataset.to_csv(file_path, index=False)

    loader = EmailDatasetLoader(file_path)

    texts, labels = loader.get_texts_and_labels()

    assert len(texts) == 2
    assert len(labels) == 2
    assert labels == [0, 1]
    assert "Meeting" in texts[0]


def test_missing_dataset_is_rejected(tmp_path: Path):
    file_path = tmp_path / "missing.csv"

    loader = EmailDatasetLoader(file_path)

    with pytest.raises(FileNotFoundError):
        loader.load()


def test_missing_required_column_is_rejected(tmp_path: Path):
    dataset = pd.DataFrame(
        {
            "sample_id": ["sample_1"],
            "label": [0],
        }
    )

    file_path = tmp_path / "invalid.csv"
    dataset.to_csv(file_path, index=False)

    loader = EmailDatasetLoader(file_path)

    with pytest.raises(ValueError):
        loader.load()