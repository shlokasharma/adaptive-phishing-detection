"""
Train the Phase 2 TF-IDF + Random Forest email baseline.

This script:

1. Loads the Phase 1 email training split.
2. Fits TF-IDF only on training emails.
3. Trains Random Forest.
4. Reports training performance.
5. Saves the trained detector and TF-IDF extractor.

The validation and test sets are intentionally not used here.
They will be handled by the formal Phase 2 evaluation pipeline.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from phishing_detection.data.email_loader import EmailDatasetLoader
from phishing_detection.features.email_tfidf import EmailTfidfExtractor
from phishing_detection.models.traditional_ml.email_random_forest import (
    EmailRandomForestDetector,
)


TRAIN_FILE = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "email"
    / "train.csv"
)

MODEL_OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "baselines"
    / "outputs"
)

MODEL_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def main() -> None:
    print("=" * 70)
    print("PHASE 2 - EMAIL RANDOM FOREST BASELINE")
    print("=" * 70)

    print("\n[1/5] Loading Phase 1 training data...")

    loader = EmailDatasetLoader(
        TRAIN_FILE
    )

    texts, labels = loader.get_texts_and_labels()

    print(
        f"Training samples: {len(texts):,}"
    )

    print("\n[2/5] Fitting TF-IDF extractor...")

    extractor = EmailTfidfExtractor(
        max_features=50_000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )

    X_train = extractor.fit_transform(
        texts
    )

    print(
        "TF-IDF matrix shape: "
        f"{X_train.shape[0]:,} x "
        f"{X_train.shape[1]:,}"
    )

    print("\n[3/5] Training Random Forest...")

    detector = EmailRandomForestDetector(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42,
    )

    detector.fit(
        X_train,
        labels,
    )

    print(
        "Random Forest training completed."
    )

    print(
        "\n[4/5] Evaluating training-set performance..."
    )

    predictions = detector.predict(
        X_train
    )

    accuracy = accuracy_score(
        labels,
        predictions,
    )

    precision = precision_score(
        labels,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        labels,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        labels,
        predictions,
        zero_division=0,
    )

    print(
        f"\nAccuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1       : {f1:.4f}"
    )

    print("\nClassification Report:")

    print(
        classification_report(
            labels,
            predictions,
            target_names=[
                "legitimate",
                "phishing",
            ],
            zero_division=0,
        )
    )

    print("\n[5/5] Saving model artifacts...")

    model_path = (
        MODEL_OUTPUT_DIR
        / "email_random_forest.pkl"
    )

    tfidf_path = (
        MODEL_OUTPUT_DIR
        / "email_rf_tfidf_extractor.pkl"
    )

    with open(
        model_path,
        "wb",
    ) as file:
        pickle.dump(
            detector,
            file,
        )

    with open(
        tfidf_path,
        "wb",
    ) as file:
        pickle.dump(
            extractor,
            file,
        )

    print(
        f"Detector saved to: {model_path}"
    )

    print(
        f"TF-IDF extractor saved to: {tfidf_path}"
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "PHASE 2 EMAIL RANDOM FOREST BASELINE COMPLETE"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()