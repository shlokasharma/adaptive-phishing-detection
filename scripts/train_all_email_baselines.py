"""
Train all Phase 2 email baseline models.

Models trained:
    1. TF-IDF + Logistic Regression
    2. TF-IDF + Linear SVM
    3. TF-IDF + Random Forest

Important:
    - Only the Phase 1 training split is used.
    - TF-IDF is fitted only on the training text.
    - No validation or test data is used in this script.
    - Trained models and feature extractors are saved as artifacts.

Label convention:
    0 = legitimate
    1 = phishing
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path

from phishing_detection.data.email_loader import EmailDatasetLoader
from phishing_detection.features.email_tfidf import EmailTfidfExtractor
from phishing_detection.models.traditional_ml.email_linear_svm import (
    EmailLinearSVMDetector,
)
from phishing_detection.models.traditional_ml.email_logistic_regression import (
    EmailLogisticRegressionDetector,
)
from phishing_detection.models.traditional_ml.email_random_forest import (
    EmailRandomForestDetector,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_FILE = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "email"
    / "train.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "baselines"
    / "models"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def save_pickle(
    object_to_save: object,
    output_path: Path,
) -> None:
    with output_path.open("wb") as file:
        pickle.dump(
            object_to_save,
            file,
            protocol=pickle.HIGHEST_PROTOCOL,
        )


def train_email_model(
    model_name: str,
    model,
    extractor_name: str,
    extractor: EmailTfidfExtractor,
    X_train,
    y_train,
) -> None:

    print("=" * 80)
    print(f"Training: {model_name}")
    print("=" * 80)

    start_time = time.perf_counter()

    print("Fitting TF-IDF extractor...")
    X_train_transformed = extractor.fit_transform(X_train)

    print(
        f"TF-IDF matrix shape: "
        f"{X_train_transformed.shape}"
    )

    print("Training classifier...")

    model.fit(
        X_train_transformed,
        y_train,
    )

    elapsed = time.perf_counter() - start_time

    model_path = OUTPUT_DIR / f"{model_name}.pkl"
    extractor_path = OUTPUT_DIR / f"{extractor_name}.pkl"

    save_pickle(
        model,
        model_path,
    )

    save_pickle(
        extractor,
        extractor_path,
    )

    print(f"Saved model: {model_path}")
    print(f"Saved extractor: {extractor_path}")
    print(f"Training time: {elapsed:.2f} seconds")
    print()


def main() -> None:

    if not TRAIN_FILE.exists():
        raise FileNotFoundError(
            f"Email training split not found: {TRAIN_FILE}"
        )

    print("Loading Phase 1 email training split...")
    loader = EmailDatasetLoader(TRAIN_FILE)

    X_train, y_train = (
        loader.get_texts_and_labels()
    )

    print(
        f"Training samples: {len(X_train)}"
    )

    print(
        "Training labels:"
    )

    print(
        f"  Legitimate: "
        f"{sum(label == 0 for label in y_train)}"
    )

    print(
        f"  Phishing: "
        f"{sum(label == 1 for label in y_train)}"
    )

    print()

    models = [
        (
            "email_logistic_regression",
            EmailLogisticRegressionDetector(),
            "email_logistic_regression_tfidf_extractor",
        ),
        (
            "email_linear_svm",
            EmailLinearSVMDetector(),
            "email_linear_svm_tfidf_extractor",
        ),
        (
            "email_random_forest",
            EmailRandomForestDetector(),
            "email_random_forest_tfidf_extractor",
        ),
    ]

    for (
        model_name,
        model,
        extractor_name,
    ) in models:

        extractor = EmailTfidfExtractor()

        train_email_model(
            model_name=model_name,
            model=model,
            extractor_name=extractor_name,
            extractor=extractor,
            X_train=X_train,
            y_train=y_train,
        )

    print("=" * 80)
    print("All email baseline models trained successfully.")
    print("=" * 80)


if __name__ == "__main__":
    main()