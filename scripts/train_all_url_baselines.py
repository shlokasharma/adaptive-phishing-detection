"""
Train all Phase 2 URL baseline models.

Models trained:
    1. Feature-based Logistic Regression
    2. Feature-based Random Forest
    3. Feature-based Gradient Boosting

Important:
    - Only the Phase 1 UCI training split is used.
    - URL preprocessing is fitted only on the training data.
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

from phishing_detection.data.url_loader import URLDatasetLoader
from phishing_detection.features.url_features import URLFeatureExtractor
from phishing_detection.models.traditional_ml.url_gradient_boosting import (
    URLGradientBoostingDetector,
)
from phishing_detection.models.traditional_ml.url_logistic_regression import (
    URLLogisticRegressionDetector,
)
from phishing_detection.models.traditional_ml.url_random_forest import (
    URLRandomForestDetector,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_FILE = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "url"
    / "uci_features"
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


def train_url_model(
    model_name: str,
    model,
    extractor_name: str,
    extractor: URLFeatureExtractor,
    X_train,
    y_train,
) -> None:

    print("=" * 80)
    print(f"Training: {model_name}")
    print("=" * 80)

    start_time = time.perf_counter()

    print("Fitting URL feature preprocessing...")

    X_train_transformed = (
        extractor.fit_transform(X_train)
    )

    print(
        f"Feature matrix shape: "
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
            f"URL training split not found: {TRAIN_FILE}"
        )

    print(
        "Loading Phase 1 UCI URL training split..."
    )

    loader = URLDatasetLoader(TRAIN_FILE)

    X_train, y_train = (
        loader.get_features_and_labels()
    )

    print(
        f"Training samples: {len(X_train)}"
    )

    print(
        "Training labels:"
    )

    print(
        f"  Legitimate: "
        f"{int((y_train == 0).sum())}"
    )

    print(
        f"  Phishing: "
        f"{int((y_train == 1).sum())}"
    )

    print(
        f"Number of URL features: "
        f"{X_train.shape[1]}"
    )

    print()

    models = [
        (
            "url_logistic_regression",
            URLLogisticRegressionDetector(),
            "url_logistic_regression_feature_extractor",
        ),
        (
            "url_random_forest",
            URLRandomForestDetector(),
            "url_random_forest_feature_extractor",
        ),
        (
            "url_gradient_boosting",
            URLGradientBoostingDetector(),
            "url_gradient_boosting_feature_extractor",
        ),
    ]

    for (
        model_name,
        model,
        extractor_name,
    ) in models:

        extractor = URLFeatureExtractor()

        train_url_model(
            model_name=model_name,
            model=model,
            extractor_name=extractor_name,
            extractor=extractor,
            X_train=X_train,
            y_train=y_train,
        )

    print("=" * 80)
    print("All URL baseline models trained successfully.")
    print("=" * 80)


if __name__ == "__main__":
    main()