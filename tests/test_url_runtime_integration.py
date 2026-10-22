from __future__ import annotations

import pandas as pd

from frontend.services.detection_service import DetectionService
from phishing_detection.agents.url_agent import URLAnalysisAgent


MODEL_PATH = "experiments/baselines/models/url_random_forest.pkl"
PREPROCESSOR_PATH = (
    "experiments/baselines/models/url_random_forest_feature_extractor.pkl"
)


def test_runtime_url_feature_builder():
    service = DetectionService()

    service.url_agent._load_artifacts()

    preprocessor = service.url_agent._preprocessor
    feature_names = list(preprocessor.feature_names)

    url = "https://example.com/login"

    features = service._build_url_features(url)

    assert isinstance(features, pd.DataFrame)
    assert features.shape[0] == 1
    assert list(features.columns) == feature_names


def test_url_agent_accepts_runtime_features():
    service = DetectionService()

    service.url_agent._load_artifacts()

    preprocessor = service.url_agent._preprocessor
    feature_names = list(preprocessor.feature_names)

    url = "https://example.com/login"

    runtime_features = service._build_url_features(url)

    assert isinstance(runtime_features, pd.DataFrame)
    assert runtime_features.shape == (1, len(feature_names))
    assert list(runtime_features.columns) == feature_names

    context_metadata = {
        "url_features": runtime_features,
    }

    assert isinstance(context_metadata["url_features"], pd.DataFrame)
    assert list(context_metadata["url_features"].columns) == feature_names


def test_runtime_url_features_are_numeric():
    service = DetectionService()

    features = service._build_url_features(
        "https://example.com/login?verify=account"
    )

    assert isinstance(features, pd.DataFrame)
    assert features.shape[0] == 1

    for column in features.columns:
        assert pd.api.types.is_numeric_dtype(features[column])


def test_runtime_url_feature_builder_is_deterministic():
    service = DetectionService()

    url = "https://example.com/login?verify=account"

    first = service._build_url_features(url)
    second = service._build_url_features(url)

    pd.testing.assert_frame_equal(first, second)