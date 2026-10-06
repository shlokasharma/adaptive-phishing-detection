import pytest

from phishing_detection.features.email_tfidf import EmailTfidfExtractor


def test_tfidf_fit_transform():
    texts = [
        "Your account has been suspended click the link",
        "Meeting scheduled for tomorrow at ten",
        "Verify your account immediately",
    ]

    extractor = EmailTfidfExtractor(
        max_features=100,
        ngram_range=(1, 2),
        min_df=1,
    )

    features = extractor.fit_transform(texts)

    assert features.shape[0] == len(texts)
    assert features.shape[1] > 0
    assert extractor.get_num_features() == features.shape[1]


def test_tfidf_transform_after_fit():
    train_texts = [
        "verify your account",
        "meeting tomorrow",
        "click the suspicious link",
    ]

    validation_texts = [
        "verify account immediately",
        "meeting scheduled tomorrow",
    ]

    extractor = EmailTfidfExtractor(
        max_features=100,
        ngram_range=(1, 2),
        min_df=1,
    )

    train_features = extractor.fit_transform(train_texts)
    validation_features = extractor.transform(validation_texts)

    assert train_features.shape[1] == validation_features.shape[1]
    assert validation_features.shape[0] == len(validation_texts)


def test_transform_before_fit_is_rejected():
    extractor = EmailTfidfExtractor(
        max_features=100,
        min_df=1,
    )

    with pytest.raises(RuntimeError):
        extractor.transform(["test email"])


def test_feature_names_are_available_after_fit():
    texts = [
        "phishing email",
        "legitimate email",
    ]

    extractor = EmailTfidfExtractor(
        max_features=100,
        min_df=1,
    )

    extractor.fit(texts)

    feature_names = extractor.get_feature_names()

    assert isinstance(feature_names, list)
    assert len(feature_names) > 0