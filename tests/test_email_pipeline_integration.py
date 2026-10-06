from phishing_detection.features.email_tfidf import EmailTfidfExtractor


def test_email_training_pipeline():
    """
    Verify the complete email text -> TF-IDF pipeline using
    a small synthetic dataset.

    Full Phase 1 training is intentionally not performed inside
    the automated test suite. Full-data experiments are executed
    through the dedicated training scripts.
    """

    texts = [
        "hello this is a normal meeting message",
        "please review the attached meeting document",
        "urgent verify your account immediately",
        "click this link to confirm your password",
        "team meeting scheduled for tomorrow",
        "your account has been suspended click here",
    ]

    labels = [0, 0, 0, 1, 0, 1]

    assert len(texts) == len(labels)
    assert set(labels).issubset({0, 1})

    extractor = EmailTfidfExtractor(
        max_features=100,
        ngram_range=(1, 2),
        min_df=1,
        max_df=1.0,
        sublinear_tf=True,
    )

    features = extractor.fit_transform(texts)

    assert features.shape[0] == len(texts)
    assert features.shape[1] > 0

    assert extractor.get_num_features() == features.shape[1]