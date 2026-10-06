"""
TF-IDF feature extraction for phishing email text.

This module provides a reusable, leakage-safe TF-IDF feature extractor
for Phase 2 email classification experiments.
"""

from __future__ import annotations

from typing import Iterable

from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer


class EmailTfidfExtractor:
    """
    Convert email text into TF-IDF feature representations.

    The vectorizer must be fitted only on the training data. Validation
    and test data are transformed using the vocabulary learned from the
    training set.
    """

    def __init__(
        self,
        max_features: int = 50_000,
        ngram_range: tuple[int, int] = (1, 2),
        min_df: int = 2,
        max_df: float = 0.95,
        sublinear_tf: bool = True,
    ) -> None:
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=sublinear_tf,
            strip_accents="unicode",
        )

        self._is_fitted = False

    def fit(self, texts: Iterable[str]) -> "EmailTfidfExtractor":
        """
        Learn the TF-IDF vocabulary and statistics from training text.
        """

        self.vectorizer.fit(texts)
        self._is_fitted = True

        return self

    def transform(self, texts: Iterable[str]) -> csr_matrix:
        """
        Transform text using the vocabulary learned during fit().
        """

        if not self._is_fitted:
            raise RuntimeError(
                "EmailTfidfExtractor must be fitted before transform()."
            )

        return self.vectorizer.transform(texts)

    def fit_transform(self, texts: Iterable[str]) -> csr_matrix:
        """
        Fit the vectorizer on training text and return the transformed data.
        """

        features = self.vectorizer.fit_transform(texts)
        self._is_fitted = True

        return features

    def get_feature_names(self) -> list[str]:
        """
        Return the learned TF-IDF feature names.
        """

        if not self._is_fitted:
            raise RuntimeError(
                "EmailTfidfExtractor must be fitted before retrieving "
                "feature names."
            )

        return self.vectorizer.get_feature_names_out().tolist()

    def get_num_features(self) -> int:
        """
        Return the number of learned TF-IDF features.
        """

        if not self._is_fitted:
            raise RuntimeError(
                "EmailTfidfExtractor must be fitted before retrieving "
                "the number of features."
            )

        return len(self.vectorizer.vocabulary_)