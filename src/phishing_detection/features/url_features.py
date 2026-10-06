"""
Preprocessing utilities for numerical URL features.

This module provides a leakage-safe numerical preprocessing pipeline
for Phase 2 URL baseline models.

The preprocessing object must be fitted only on training data.
Validation and test data are transformed using the parameters learned
from the training data.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


class URLFeatureExtractor:
    """
    Leakage-safe preprocessing for URL numerical features.

    Pipeline:

        Missing-value imputation
                ↓
        Standardization

    Notes
    -----
    Standardization is useful for distance/linear models such as
    Logistic Regression. Tree-based models are less sensitive to
    feature scaling, but using a common representation keeps the
    pipeline consistent and reproducible.
    """

    def __init__(
        self,
    ) -> None:
        self.imputer = SimpleImputer(
            strategy="median"
        )

        self.scaler = StandardScaler()

        self.feature_names: list[str] = []

        self._is_fitted = False

    def fit(
        self,
        X: pd.DataFrame,
    ) -> "URLFeatureExtractor":
        """
        Learn preprocessing parameters from training data only.
        """

        if not isinstance(
            X,
            pd.DataFrame,
        ):
            raise TypeError(
                "X must be a pandas DataFrame."
            )

        self.feature_names = (
            X.columns.tolist()
        )

        imputed = (
            self.imputer.fit_transform(X)
        )

        self.scaler.fit(
            imputed
        )

        self._is_fitted = True

        return self

    def transform(
        self,
        X: pd.DataFrame,
    ):
        """
        Transform URL features using parameters learned during fit().
        """

        self._check_fitted()

        self._validate_columns(X)

        imputed = (
            self.imputer.transform(X)
        )

        return self.scaler.transform(
            imputed
        )

    def fit_transform(
        self,
        X: pd.DataFrame,
    ):
        """
        Fit preprocessing on training data and transform it.
        """

        self.fit(X)

        return self.transform(X)

    def get_feature_names(
        self,
    ) -> list[str]:
        """
        Return the learned feature names.
        """

        self._check_fitted()

        return list(
            self.feature_names
        )

    def get_num_features(
        self,
    ) -> int:
        """
        Return the number of URL features.
        """

        self._check_fitted()

        return len(
            self.feature_names
        )

    def _validate_columns(
        self,
        X: pd.DataFrame,
    ) -> None:
        """
        Ensure transformed data has the same feature columns as training.
        """

        if (
            X.columns.tolist()
            != self.feature_names
        ):
            raise ValueError(
                "URL feature columns do not match "
                "the columns observed during fit()."
            )

    def _check_fitted(
        self,
    ) -> None:
        """
        Ensure the extractor has been fitted.
        """

        if not self._is_fitted:
            raise RuntimeError(
                "URLFeatureExtractor must be fitted "
                "before transform()."
            )