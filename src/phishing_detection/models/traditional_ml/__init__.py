"""
Traditional machine-learning phishing detection models.
"""

from phishing_detection.models.traditional_ml.email_linear_svm import (
    EmailLinearSVMDetector,
)
from phishing_detection.models.traditional_ml.email_logistic_regression import (
    EmailLogisticRegressionDetector,
)
from phishing_detection.models.traditional_ml.email_random_forest import (
    EmailRandomForestDetector,
)
from phishing_detection.models.traditional_ml.url_gradient_boosting import (
    URLGradientBoostingDetector,
)
from phishing_detection.models.traditional_ml.url_logistic_regression import (
    URLLogisticRegressionDetector,
)
from phishing_detection.models.traditional_ml.url_random_forest import (
    URLRandomForestDetector,
)

__all__ = [
    "EmailLinearSVMDetector",
    "EmailLogisticRegressionDetector",
    "EmailRandomForestDetector",
    "URLGradientBoostingDetector",
    "URLLogisticRegressionDetector",
    "URLRandomForestDetector",
]