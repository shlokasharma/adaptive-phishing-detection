"""
Structured evidence schema for Phase 3 phishing detection agents.

Every specialized agent should represent its findings using the
Evidence object defined in this module.

The schema is intentionally generic so that it can represent evidence
from machine-learning models, rule-based analysis, sender inspection,
URL analysis, threat intelligence, webpage analysis, and future
LLM-based reasoning.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


VALID_EVIDENCE_POLARITIES = {
    "supports_phishing",
    "supports_legitimate",
    "neutral",
}


@dataclass
class Evidence:
    """
    Standardized representation of a single piece of evidence.

    Parameters
    ----------
    evidence_type:
        Category of evidence.

        Examples:
            "email_text"
            "url"
            "sender"
            "header"
            "model_prediction"
            "threat_intelligence"

    source:
        Name of the agent or component that produced the evidence.

    description:
        Human-readable explanation of what the evidence represents.

    value:
        Optional machine-readable value associated with the evidence.

    polarity:
        Direction of the evidence:

            supports_phishing
            supports_legitimate
            neutral

    confidence:
        Confidence of the component producing this evidence.

    reliability:
        Estimated reliability of the evidence source.

    metadata:
        Additional structured information.
    """

    evidence_type: str
    source: str
    description: str
    value: Any = None
    polarity: str = "neutral"
    confidence: float | None = None
    reliability: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate the evidence object."""

        if not self.evidence_type.strip():
            raise ValueError(
                "evidence_type must not be empty."
            )

        if not self.source.strip():
            raise ValueError(
                "source must not be empty."
            )

        if not self.description.strip():
            raise ValueError(
                "description must not be empty."
            )

        if self.polarity not in VALID_EVIDENCE_POLARITIES:
            raise ValueError(
                "Invalid evidence polarity: "
                f"{self.polarity!r}. Expected one of "
                f"{sorted(VALID_EVIDENCE_POLARITIES)}."
            )

        self._validate_probability(
            self.confidence,
            "confidence",
        )

        self._validate_probability(
            self.reliability,
            "reliability",
        )

        if not isinstance(self.metadata, dict):
            raise TypeError(
                "metadata must be a dictionary."
            )

    @staticmethod
    def _validate_probability(
        value: float | None,
        field_name: str,
    ) -> None:
        """Validate an optional probability-like value."""

        if value is None:
            return

        if not isinstance(value, (int, float)):
            raise TypeError(
                f"{field_name} must be a number or None."
            )

        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(
                f"{field_name} must be between 0 and 1."
            )

    def to_dict(self) -> dict[str, Any]:
        """
        Convert evidence into a serializable dictionary.
        """
        return asdict(self)

    @property
    def supports_phishing(self) -> bool:
        """Return True when the evidence supports phishing."""
        return self.polarity == "supports_phishing"

    @property
    def supports_legitimate(self) -> bool:
        """Return True when the evidence supports legitimate."""
        return self.polarity == "supports_legitimate"

    @property
    def is_neutral(self) -> bool:
        """Return True when the evidence is neutral."""
        return self.polarity == "neutral"