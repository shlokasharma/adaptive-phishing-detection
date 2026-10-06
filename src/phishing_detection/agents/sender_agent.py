"""
Sender Analysis Agent.

This module analyzes sender and email-header information for
potential phishing indicators.

The agent is intentionally model-independent. It produces
structured evidence from sender identity, domain consistency,
and authentication-related signals.

The agent does not make the final system-level decision.
"""

from __future__ import annotations

import re
import time
from email.utils import parseaddr
from typing import Any
from urllib.parse import urlparse

from phishing_detection.agents.base import (
    AgentContext,
    AgentExecutionMetadata,
    BaseAgent,
)
from phishing_detection.agents.evidence import Evidence
from phishing_detection.agents.result import AgentResult


class SenderAnalysisAgent(BaseAgent):
    """
    Specialized agent for sender and email-header analysis.

    The agent evaluates:

    - sender address validity
    - sender/reply-to consistency
    - sender domain consistency
    - SPF result
    - DKIM result
    - DMARC result
    - authentication failures
    """

    DEFAULT_AGENT_NAME = "sender_analysis_agent"
    DEFAULT_MODALITY = "sender"

    EMAIL_PATTERN = re.compile(
        r"^[^@\s]+@[^@\s]+\.[^@\s]+$"
    )

    AUTHENTICATION_VALUES = {
        "pass",
        "fail",
        "softfail",
        "neutral",
        "none",
        "temperror",
        "permerror",
    }

    def __init__(
        self,
        agent_name: str = DEFAULT_AGENT_NAME,
    ) -> None:
        self.agent_name = agent_name

    def get_agent_name(self) -> str:
        """Return the unique agent name."""
        return self.agent_name

    def get_modality(self) -> str:
        """Return the modality analyzed by this agent."""
        return self.DEFAULT_MODALITY

    def get_capabilities(self) -> list[str]:
        """Return capabilities provided by the agent."""
        return [
            "sender_address_analysis",
            "sender_domain_analysis",
            "reply_to_analysis",
            "authentication_analysis",
            "header_consistency_analysis",
        ]

    def can_analyze(self, context: AgentContext) -> bool:
        """
        Determine whether sender analysis can be performed.

        Sender information may be supplied either through the
        dedicated ``sender`` field or through headers.
        """

        has_sender = bool(
            context.sender
            and context.sender.strip()
        )

        has_headers = bool(
            context.headers
            and isinstance(context.headers, dict)
        )

        return has_sender or has_headers

    @staticmethod
    def _extract_email_address(value: str | None) -> str:
        """Extract an email address from a header value."""

        if not value:
            return ""

        _, address = parseaddr(value)

        return address.strip().lower()

    @staticmethod
    def _extract_domain(email_address: str) -> str:
        """Extract the domain portion of an email address."""

        if "@" not in email_address:
            return ""

        return email_address.rsplit("@", 1)[1].lower()

    @classmethod
    def _is_valid_email(cls, email_address: str) -> bool:
        """Return whether an email address has a basic valid format."""

        return bool(
            cls.EMAIL_PATTERN.match(email_address)
        )

    @staticmethod
    def _normalize_authentication_value(
        value: Any,
    ) -> str:
        """
        Normalize authentication results.

        Examples:
            "pass" -> "pass"
            "Pass" -> "pass"
            "spf=pass" -> "pass"
        """

        if value is None:
            return ""

        text = str(value).strip().lower()

        if "=" in text:
            text = text.split("=", 1)[1]

        return text.split()[0]

    @staticmethod
    def _domain_from_url(url: str | None) -> str:
        """Extract a domain from a URL if available."""

        if not url:
            return ""

        try:
            parsed = urlparse(url)

            hostname = parsed.hostname

            if hostname:
                return hostname.lower()

        except ValueError:
            pass

        return ""

    def _build_evidence(
        self,
        *,
        evidence_type: str,
        description: str,
        value: Any,
        polarity: str,
        confidence: float,
        reliability: float,
        metadata: dict[str, Any] | None = None,
    ) -> Evidence:
        """Construct a validated Evidence object."""

        return Evidence(
            evidence_type=evidence_type,
            source=self.agent_name,
            description=description,
            value=value,
            polarity=polarity,
            confidence=confidence,
            reliability=reliability,
            metadata=metadata or {},
        )

    def _analyze_sender_identity(
        self,
        context: AgentContext,
    ) -> list[Evidence]:
        """Analyze sender address and domain."""

        evidence: list[Evidence] = []

        sender = self._extract_email_address(
            context.sender
        )

        if not sender:
            sender = self._extract_email_address(
                context.headers.get("from")
            )

        if not sender:
            return evidence

        if not self._is_valid_email(sender):
            evidence.append(
                self._build_evidence(
                    evidence_type="sender_identity",
                    description=(
                        "The sender address does not match a "
                        "basic valid email-address structure."
                    ),
                    value=sender,
                    polarity="supports_phishing",
                    confidence=0.85,
                    reliability=0.90,
                )
            )

            return evidence

        domain = self._extract_domain(sender)

        evidence.append(
            self._build_evidence(
                evidence_type="sender_identity",
                description=(
                    "A syntactically valid sender address was "
                    "identified."
                ),
                value={
                    "sender": sender,
                    "domain": domain,
                },
                polarity="neutral",
                confidence=0.90,
                reliability=0.95,
            )
        )

        return evidence

    def _analyze_reply_to(
        self,
        context: AgentContext,
    ) -> list[Evidence]:
        """Analyze sender and Reply-To consistency."""

        sender = self._extract_email_address(
            context.sender
        )

        if not sender:
            sender = self._extract_email_address(
                context.headers.get("from")
            )

        reply_to = self._extract_email_address(
            context.headers.get("reply-to")
        )

        if not sender or not reply_to:
            return []

        sender_domain = self._extract_domain(sender)
        reply_domain = self._extract_domain(reply_to)

        if sender_domain != reply_domain:
            return [
                self._build_evidence(
                    evidence_type="reply_to_mismatch",
                    description=(
                        "The Reply-To domain differs from the "
                        "sender domain."
                    ),
                    value={
                        "sender_domain": sender_domain,
                        "reply_to_domain": reply_domain,
                    },
                    polarity="supports_phishing",
                    confidence=0.80,
                    reliability=0.85,
                )
            ]

        return [
            self._build_evidence(
                evidence_type="reply_to_consistency",
                description=(
                    "The Reply-To domain is consistent with "
                    "the sender domain."
                ),
                value={
                    "sender_domain": sender_domain,
                    "reply_to_domain": reply_domain,
                },
                polarity="supports_legitimate",
                confidence=0.65,
                reliability=0.80,
            )
        ]

    def _analyze_authentication(
        self,
        context: AgentContext,
    ) -> list[Evidence]:
        """Analyze SPF, DKIM, and DMARC results."""

        evidence: list[Evidence] = []

        authentication_fields = (
            "spf",
            "dkim",
            "dmarc",
        )

        for field_name in authentication_fields:
            raw_value = context.headers.get(field_name)

            if raw_value is None:
                raw_value = context.headers.get(
                    f"{field_name}-result"
                )

            result = self._normalize_authentication_value(
                raw_value
            )

            if not result:
                continue

            if result not in self.AUTHENTICATION_VALUES:
                continue

            if result == "pass":
                polarity = "supports_legitimate"
                confidence = 0.70
            elif result in {
                "fail",
                "softfail",
                "permerror",
            }:
                polarity = "supports_phishing"
                confidence = 0.80
            else:
                polarity = "neutral"
                confidence = 0.45

            evidence.append(
                self._build_evidence(
                    evidence_type=(
                        f"{field_name}_authentication"
                    ),
                    description=(
                        f"{field_name.upper()} authentication "
                        f"result: {result}."
                    ),
                    value=result,
                    polarity=polarity,
                    confidence=confidence,
                    reliability=0.85,
                )
            )

        return evidence

    def _analyze_domain_consistency(
        self,
        context: AgentContext,
    ) -> list[Evidence]:
        """
        Compare sender domain with a supplied URL domain.

        This provides a lightweight cross-modality consistency
        signal without making a final phishing decision.
        """

        sender = self._extract_email_address(
            context.sender
        )

        if not sender:
            sender = self._extract_email_address(
                context.headers.get("from")
            )

        url_domain = self._domain_from_url(
            context.url
        )

        if not sender or not url_domain:
            return []

        sender_domain = self._extract_domain(sender)

        if not sender_domain:
            return []

        if sender_domain == url_domain:
            return [
                self._build_evidence(
                    evidence_type="domain_consistency",
                    description=(
                        "The sender domain matches the domain "
                        "of the supplied URL."
                    ),
                    value={
                        "sender_domain": sender_domain,
                        "url_domain": url_domain,
                    },
                    polarity="supports_legitimate",
                    confidence=0.60,
                    reliability=0.75,
                )
            ]

        return [
            self._build_evidence(
                evidence_type="domain_mismatch",
                description=(
                    "The sender domain differs from the domain "
                    "of the supplied URL."
                ),
                value={
                    "sender_domain": sender_domain,
                    "url_domain": url_domain,
                },
                polarity="supports_phishing",
                confidence=0.70,
                reliability=0.75,
            )
        ]

    @staticmethod
    def _derive_probability(
        evidence: list[Evidence],
    ) -> float:
        """
        Derive a lightweight phishing probability from evidence.

        This is NOT the final system decision.

        The score is intentionally conservative and is only used
        to expose a standardized AgentResult to the orchestrator.
        """

        phishing_weight = 0.0
        legitimate_weight = 0.0

        for item in evidence:
            weight = (
                float(item.confidence or 0.0)
                * float(item.reliability or 0.0)
            )

            if item.supports_phishing:
                phishing_weight += weight

            elif item.supports_legitimate:
                legitimate_weight += weight

        total_weight = (
            phishing_weight
            + legitimate_weight
        )

        if total_weight == 0.0:
            return 0.5

        return round(
            phishing_weight / total_weight,
            10,
        )

    def analyze(
        self,
        context: AgentContext,
    ) -> AgentResult:
        """Analyze sender information and return AgentResult."""

        self.validate_context(context)

        if not self.can_analyze(context):
            raise ValueError(
                "SenderAnalysisAgent requires sender information "
                "or email headers."
            )

        start_time = time.perf_counter()

        evidence: list[Evidence] = []

        evidence.extend(
            self._analyze_sender_identity(context)
        )

        evidence.extend(
            self._analyze_reply_to(context)
        )

        evidence.extend(
            self._analyze_authentication(context)
        )

        evidence.extend(
            self._analyze_domain_consistency(context)
        )

        phishing_probability = self._derive_probability(
            evidence
        )

        label = int(
            phishing_probability >= 0.5
        )

        label_name = (
            "phishing"
            if label == 1
            else "legitimate"
        )

        confidence = (
            phishing_probability
            if label == 1
            else 1.0 - phishing_probability
        )

        uncertainty = round(
            1.0 - confidence,
            10,
        )

        execution_time_ms = (
            time.perf_counter() - start_time
        ) * 1000.0

        execution_metadata = AgentExecutionMetadata(
            execution_time_ms=execution_time_ms,
            model_name="sender_rule_analysis",
            additional_metadata={
                "analysis_type": "sender_and_headers",
                "evidence_count": len(evidence),
            },
        )

        return AgentResult(
            agent_name=self.agent_name,
            modality=self.DEFAULT_MODALITY,
            label=label,
            label_name=label_name,
            phishing_probability=phishing_probability,
            confidence=confidence,
            uncertainty=uncertainty,
            evidence=evidence,
            execution_metadata=execution_metadata,
            metadata={
                "analysis_type": "sender_and_headers",
                "phase": "3.6",
            },
        )