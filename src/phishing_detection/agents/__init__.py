"""
Specialized phishing detection agents.

Phase 3 introduces agent-based analysis on top of the
Phase 2 baseline detection models.
"""

from phishing_detection.agents.base import (
    AgentContext,
    AgentExecutionMetadata,
    BaseAgent,
)
from phishing_detection.agents.evidence import Evidence
from phishing_detection.agents.result import AgentResult
from phishing_detection.agents.email_agent import EmailAnalysisAgent
from phishing_detection.agents.url_agent import URLAnalysisAgent
from phishing_detection.agents.sender_agent import SenderAnalysisAgent
from phishing_detection.agents.confidence import (
    ConfidenceAssessment,
    assess_agent_result,
    calculate_aggregate_evidence_strength,
    calculate_confidence,
    calculate_evidence_strength,
    calculate_uncertainty,
)

__all__ = [
    "AgentContext",
    "AgentExecutionMetadata",
    "BaseAgent",
    "Evidence",
    "AgentResult",
    "EmailAnalysisAgent",
    "URLAnalysisAgent",
    "SenderAnalysisAgent",
    "ConfidenceAssessment",
    "assess_agent_result",
    "calculate_aggregate_evidence_strength",
    "calculate_confidence",
    "calculate_evidence_strength",
    "calculate_uncertainty",
]