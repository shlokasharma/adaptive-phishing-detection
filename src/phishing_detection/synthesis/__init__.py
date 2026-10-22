"""
Phase 7: Evidence Synthesis & LLM Reasoning.

Provides standardized evidence collection, multi-agent conflict resolution,
cautious risk assessment, and grounded natural-language explanations without
requiring a local LLM or GPU.
"""

from __future__ import annotations

from phishing_detection.synthesis.collector import EvidenceCollector
from phishing_detection.synthesis.conflict_resolver import ConflictResolver
from phishing_detection.synthesis.explainer import DeterministicExplainer
from phishing_detection.synthesis.llm_adapter import (
    BaseLLMAdapter,
    CloudLLMAdapter,
    LLMConfig,
    NullLLMAdapter,
)
from phishing_detection.synthesis.schema import (
    ConflictReport,
    EvidenceCategory,
    EvidencePolarity,
    EvidenceStatus,
    RiskLevel,
    ScoreType,
    StandardizedEvidence,
    SynthesisAssessment,
)
from phishing_detection.synthesis.service import EvidenceSynthesisService
from phishing_detection.synthesis.synthesizer import EvidenceSynthesizer

__all__ = [
    # Schema & data structures
    "EvidenceCategory",
    "EvidencePolarity",
    "EvidenceStatus",
    "ScoreType",
    "RiskLevel",
    "StandardizedEvidence",
    "ConflictReport",
    "SynthesisAssessment",
    # Components
    "EvidenceCollector",
    "ConflictResolver",
    "DeterministicExplainer",
    # LLM Adapters
    "BaseLLMAdapter",
    "NullLLMAdapter",
    "CloudLLMAdapter",
    "LLMConfig",
    # Core Engine & Service
    "EvidenceSynthesizer",
    "EvidenceSynthesisService",
]
