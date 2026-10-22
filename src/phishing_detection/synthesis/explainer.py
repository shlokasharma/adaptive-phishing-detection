"""
Deterministic Natural-Language Explanation Generator for Phase 7 (Evidence Synthesis & LLM Reasoning).

Generates grounded, structured explanations from synthesized evidence and conflict reports
using lightweight Python logic and templates.
NO LOCAL LLM, NO GPU, NO API KEY, AND NO EXTERNAL NETWORK CALLS REQUIRED.
"""

from __future__ import annotations

import re
from typing import Sequence

from phishing_detection.synthesis.schema import (
    ConflictReport,
    EvidenceCategory,
    EvidencePolarity,
    RiskLevel,
    StandardizedEvidence,
)


def _sanitize_text_for_display(text: str, max_length: int = 120) -> str:
    """Sanitize user-provided text to prevent markdown injection or control character breaks."""
    if not text:
        return ""
    # Strip dangerous characters and markdown control injections
    clean = re.sub(r"[\r\n\t]+", " ", str(text)).strip()
    clean = re.sub(r"[`*_{}\[\]<>]", "", clean)
    if len(clean) > max_length:
        clean = clean[:max_length] + "..."
    return clean


class DeterministicExplainer:
    """
    Generates natural-language reasoning explanations strictly grounded in observed evidence.
    """

    def generate_explanation(
        self,
        final_verdict: str,
        risk_level: RiskLevel | str,
        confidence: float,
        uncertainty: float,
        evidence_items: Sequence[StandardizedEvidence],
        conflict_report: ConflictReport,
        missing_components: Sequence[str] | None = None,
        failed_components: Sequence[dict] | None = None,
    ) -> tuple[str, str]:
        """
        Generate a concise summary and a comprehensive structured explanation.

        Returns
        -------
        tuple[str, str]
            (short_summary, full_explanation_markdown)
        """
        risk_str = risk_level.value if isinstance(risk_level, RiskLevel) else str(risk_level)
        conf_pct = round(confidence * 100.0, 1)

        phish_items = [e for e in evidence_items if e.polarity == EvidencePolarity.SUPPORTS_PHISHING]
        legit_items = [e for e in evidence_items if e.polarity == EvidencePolarity.SUPPORTS_LEGITIMATE]
        inconclusive_items = [
            e for e in evidence_items
            if e.polarity in (EvidencePolarity.INCONCLUSIVE, EvidencePolarity.NEUTRAL)
        ]

        # 1. Summary
        if final_verdict == "phishing":
            summary = (
                f"Classified as PHISHING ({risk_str.upper()} risk, {conf_pct}% confidence). "
                f"Supported by {len(phish_items)} malicious indicator(s)."
            )
        elif final_verdict == "legitimate":
            summary = (
                f"Classified as LEGITIMATE ({risk_str.upper()} risk, {conf_pct}% confidence). "
                f"Supported by {len(legit_items)} benign indicator(s)."
            )
        else:
            summary = (
                f"Classified as UNCERTAIN ({risk_str.upper()} risk, {conf_pct}% confidence). "
                f"Evidence is contradictory or insufficient for definitive classification."
            )

        # 2. Detailed Explanation Sections
        sections: list[str] = []

        # Section A: Conclusion Overview
        sections.append(f"### Assessment: {final_verdict.capitalize()} ({risk_str.capitalize()} Risk)")
        sections.append(
            f"The adaptive phishing detection system evaluated the input with **{conf_pct}% confidence** "
            f"(uncertainty: {uncertainty:.1%})."
        )

        # Section B: Key Influential Findings
        sections.append("#### Key Observed Findings")
        if phish_items:
            sections.append("**Indicators supporting phishing:**")
            for item in phish_items[:4]:  # Top 4
                finding = _sanitize_text_for_display(item.finding)
                score_str = f" [score: {item.score:.2f}]" if item.score is not None else ""
                sections.append(f"- **{item.source}** ({item.category.value}): {finding}{score_str}")
        if legit_items:
            sections.append("**Indicators supporting legitimate:**")
            for item in legit_items[:4]:
                finding = _sanitize_text_for_display(item.finding)
                score_str = f" [score: {item.score:.2f}]" if item.score is not None else ""
                sections.append(f"- **{item.source}** ({item.category.value}): {finding}{score_str}")
        if not phish_items and not legit_items:
            sections.append("- No strong directional evidence was observed across executed components.")

        # Section C: Agreement and Conflict Analysis
        sections.append("#### Component Agreement & Conflict Analysis")
        if conflict_report.has_conflict:
            sections.append(f"- **Conflict Detected**: {conflict_report.description}")
            sections.append(f"- **Resolution Strategy**: {conflict_report.resolution_strategy}")
            sections.append(f"- **Rationale**: {conflict_report.resolution_rationale}")
            if conflict_report.confidence_discount > 0:
                sections.append(
                    f"- **Confidence Adjustment**: Applied a {conflict_report.confidence_discount:.1%} "
                    "discount to account for inter-component disagreement."
                )
        else:
            sections.append(
                "- **Consensus**: Executed detection components concurred without contradictory predictions."
            )

        # Section D: Limitations, Missing Modalities & Component Failures
        limitations: list[str] = []
        if missing_components:
            for mc in missing_components:
                limitations.append(f"Unavailable modality / component: {mc}")
        if failed_components:
            for fc in failed_components:
                limitations.append(f"Component execution error: {fc.get('component')} ({fc.get('reason')})")
        if uncertainty > 0.30:
            limitations.append(f"Residual uncertainty is elevated ({uncertainty:.1%}).")

        sections.append("#### Uncertainty & Limitations")
        if limitations:
            for lim in limitations:
                sections.append(f"- {lim}")
        else:
            sections.append("- All required modalities were analyzed; no component failures were encountered.")

        full_explanation = "\n\n".join(sections)
        return summary, full_explanation
