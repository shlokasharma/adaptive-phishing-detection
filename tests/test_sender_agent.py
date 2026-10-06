from __future__ import annotations

import pytest

from phishing_detection.agents.base import AgentContext
from phishing_detection.agents.result import AgentResult
from phishing_detection.agents.sender_agent import SenderAnalysisAgent


def create_agent() -> SenderAnalysisAgent:
    return SenderAnalysisAgent()


def test_agent_metadata():
    agent = create_agent()

    assert (
        agent.get_agent_name()
        == "sender_analysis_agent"
    )

    assert agent.get_modality() == "sender"

    capabilities = agent.get_capabilities()

    assert "sender_address_analysis" in capabilities
    assert "sender_domain_analysis" in capabilities
    assert "reply_to_analysis" in capabilities
    assert "authentication_analysis" in capabilities


def test_can_analyze_with_sender():
    agent = create_agent()

    context = AgentContext(
        sender="security@example.com"
    )

    assert agent.can_analyze(context) is True


def test_can_analyze_with_headers():
    agent = create_agent()

    context = AgentContext(
        headers={
            "from": "security@example.com"
        }
    )

    assert agent.can_analyze(context) is True


def test_cannot_analyze_without_sender_information():
    agent = create_agent()

    context = AgentContext()

    assert agent.can_analyze(context) is False


def test_valid_sender_generates_neutral_evidence():
    agent = create_agent()

    context = AgentContext(
        sender="security@example.com"
    )

    result = agent.analyze(context)

    assert isinstance(result, AgentResult)

    assert result.evidence_count >= 1

    sender_evidence = [
        item
        for item in result.evidence
        if item.evidence_type == "sender_identity"
    ]

    assert len(sender_evidence) == 1
    assert (
        sender_evidence[0].polarity
        == "neutral"
    )


def test_invalid_sender_supports_phishing():
    agent = create_agent()

    context = AgentContext(
        sender="invalid-sender"
    )

    result = agent.analyze(context)

    assert result.evidence_count >= 1

    assert any(
        item.supports_phishing
        for item in result.evidence
    )


def test_reply_to_domain_mismatch():
    agent = create_agent()

    context = AgentContext(
        sender="security@example.com",
        headers={
            "reply-to": "support@malicious.example",
        },
    )

    result = agent.analyze(context)

    mismatch = [
        item
        for item in result.evidence
        if item.evidence_type == "reply_to_mismatch"
    ]

    assert len(mismatch) == 1
    assert mismatch[0].supports_phishing is True


def test_reply_to_domain_consistency():
    agent = create_agent()

    context = AgentContext(
        sender="security@example.com",
        headers={
            "reply-to": "support@example.com",
        },
    )

    result = agent.analyze(context)

    consistency = [
        item
        for item in result.evidence
        if item.evidence_type
        == "reply_to_consistency"
    ]

    assert len(consistency) == 1

    assert (
        consistency[0].supports_legitimate
        is True
    )


def test_spf_pass_supports_legitimate():
    agent = create_agent()

    context = AgentContext(
        sender="security@example.com",
        headers={
            "spf": "pass",
        },
    )

    result = agent.analyze(context)

    spf = [
        item
        for item in result.evidence
        if item.evidence_type
        == "spf_authentication"
    ]

    assert len(spf) == 1
    assert spf[0].supports_legitimate is True


def test_spf_fail_supports_phishing():
    agent = create_agent()

    context = AgentContext(
        sender="security@example.com",
        headers={
            "spf": "fail",
        },
    )

    result = agent.analyze(context)

    spf = [
        item
        for item in result.evidence
        if item.evidence_type
        == "spf_authentication"
    ]

    assert len(spf) == 1
    assert spf[0].supports_phishing is True


def test_dkim_pass_is_detected():
    agent = create_agent()

    context = AgentContext(
        sender="security@example.com",
        headers={
            "dkim": "pass",
        },
    )

    result = agent.analyze(context)

    assert any(
        item.evidence_type
        == "dkim_authentication"
        for item in result.evidence
    )


def test_dmarc_fail_supports_phishing():
    agent = create_agent()

    context = AgentContext(
        sender="security@example.com",
        headers={
            "dmarc": "fail",
        },
    )

    result = agent.analyze(context)

    dmarc = [
        item
        for item in result.evidence
        if item.evidence_type
        == "dmarc_authentication"
    ]

    assert len(dmarc) == 1
    assert dmarc[0].supports_phishing is True


def test_sender_and_url_domain_match():
    agent = create_agent()

    context = AgentContext(
        sender="security@example.com",
        url="https://example.com/login",
    )

    result = agent.analyze(context)

    evidence = [
        item
        for item in result.evidence
        if item.evidence_type
        == "domain_consistency"
    ]

    assert len(evidence) == 1
    assert evidence[0].supports_legitimate is True


def test_sender_and_url_domain_mismatch():
    agent = create_agent()

    context = AgentContext(
        sender="security@example.com",
        url="https://malicious.example/login",
    )

    result = agent.analyze(context)

    evidence = [
        item
        for item in result.evidence
        if item.evidence_type
        == "domain_mismatch"
    ]

    assert len(evidence) == 1
    assert evidence[0].supports_phishing is True


def test_agent_returns_standardized_result():
    agent = create_agent()

    context = AgentContext(
        sender="security@example.com",
        headers={
            "spf": "pass",
            "dkim": "pass",
            "dmarc": "pass",
        },
    )

    result = agent.analyze(context)

    assert isinstance(result, AgentResult)

    assert (
        result.agent_name
        == "sender_analysis_agent"
    )

    assert result.modality == "sender"

    assert 0.0 <= result.phishing_probability <= 1.0
    assert 0.0 <= result.confidence <= 1.0
    assert 0.0 <= result.uncertainty <= 1.0


def test_execution_metadata_is_generated():
    agent = create_agent()

    context = AgentContext(
        sender="security@example.com"
    )

    result = agent.analyze(context)

    assert result.execution_metadata is not None

    assert (
        result.execution_metadata.model_name
        == "sender_rule_analysis"
    )

    assert (
        result.execution_metadata.execution_time_ms
        >= 0.0
    )


def test_authentication_result_with_equals_format():
    agent = create_agent()

    context = AgentContext(
        sender="security@example.com",
        headers={
            "spf-result": "spf=pass",
        },
    )

    result = agent.analyze(context)

    assert any(
        item.evidence_type
        == "spf_authentication"
        for item in result.evidence
    )


def test_analyze_without_sender_raises_error():
    agent = create_agent()

    context = AgentContext()

    with pytest.raises(
        ValueError,
        match="requires sender information",
    ):
        agent.analyze(context)