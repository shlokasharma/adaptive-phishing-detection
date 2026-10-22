from __future__ import annotations

from frontend.services.detection_service import (
    DetectionService,
)


def test_detection_service_initializes():

    service = DetectionService()

    assert service.email_agent is not None
    assert service.url_agent is not None
    assert service.sender_agent is not None
    assert service.orchestrator is not None
    assert service.pipeline is not None
    assert service.xai_service is not None


def test_email_analysis_returns_xai_field():

    service = DetectionService()

    result = service.analyze(
        mode="Email",
        sender="security@example.com",
        receiver="user@example.com",
        subject="Account verification required",
        body=(
            "Your account requires immediate verification. "
            "Please visit http://example-phishing.test "
            "to continue."
        ),
    )

    assert isinstance(result, dict)

    assert "explanations" in result
    assert "xai_status" in result

    assert result["xai_status"] in {
        "available",
        "error",
    }


def test_url_analysis_returns_xai_field():

    service = DetectionService()

    result = service.analyze(
        mode="URL",
        url="https://example.com",
    )

    assert isinstance(result, dict)

    assert "explanations" in result
    assert "xai_status" in result

    assert result["xai_status"] in {
        "available",
        "error",
    }