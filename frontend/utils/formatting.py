from __future__ import annotations

from typing import Any


def format_probability(value: Any) -> str:
    """Format probability/confidence values."""

    if value is None:
        return "N/A"

    try:
        numeric = float(value)

        if 0.0 <= numeric <= 1.0:
            return f"{numeric * 100:.2f}%"

        return f"{numeric:.2f}%"

    except (TypeError, ValueError):
        return str(value)


def format_number(value: Any, digits: int = 3) -> str:
    """Format a numeric value safely."""

    if value is None:
        return "N/A"

    try:
        return f"{float(value):.{digits}f}"

    except (TypeError, ValueError):
        return str(value)


def format_label(value: Any) -> str:
    """Convert internal identifiers to readable labels."""

    if value is None:
        return "N/A"

    text = str(value)

    for old, new in {
        "_": " ",
        "-": " ",
    }.items():
        text = text.replace(old, new)

    return text.strip().title()


def format_risk_level(
    phishing_probability: Any,
) -> str:
    """Return a presentation-level risk category."""

    try:
        probability = float(phishing_probability)

    except (TypeError, ValueError):
        return "Unknown"

    if probability >= 0.90:
        return "Critical"

    if probability >= 0.70:
        return "High"

    if probability >= 0.40:
        return "Moderate"

    if probability >= 0.20:
        return "Low"

    return "Minimal"


def get_first_value(
    data: dict[str, Any],
    *keys: str,
    default: Any = None,
) -> Any:
    """Return the first non-null value from a dictionary."""

    for key in keys:

        if key in data and data[key] is not None:
            return data[key]

    return default


def normalize_agent_results(
    agent_results: Any,
) -> list[dict[str, Any]]:
    """Normalize agent results to a list."""

    if agent_results is None:
        return []

    if isinstance(agent_results, dict):

        normalized = []

        for name, value in agent_results.items():

            if hasattr(value, "to_dict"):
                value = value.to_dict()

            if isinstance(value, dict):

                item = dict(value)
                item.setdefault("agent_name", name)
                normalized.append(item)

            else:

                normalized.append(
                    {
                        "agent_name": name,
                        "result": value,
                    }
                )

        return normalized

    if isinstance(agent_results, (list, tuple)):

        normalized = []

        for index, value in enumerate(agent_results):

            if hasattr(value, "to_dict"):
                value = value.to_dict()

            if isinstance(value, dict):
                normalized.append(value)

            else:
                normalized.append(
                    {
                        "agent_name": f"agent_{index + 1}",
                        "result": value,
                    }
                )

        return normalized

    return []


def normalize_evidence(
    evidence: Any,
) -> list[dict[str, Any]]:
    """Normalize evidence items."""

    if evidence is None:
        return []

    if isinstance(evidence, dict):
        evidence = list(evidence.values())

    if not isinstance(evidence, (list, tuple)):
        evidence = [evidence]

    normalized = []

    for item in evidence:

        if hasattr(item, "to_dict"):
            item = item.to_dict()

        if isinstance(item, dict):
            normalized.append(item)

        else:
            normalized.append(
                {
                    "description": str(item),
                }
            )

    return normalized


def normalize_trace(
    trace: Any,
) -> list[dict[str, Any]]:
    """Normalize decision-trace entries."""

    if trace is None:
        return []

    if isinstance(trace, dict):
        trace = [trace]

    if not isinstance(trace, (list, tuple)):
        return [
            {
                "action": "trace",
                "description": str(trace),
            }
        ]

    normalized = []

    for item in trace:

        if hasattr(item, "to_dict"):
            item = item.to_dict()

        if isinstance(item, dict):
            normalized.append(item)

        else:
            normalized.append(
                {
                    "action": "trace",
                    "description": str(item),
                }
            )

    return normalized


def normalize_explanations(
    explanations: Any,
) -> dict[str, Any]:
    """Normalize XAI output if supplied by the backend."""

    if explanations is None:
        return {}

    if hasattr(explanations, "to_dict"):
        explanations = explanations.to_dict()

    if isinstance(explanations, dict):
        return explanations

    if isinstance(explanations, list):

        return {
            f"explanation_{index + 1}": value
            for index, value in enumerate(explanations)
        }

    return {
        "explanation": explanations,
    }


def get_agent_status(
    agent_name: str,
    executed_agents: set[str],
) -> str:
    """Return a UI status for an analysis agent."""

    normalized = agent_name.lower()

    if normalized in executed_agents:
        return "Executed"

    return "Not required"


def agent_display_name(agent_name: str) -> str:
    """Return a human-readable agent name."""

    mapping = {
        "email_analysis_agent": "Email Analysis Agent",
        "url_analysis_agent": "URL Analysis Agent",
        "sender_analysis_agent": "Sender Analysis Agent",
        "orchestrator_agent": "Orchestrator Agent",
    }

    return mapping.get(
        str(agent_name).lower(),
        format_label(agent_name),
    )