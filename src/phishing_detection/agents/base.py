"""
Common architecture for specialized phishing detection agents.

All Phase 3 agents must implement the interface defined in this module.

The interface is intentionally model-agnostic. An agent may internally
use a machine-learning model, rule-based analysis, external evidence,
or a combination of these mechanisms.

The goal is to provide a stable contract for the later orchestrator,
explainability layer, adaptive evidence selection, and reinforcement
learning components.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class AgentContext:
    """
    Input context supplied to an analysis agent.

    Parameters
    ----------
    input_text:
        Primary textual input available to the agent.

    url:
        Optional URL associated with the input.

    sender:
        Optional sender information.

    headers:
        Optional email/header metadata.

    metadata:
        Additional arbitrary information that may be useful to the agent.
    """

    input_text: str = ""
    url: str | None = None
    sender: str | None = None
    headers: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentExecutionMetadata:
    """
    Metadata describing an agent execution.
    """

    execution_time_ms: float = 0.0
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    model_name: str | None = None
    additional_metadata: dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    """
    Abstract base class for all Phase 3 phishing detection agents.

    Every specialized agent must provide:

    1. A unique agent name.
    2. A modality.
    3. An analysis method.
    4. A capability description.

    The interface deliberately avoids assumptions about the underlying
    machine-learning model so that future agents can use traditional ML,
    transformers, LLMs, threat intelligence, browser analysis, or rules.
    """

    @abstractmethod
    def get_agent_name(self) -> str:
        """
        Return the unique name of the agent.
        """
        raise NotImplementedError

    @abstractmethod
    def get_modality(self) -> str:
        """
        Return the modality analyzed by the agent.

        Examples
        --------
        "email"
        "url"
        "sender"
        "webpage"
        "threat_intelligence"
        """
        raise NotImplementedError

    @abstractmethod
    def get_capabilities(self) -> list[str]:
        """
        Return a list describing the capabilities of the agent.
        """
        raise NotImplementedError

    @abstractmethod
    def analyze(self, context: AgentContext) -> Any:
        """
        Analyze the supplied context.

        Concrete agents should return a structured result object.
        """
        raise NotImplementedError

    def can_analyze(self, context: AgentContext) -> bool:
        """
        Determine whether the agent has sufficient input to execute.

        The default implementation allows execution for every context.
        Specialized agents may override this method when they require
        particular input fields.
        """
        del context
        return True

    def validate_context(self, context: AgentContext) -> None:
        """
        Validate that the supplied context is an AgentContext instance.
        """
        if not isinstance(context, AgentContext):
            raise TypeError(
                "context must be an instance of AgentContext."
            )

    def get_execution_metadata(
        self,
        *,
        execution_time_ms: float = 0.0,
        model_name: str | None = None,
        additional_metadata: dict[str, Any] | None = None,
    ) -> AgentExecutionMetadata:
        """
        Construct standardized execution metadata.
        """
        return AgentExecutionMetadata(
            execution_time_ms=float(execution_time_ms),
            model_name=model_name,
            additional_metadata=additional_metadata or {},
        )