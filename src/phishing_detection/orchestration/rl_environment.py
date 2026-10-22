"""
Reinforcement Learning Environment for Adaptive Phishing Orchestration (Phase 6).

Provides an episodic Markov Decision Process (MDP) environment for training
and evaluating orchestration policies against multi-agent phishing detection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from phishing_detection.agents.base import AgentContext, BaseAgent
from phishing_detection.agents.email_agent import EmailAnalysisAgent
from phishing_detection.agents.result import AgentResult
from phishing_detection.agents.sender_agent import SenderAnalysisAgent
from phishing_detection.agents.url_agent import URLAnalysisAgent
from phishing_detection.orchestration.rl_actions import (
    NUM_ACTIONS,
    OrchestrationAction,
    compute_action_mask,
    get_valid_actions,
)
from phishing_detection.orchestration.rl_reward import (
    RewardConfig,
    calculate_step_reward,
    calculate_terminal_reward,
)
from phishing_detection.orchestration.rl_state import (
    STATE_DIMENSION,
    RLState,
    encode_state,
)

logger = logging.getLogger(__name__)


@dataclass
class StepResult:
    """Result of an environment step."""

    state: np.ndarray
    reward: float
    done: bool
    info: dict[str, Any] = field(default_factory=dict)


class PhishingRLEnvironment:
    """
    Episodic RL environment for adaptive phishing detection orchestration.

    Parameters
    ----------
    email_agent:
        Email analysis agent.
    url_agent:
        URL analysis agent.
    sender_agent:
        Sender analysis agent.
    max_agents:
        Maximum allowed agent execution budget per episode.
    reward_config:
        Configuration for step and terminal rewards.
    confidence_threshold:
        Threshold for state contradiction check.
    """

    def __init__(
        self,
        email_agent: BaseAgent,
        url_agent: BaseAgent,
        sender_agent: BaseAgent,
        max_agents: int = 3,
        reward_config: RewardConfig | None = None,
        confidence_threshold: float = 0.90,
    ) -> None:
        self.email_agent = email_agent
        self.url_agent = url_agent
        self.sender_agent = sender_agent
        self.max_agents = max(1, int(max_agents))
        self.reward_config = reward_config or RewardConfig()
        self.confidence_threshold = float(confidence_threshold)

        self._agent_map: dict[OrchestrationAction, BaseAgent] = {
            OrchestrationAction.RUN_EMAIL_AGENT: self.email_agent,
            OrchestrationAction.RUN_URL_AGENT: self.url_agent,
            OrchestrationAction.RUN_SENDER_AGENT: self.sender_agent,
        }

        # Episode state
        self.context: AgentContext | None = None
        self.ground_truth: int | None = None
        self.results: list[AgentResult] = []
        self.executed_agents: list[str] = []
        self.available_agents: list[str] = []
        self.remaining_budget: int = self.max_agents
        self.done: bool = False
        self.current_state: RLState | None = None

    def reset(
        self,
        context: AgentContext,
        ground_truth: int | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Reset environment for a new detection episode.

        Parameters
        ----------
        context:
            The input context to analyze.
        ground_truth:
            Optional ground-truth binary label (1=phishing, 0=legitimate).
            Used solely for terminal reward calculation during training.

        Returns
        -------
        tuple[np.ndarray, dict[str, Any]]
            Initial state vector and information dictionary.
        """
        self.context = context
        self.ground_truth = ground_truth
        self.results = []
        self.executed_agents = []
        self.remaining_budget = self.max_agents
        self.done = False

        # Determine eligible agents for this context
        self.available_agents = []
        for action, agent in self._agent_map.items():
            if agent.can_analyze(context):
                self.available_agents.append(agent.get_agent_name())

        self.current_state = encode_state(
            results=self.results,
            available_agents=self.available_agents,
            max_agents=self.max_agents,
            remaining_budget=self.remaining_budget,
            confidence_threshold=self.confidence_threshold,
        )

        valid_actions = get_valid_actions(
            executed_agents=self.executed_agents,
            available_agents=self.available_agents,
            remaining_budget=self.remaining_budget,
        )

        info = {
            "valid_actions": valid_actions,
            "action_mask": compute_action_mask(
                self.executed_agents, self.available_agents, self.remaining_budget
            ),
            "available_agents": list(self.available_agents),
            "budget": self.remaining_budget,
        }

        return self.current_state.features, info

    def get_valid_actions(self) -> list[int]:
        """Return currently valid action indices."""
        return get_valid_actions(
            executed_agents=self.executed_agents,
            available_agents=self.available_agents,
            remaining_budget=self.remaining_budget,
        )

    def step(self, action: int | OrchestrationAction) -> StepResult:
        """
        Take an orchestration action.

        Parameters
        ----------
        action:
            Discrete action to execute.

        Returns
        -------
        StepResult
            Container with state, reward, done flag, and info.
        """
        if self.done:
            raise RuntimeError("Cannot step in an environment that has already finished. Call reset().")

        if self.context is None:
            raise RuntimeError("Environment has not been initialized. Call reset() before step().")

        action_enum = OrchestrationAction(action)
        valid_actions = self.get_valid_actions()
        is_valid = int(action_enum.value) in valid_actions

        info: dict[str, Any] = {
            "action_taken": action_enum.name,
            "is_valid": is_valid,
        }

        # 1. Invalid action handling
        if not is_valid:
            step_reward = calculate_step_reward(action_enum, is_valid=False, config=self.reward_config)
            info["reason"] = f"Action {action_enum.name} is ineligible or budget is exhausted."
            return StepResult(
                state=self.current_state.features,
                reward=step_reward,
                done=self.done,
                info=info,
            )

        # 2. Terminal action: STOP_AND_CLASSIFY
        if action_enum.is_terminal:
            self.done = True
            terminal_reward = 0.0
            breakdown = {}
            if self.ground_truth is not None:
                terminal_reward, breakdown = calculate_terminal_reward(
                    results=self.results,
                    ground_truth=self.ground_truth,
                    config=self.reward_config,
                )
            info["terminal_breakdown"] = breakdown
            info["reason"] = "Policy decided to STOP_AND_CLASSIFY."
            return StepResult(
                state=self.current_state.features,
                reward=terminal_reward,
                done=True,
                info=info,
            )

        # 3. Agent execution action
        target_agent = self._agent_map.get(action_enum)
        agent_name = target_agent.get_agent_name() if target_agent else "unknown"

        step_reward = calculate_step_reward(action_enum, is_valid=True, config=self.reward_config)

        try:
            agent_result = target_agent.analyze(self.context)
            self.results.append(agent_result)
            self.executed_agents.append(agent_name)
            self.remaining_budget -= 1
            info["agent_executed"] = agent_name
            info["phishing_probability"] = agent_result.phishing_probability
        except Exception as exc:
            logger.warning("Agent %s raised exception during step: %s", agent_name, exc)
            self.remaining_budget -= 1
            info["agent_error"] = str(exc)

        # Update state
        self.current_state = encode_state(
            results=self.results,
            available_agents=self.available_agents,
            max_agents=self.max_agents,
            remaining_budget=self.remaining_budget,
            confidence_threshold=self.confidence_threshold,
        )

        # Check if episode must terminate due to budget exhaustion or no valid actions
        new_valid_actions = self.get_valid_actions()
        info["valid_actions"] = new_valid_actions
        info["remaining_budget"] = self.remaining_budget

        # If budget exhausted or only STOP remains, check if terminal
        if self.remaining_budget <= 0 or new_valid_actions == [OrchestrationAction.STOP_AND_CLASSIFY.value]:
            self.done = True
            terminal_reward = 0.0
            breakdown = {}
            if self.ground_truth is not None:
                terminal_reward, breakdown = calculate_terminal_reward(
                    results=self.results,
                    ground_truth=self.ground_truth,
                    config=self.reward_config,
                )
            info["terminal_breakdown"] = breakdown
            info["reason"] = "Episode terminated: budget exhausted or no further agents available."
            return StepResult(
                state=self.current_state.features,
                reward=step_reward + terminal_reward,
                done=True,
                info=info,
            )

        return StepResult(
            state=self.current_state.features,
            reward=step_reward,
            done=False,
            info=info,
        )


__all__ = [
    "StepResult",
    "PhishingRLEnvironment",
]
