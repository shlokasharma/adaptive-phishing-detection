"""
Reinforcement Learning State Representation for Phase 6.

Constructs a deterministic, numeric feature vector representing the current
state of evidence collection and orchestration context.
No ground-truth labels are ever included in the state representation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from phishing_detection.agents.confidence import (
    calculate_aggregate_evidence_strength,
    calculate_confidence,
    calculate_uncertainty,
)
from phishing_detection.agents.result import AgentResult


# Standard feature names in exact order
STATE_FEATURE_NAMES: list[str] = [
    "combined_phishing_probability",
    "confidence",
    "uncertainty",
    "aggregate_evidence_strength",
    "evidence_count_normalized",
    "agents_executed_normalized",
    "budget_remaining_normalized",
    "email_executed",
    "url_executed",
    "sender_executed",
    "has_contradiction",
    "email_available",
    "url_available",
    "sender_available",
]

STATE_DIMENSION: int = len(STATE_FEATURE_NAMES)
MAX_EVIDENCE_NORM: float = 10.0


@dataclass(frozen=True)
class RLState:
    """
    Structured representation of the RL orchestration state.

    Attributes
    ----------
    features:
        1D numpy array of shape (STATE_DIMENSION,) containing normalized floats in [0, 1].
    feature_names:
        Names corresponding to each element of the feature vector.
    """

    features: np.ndarray
    feature_names: list[str]

    def to_dict(self) -> dict[str, float]:
        """Convert state vector to a named dictionary."""
        return {
            name: float(self.features[i])
            for i, name in enumerate(self.feature_names)
        }

    def to_array(self) -> np.ndarray:
        """Return the underlying numpy array."""
        return self.features.copy()


def _sanitize_float(val: float, default: float = 0.0, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Ensure a value is a finite float bounded between min_val and max_val."""
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return default
        return max(min_val, min(max_val, f))
    except (TypeError, ValueError):
        return default


def encode_state(
    results: Sequence[AgentResult],
    available_agents: Sequence[str],
    max_agents: int = 3,
    remaining_budget: int | None = None,
    confidence_threshold: float = 0.90,
) -> RLState:
    """
    Construct the normalized numeric state vector from current agent results.

    Parameters
    ----------
    results:
        Sequence of AgentResult objects collected so far in the current episode.
    available_agents:
        List of agent names that are eligible to run on the context.
    max_agents:
        Total agent budget allowed for the episode.
    remaining_budget:
        Number of remaining executions. If None, calculated as max_agents - len(results).
    confidence_threshold:
        Threshold used to assess contradictory confident predictions.

    Returns
    -------
    RLState
        Deterministic, normalized state object with 14 features in [0.0, 1.0].
    """
    if max_agents < 1:
        max_agents = 1

    executed_count = len(results)
    if remaining_budget is None:
        rem_budget = max(0, max_agents - executed_count)
    else:
        rem_budget = max(0, remaining_budget)

    executed_names = {r.agent_name for r in results}
    available_set = set(available_agents)

    # 1. Combined probability, confidence, uncertainty
    if results:
        probs = []
        for r in results:
            p = _sanitize_float(r.phishing_probability, default=0.5)
            probs.append(p)
        comb_prob = sum(probs) / len(probs)
        conf = calculate_confidence(comb_prob)
        unc = calculate_uncertainty(conf)
    else:
        comb_prob = 0.5
        conf = 0.5
        unc = 0.5

    # 2. Evidence items and strength
    all_evidence = [ev for r in results for ev in r.evidence]
    evidence_count = len(all_evidence)
    ev_strength = calculate_aggregate_evidence_strength(all_evidence)

    # 3. Contradiction detection among confident agents
    has_contradiction = 0.0
    if len(results) > 1:
        confident_labels = [
            int(r.phishing_probability >= 0.5)
            for r in results
            if calculate_confidence(r.phishing_probability) >= confidence_threshold
        ]
        if len(set(confident_labels)) > 1:
            has_contradiction = 1.0

    # 4. Normalized metrics
    ev_count_norm = min(evidence_count, MAX_EVIDENCE_NORM) / MAX_EVIDENCE_NORM
    exec_norm = min(executed_count, max_agents) / float(max_agents)
    budget_norm = min(rem_budget, max_agents) / float(max_agents)

    # 5. Agent status flags
    email_exec = 1.0 if "email_analysis_agent" in executed_names else 0.0
    url_exec = 1.0 if "url_analysis_agent" in executed_names else 0.0
    sender_exec = 1.0 if "sender_analysis_agent" in executed_names else 0.0

    email_avail = 1.0 if "email_analysis_agent" in available_set else 0.0
    url_avail = 1.0 if "url_analysis_agent" in available_set else 0.0
    sender_avail = 1.0 if "sender_analysis_agent" in available_set else 0.0

    feature_values = [
        _sanitize_float(comb_prob, 0.5),
        _sanitize_float(conf, 0.5),
        _sanitize_float(unc, 0.5),
        _sanitize_float(ev_strength, 0.0),
        _sanitize_float(ev_count_norm, 0.0),
        _sanitize_float(exec_norm, 0.0),
        _sanitize_float(budget_norm, 1.0),
        email_exec,
        url_exec,
        sender_exec,
        has_contradiction,
        email_avail,
        url_avail,
        sender_avail,
    ]

    features_array = np.array(feature_values, dtype=np.float32)
    return RLState(features=features_array, feature_names=list(STATE_FEATURE_NAMES))


__all__ = [
    "STATE_FEATURE_NAMES",
    "STATE_DIMENSION",
    "MAX_EVIDENCE_NORM",
    "RLState",
    "encode_state",
]
