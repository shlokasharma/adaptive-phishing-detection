"""
Evaluate Reinforcement Learning Policy vs Deterministic and Baseline Policies (Phase 6).

Compares three orchestration configurations on held-out data:
1. Fixed / Full Orchestration: Executes all eligible agents.
2. Phase 5 Deterministic Policy (AdaptiveEvidencePolicy): Rules-based early stopping.
3. Phase 6 RL Policy (RLEvidencePolicy): Learned adaptive action selection.

Evaluates accuracy, recall, precision, F1, false negative rate, average agent count,
and execution latency.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from phishing_detection.agents.base import AgentContext
from phishing_detection.agents.email_agent import EmailAnalysisAgent
from phishing_detection.agents.sender_agent import SenderAnalysisAgent
from phishing_detection.agents.url_agent import URLAnalysisAgent
from phishing_detection.orchestration.adaptive_policy import AdaptiveEvidencePolicy
from phishing_detection.orchestration.orchestrator import OrchestratorAgent
from phishing_detection.orchestration.rl_actions import (
    ACTION_NAMES,
    OrchestrationAction,
)
from phishing_detection.orchestration.rl_policy import (
    DeepQPolicy,
    RLEvidencePolicy,
    TabularQPolicy,
)
from phishing_detection.orchestration.rl_reward import RewardConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

URL_REGEX = re.compile(r"https?://[^\s<>\"']+|www\.[^\s<>\"']+")
EMAIL_REGEX = re.compile(r"[\w\.-]+@[\w\.-]+\.\w+")


def create_agent_context(row: pd.Series) -> tuple[AgentContext, int]:
    """Extract context and ground-truth label from a split dataset row."""
    text = str(row.get("clean_text", ""))
    label = int(row.get("label", 0))

    found_urls = URL_REGEX.findall(text)
    url = found_urls[0] if found_urls else ""

    found_emails = EMAIL_REGEX.findall(text)
    sender = found_emails[0] if found_emails else ""

    metadata = {}
    if url:
        metadata["url_features"] = {"url_length": len(url)}

    context = AgentContext(
        input_text=text,
        url=url,
        sender=sender,
        headers={"spf": "pass"} if sender else {},
        metadata=metadata,
    )
    return context, label


def build_agents(project_root: Path) -> tuple[EmailAnalysisAgent, URLAnalysisAgent, SenderAnalysisAgent]:
    """Instantiate standard agents with baseline models."""
    email_model_path = project_root / "experiments" / "baselines" / "models" / "email_linear_svm.pkl"
    email_tfidf_path = project_root / "experiments" / "baselines" / "models" / "email_linear_svm_tfidf_extractor.pkl"

    url_model_path = project_root / "experiments" / "baselines" / "models" / "url_gradient_boosting.pkl"
    url_prep_path = project_root / "experiments" / "baselines" / "models" / "url_gradient_boosting_feature_extractor.pkl"

    email_agent = EmailAnalysisAgent(
        model_path=email_model_path,
        extractor_path=email_tfidf_path,
    )
    url_agent = URLAnalysisAgent(
        model_path=url_model_path,
        preprocessor_path=url_prep_path,
    )
    sender_agent = SenderAnalysisAgent()

    return email_agent, url_agent, sender_agent


def evaluate_orchestrator(
    orchestrator: OrchestratorAgent,
    contexts_and_labels: list[tuple[AgentContext, int]],
    reward_config: RewardConfig,
) -> dict[str, Any]:
    """Run orchestration over all samples and compute metrics."""
    tp, tn, fp, fn = 0, 0, 0, 0
    total_time_ms = 0.0
    agent_executions_count: list[int] = []
    agent_usage = {"email_analysis_agent": 0, "url_analysis_agent": 0, "sender_analysis_agent": 0}
    total_reward = 0.0

    for context, y_true in contexts_and_labels:
        t0 = time.perf_counter()
        result = orchestrator.analyze(context)
        t1 = time.perf_counter()

        elapsed_ms = (t1 - t0) * 1000.0
        total_time_ms += elapsed_ms

        y_pred = int(result.label)
        executed_agents = [res.agent_name for res in result.agent_results]
        agent_executions_count.append(len(executed_agents))

        for agent_name in executed_agents:
            if agent_name in agent_usage:
                agent_usage[agent_name] += 1

        # Classification outcome
        if y_true == 1 and y_pred == 1:
            tp += 1
            reward = reward_config.reward_tp
        elif y_true == 0 and y_pred == 0:
            tn += 1
            reward = reward_config.reward_tn
        elif y_true == 0 and y_pred == 1:
            fp += 1
            reward = -reward_config.penalty_fp
        else:  # y_true == 1 and y_pred == 0
            fn += 1
            reward = -reward_config.penalty_fn

        # Add step cost deduction
        reward -= len(executed_agents) * reward_config.step_cost
        total_reward += reward

    n = len(contexts_and_labels)
    acc = (tp + tn) / n if n > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    avg_agents = float(np.mean(agent_executions_count)) if agent_executions_count else 0.0
    avg_latency = total_time_ms / n if n > 0 else 0.0
    mean_reward = total_reward / n if n > 0 else 0.0

    return {
        "total_samples": n,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "false_positive_rate": round(fpr, 4),
        "false_negative_rate": round(fnr, 4),
        "avg_agents_executed": round(avg_agents, 2),
        "agent_usage": agent_usage,
        "avg_latency_ms": round(avg_latency, 2),
        "mean_reward": round(mean_reward, 2),
    }


def run_evaluation(
    data_path: Path | None = None,
    sample_size: int = 100,
    seed: int = 42,
    rl_model_path: Path | None = None,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Execute comparative evaluation between Fixed, Phase 5 Deterministic, and Phase 6 RL."""
    if data_path is None:
        data_path = PROJECT_ROOT / "data" / "splits" / "email" / "validation.csv"

    if rl_model_path is None:
        rl_model_path = PROJECT_ROOT / "experiments" / "models" / "rl" / "dqn_policy.json"

    if output_path is None:
        out_dir = PROJECT_ROOT / "experiments" / "evaluations"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / "rl_evaluation_summary.json"
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Loading evaluation dataset from: %s", data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found: {data_path}")

    df = pd.read_csv(data_path)
    valid_rows = df[df["clean_text"].notna() & (df["clean_text"].str.strip() != "")]
    total_valid = len(valid_rows)

    # Subsample deterministically
    rng = np.random.default_rng(seed)
    eval_n = min(sample_size, total_valid)
    chosen_indices = rng.choice(total_valid, size=eval_n, replace=False)
    sample_df = valid_rows.iloc[chosen_indices]

    contexts_and_labels = [create_agent_context(row) for _, row in sample_df.iterrows()]
    logger.info("Prepared %d evaluation samples (seed=%d).", len(contexts_and_labels), seed)

    email_agent, url_agent, sender_agent = build_agents(PROJECT_ROOT)
    reward_config = RewardConfig()

    results: dict[str, Any] = {}

    # 1. Baseline: Fixed full execution (never stops early)
    logger.info("Running Configuration 1: Fixed / Full Orchestration (All eligible agents)...")
    fixed_policy = AdaptiveEvidencePolicy(
        confidence_threshold=1.0,
        min_evidence_count=999,
    )
    orch_fixed = OrchestratorAgent(
        email_agent=email_agent,
        url_agent=url_agent,
        sender_agent=sender_agent,
        max_agents=3,
        policy=fixed_policy,
    )
    results["fixed_baseline"] = evaluate_orchestrator(orch_fixed, contexts_and_labels, reward_config)

    # 2. Phase 5 Deterministic Adaptive Policy
    logger.info("Running Configuration 2: Phase 5 Deterministic Adaptive Evidence Policy...")
    det_policy = AdaptiveEvidencePolicy(
        confidence_threshold=0.90,
        min_evidence_count=1,
    )
    orch_deterministic = OrchestratorAgent(
        email_agent=email_agent,
        url_agent=url_agent,
        sender_agent=sender_agent,
        max_agents=3,
        policy=det_policy,
    )
    results["phase5_deterministic"] = evaluate_orchestrator(orch_deterministic, contexts_and_labels, reward_config)

    # 3. Phase 6 Reinforcement Learning Policy
    logger.info("Running Configuration 3: Phase 6 RL Policy (Loaded from %s)...", rl_model_path)
    rl_policy = RLEvidencePolicy(
        model_path=rl_model_path,
        fallback_policy=det_policy,
    )
    orch_rl = OrchestratorAgent(
        email_agent=email_agent,
        url_agent=url_agent,
        sender_agent=sender_agent,
        max_agents=3,
        policy=rl_policy,
    )
    results["phase6_rl"] = evaluate_orchestrator(orch_rl, contexts_and_labels, reward_config)

    # Print summary table
    print("\n" + "=" * 90)
    print("PHASE 6 ADAPTIVE ORCHESTRATION COMPARATIVE EVALUATION")
    print("=" * 90)
    print(f"{'Metric':<25} | {'Fixed Baseline':<18} | {'Phase 5 Deterministic':<22} | {'Phase 6 RL Policy':<18}")
    print("-" * 90)

    keys = [
        ("Accuracy", "accuracy"),
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("F1-Score", "f1_score"),
        ("False Positive Rate", "false_positive_rate"),
        ("False Negative Rate", "false_negative_rate"),
        ("Avg Agents Executed", "avg_agents_executed"),
        ("Avg Latency (ms)", "avg_latency_ms"),
        ("Mean Reward", "mean_reward"),
    ]

    for label, key in keys:
        fb_val = results["fixed_baseline"][key]
        det_val = results["phase5_deterministic"][key]
        rl_val = results["phase6_rl"][key]
        print(f"{label:<25} | {str(fb_val):<18} | {str(det_val):<22} | {str(rl_val):<18}")

    print("=" * 90)
    print("Agent Execution Counts:")
    for cfg in ["fixed_baseline", "phase5_deterministic", "phase6_rl"]:
        print(f"  {cfg:<22}: {results[cfg]['agent_usage']}")
    print("=" * 90)

    # Save summary artifact
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved evaluation summary to: %s", output_path)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RL Policy against baselines.")
    parser.add_argument("--samples", type=int, default=100, help="Number of evaluation samples.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    parser.add_argument("--data-path", type=str, default=None, help="Custom path to evaluation split.")
    parser.add_argument("--model-path", type=str, default=None, help="Path to trained RL model artifact.")
    parser.add_argument("--output-path", type=str, default=None, help="Path to save evaluation summary JSON.")

    args = parser.parse_args()

    data_p = Path(args.data_path) if args.data_path else None
    model_p = Path(args.model_path) if args.model_path else None
    out_p = Path(args.output_path) if args.output_path else None

    run_evaluation(
        data_path=data_p,
        sample_size=args.samples,
        seed=args.seed,
        rl_model_path=model_p,
        output_path=out_p,
    )


if __name__ == "__main__":
    main()
