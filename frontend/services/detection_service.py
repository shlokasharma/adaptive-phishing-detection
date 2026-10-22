from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from phishing_detection.agents.email_agent import EmailAnalysisAgent
from phishing_detection.agents.sender_agent import SenderAnalysisAgent
from phishing_detection.agents.url_agent import URLAnalysisAgent
from phishing_detection.features.url_runtime import build_runtime_url_features
from phishing_detection.orchestration.adaptive_policy import AdaptiveEvidencePolicy
from phishing_detection.orchestration.orchestrator import OrchestratorAgent
from phishing_detection.pipeline.phishing_pipeline import PhishingDetectionPipeline
from phishing_detection.synthesis.service import EvidenceSynthesisService
from phishing_detection.threat_intelligence.service import ThreatIntelligenceService
from frontend.services.xai_service import XAIService

logger = logging.getLogger(__name__)

# ============================================================
# PROJECT PATHS & ARTIFACTS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "experiments" / "baselines" / "models"
RL_MODEL_PATH = PROJECT_ROOT / "experiments" / "models" / "rl" / "dqn_policy.json"

EMAIL_MODEL_PATH = MODEL_DIR / "email_linear_svm.pkl"
EMAIL_EXTRACTOR_PATH = MODEL_DIR / "email_linear_svm_tfidf_extractor.pkl"

URL_MODEL_PATH = MODEL_DIR / "url_random_forest.pkl"
URL_PREPROCESSOR_PATH = MODEL_DIR / "url_random_forest_feature_extractor.pkl"


# ============================================================
# DETECTION SERVICE
# ============================================================

class DetectionService:
    """
    Research-grade frontend adapter for the multi-agent adaptive
    phishing detection system.

    Connects Streamlit UI directly to:
    - Specialized detection agents (Email, URL, Sender)
    - Model explainability & feature attribution (SHAP, LIME, rule-based)
    - Deterministic adaptive evidence orchestration
    - Deep Q-Network (DQN) reinforcement learning orchestration
    - Grounded evidence synthesis & cross-modality conflict resolution
    - Multi-provider threat intelligence & reputation caching
    """

    def __init__(self, policy_type: str = "adaptive") -> None:
        self.email_agent = EmailAnalysisAgent(
            model_path=EMAIL_MODEL_PATH,
            extractor_path=EMAIL_EXTRACTOR_PATH,
        )

        self.url_agent = URLAnalysisAgent(
            model_path=URL_MODEL_PATH,
            preprocessor_path=URL_PREPROCESSOR_PATH,
        )

        self.sender_agent = SenderAnalysisAgent()

        self._current_policy_type = policy_type
        self.orchestrator = self._create_orchestrator(policy_type)

        self.synthesis_service = EvidenceSynthesisService()
        self.threat_intel_service = ThreatIntelligenceService()

        self.pipeline = PhishingDetectionPipeline(
            orchestrator=self.orchestrator,
            pipeline_name="adaptive_phishing_detection_pipeline",
            pipeline_version="1.0.0",
            synthesis_service=self.synthesis_service,
            enable_synthesis=True,
            threat_intelligence_service=self.threat_intel_service,
            enable_threat_intel=True,
        )

        self.xai_service = XAIService()

    def _create_orchestrator(self, policy_type: str) -> OrchestratorAgent:
        """Instantiate an orchestrator configured with the selected decision policy."""
        policy_lower = str(policy_type).strip().lower()

        if any(keyword in policy_lower for keyword in ("rl", "reinforcement", "dqn", "deep_q", "q_network", "phase6")):
            try:
                from phishing_detection.orchestration.rl_policy import RLEvidencePolicy

                policy = RLEvidencePolicy(model_path=RL_MODEL_PATH)
            except Exception as exc:
                logger.warning("Failed to load RL policy (%s), falling back to adaptive policy.", exc)
                policy = AdaptiveEvidencePolicy()
        elif any(keyword in policy_lower for keyword in ("fixed", "static", "baseline")):
            policy = None  # None delegates to default fixed threshold orchestration
        else:
            policy = AdaptiveEvidencePolicy()

        return OrchestratorAgent(
            email_agent=self.email_agent,
            url_agent=self.url_agent,
            sender_agent=self.sender_agent,
            confidence_threshold=0.90,
            max_agents=3,
            policy=policy,
        )

    def set_policy(self, policy_type: str) -> None:
        """Dynamically update the active orchestration decision policy."""
        if policy_type != self._current_policy_type:
            self.orchestrator = self._create_orchestrator(policy_type)
            self.pipeline.orchestrator = self.orchestrator
            self._current_policy_type = policy_type

    @staticmethod
    def build_email_text(
        subject: str = "",
        body: str = "",
    ) -> str:
        """Format subject and body into canonical email text."""
        subject = (subject or "").strip()
        body = (body or "").strip()

        if subject and body:
            return f"Subject: {subject}\n\n{body}"
        return subject or body

    def _build_url_features(self, url: str) -> pd.DataFrame:
        """
        Convert a raw URL string into the frozen 30-feature schema expected
        by URLAnalysisAgent and the URL classifier.
        """
        if not url or not str(url).strip():
            raise ValueError("URL must not be empty.")

        url_clean = str(url).strip()
        self.url_agent._load_artifacts()
        preprocessor = self.url_agent._preprocessor

        if preprocessor is None:
            raise RuntimeError("URL preprocessing artifact could not be loaded.")

        feature_names = getattr(preprocessor, "feature_names", None)
        if feature_names is None:
            raise RuntimeError(
                "The frozen URL preprocessing artifact does not expose its feature schema."
            )

        runtime_features = build_runtime_url_features(
            url=url_clean,
            feature_names=list(feature_names),
        )

        return pd.DataFrame(
            [runtime_features],
            columns=list(feature_names),
        )

    @staticmethod
    def serialize(value: Any) -> Any:
        """Convert backend objects into Streamlit-safe JSON-compatible dictionaries."""
        if value is None:
            return None

        if isinstance(value, (str, int, float, bool)):
            return value

        if isinstance(value, Path):
            return str(value)

        if hasattr(value, "value"):
            return DetectionService.serialize(value.value)

        if hasattr(value, "to_dict"):
            try:
                return DetectionService.serialize(value.to_dict())
            except Exception:
                pass

        if isinstance(value, dict):
            return {
                str(k): DetectionService.serialize(v)
                for k, v in value.items()
            }

        if isinstance(value, (list, tuple)):
            return [DetectionService.serialize(item) for item in value]

        if isinstance(value, pd.DataFrame):
            return value.to_dict(orient="records")

        if hasattr(value, "__dict__"):
            return {
                str(k): DetectionService.serialize(v)
                for k, v in vars(value).items()
                if not k.startswith("_")
            }

        return str(value)

    def analyze(
        self,
        mode: str = "email",
        subject: str = "",
        body: str = "",
        sender: str = "",
        receiver: str = "",
        url: str = "",
        policy_type: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Execute end-to-end adaptive phishing analysis.
        """
        mode_str = (mode or "email").strip().lower()
        subject_str = (subject or "").strip()
        body_str = (body or "").strip()
        sender_str = (sender or "").strip()
        receiver_str = (receiver or "").strip()
        url_str = (url or "").strip()

        if policy_type:
            self.set_policy(policy_type)

        email_text = self.build_email_text(
            subject=subject_str,
            body=body_str,
        )

        metadata: dict[str, Any] = {
            "frontend": True,
            "frontend_version": "1.0.0",
            "analysis_mode": mode,
            "input_text": email_text if email_text else None,
            "subject": subject_str,
            "body": body_str,
            "sender": sender_str if sender_str else None,
            "receiver": receiver_str if receiver_str else None,
            "policy_type": self._current_policy_type,
        }

        # URL feature construction
        target_url = url_str if url_str else None
        if target_url:
            try:
                url_features_df = self._build_url_features(target_url)
                metadata["url_features"] = {
                    col: float(url_features_df.iloc[0][col])
                    for col in url_features_df.columns
                }
            except Exception as exc:
                logger.warning("Failed to extract runtime URL features: %s", exc)

        pipeline_email = email_text if email_text else None
        pipeline_sender = sender_str if sender_str else None

        pipeline_result = self.pipeline.analyze(
            email_text=pipeline_email,
            url=target_url,
            sender=pipeline_sender,
            headers={},
            metadata=metadata,
        )

        result = pipeline_result.to_dict()

        # Run Model Explainability
        try:
            agent_results = getattr(pipeline_result, "agent_results", [])
            decision_trace = getattr(pipeline_result, "decision_trace", [])
            xai_result = self.xai_service.build_explanation_result(
                agent_results=agent_results,
                decision_trace=decision_trace,
                metadata=metadata,
            )
            result["explanations"] = xai_result
            result["xai_status"] = "available"
        except Exception as exc:
            logger.warning("Explainability integration encountered error: %s", exc)
            result["explanations"] = {"error": str(exc)}
            result["xai_status"] = "error"

        result["frontend"] = {
            "analysis_mode": mode,
            "policy_type": self._current_policy_type,
            "url_supplied": bool(target_url),
            "sender_supplied": bool(pipeline_sender),
            "email_supplied": bool(pipeline_email),
        }

        return result


# ============================================================
# SINGLETON
# ============================================================

_service: DetectionService | None = None


def get_detection_service(policy_type: str = "adaptive") -> DetectionService:
    global _service
    if _service is None:
        _service = DetectionService(policy_type=policy_type)
    elif policy_type and policy_type != _service._current_policy_type:
        _service.set_policy(policy_type)
    return _service


def analyze_input(**kwargs: Any) -> dict[str, Any]:
    return get_detection_service().analyze(**kwargs)
