# Phase 9: Research-Grade Frontend, Live Adaptive Decision Visualization, Evaluation, and Deployment

## 1. Executive Summary & Research Contribution

The goal of Phase 9 is to make the entire multi-phase **Adaptive Phishing Detection Framework** transparent, interactive, explainable, and empirically verifiable.

Unlike conventional phishing detection systems that deploy monolithic classifiers or static ensembles—where every model and feature extraction pipeline executes regardless of sample difficulty—this system dynamically orchestrates specialized agents:
- **Email Analysis Agent**: Fast lexical analysis using a Linear SVM over a 50,000-feature TF-IDF space.
- **URL Analysis Agent**: Deep structural feature inspection evaluated with Random Forest over 30 UCI phishing-website indicators.
- **Sender Analysis Agent**: Heuristic and rule-based evaluation of SPF/DKIM authentication and domain masquerading.
- **Phase 8 Threat Intelligence Layer**: Multi-provider external reputation checks (VirusTotal, Google Safe Browsing, and local TTL cache).
- **Phase 7 Evidence Synthesis**: Grounded reasoning and cross-modality conflict resolution that prevents compromised accounts from masking malicious payloads.
- **Phase 5 & 6 Adaptive Policies**: Deterministic evidence thresholding and Deep Q-Network (DQN) reinforcement learning that balance detection performance against computational cost.

---

## 2. Architecture & Component Decomposition

```
User Submission (Email Subject, Body, Sender, URL)
                       │
                       ▼
         DetectionService (Frontend Adapter)
                       │
                       ▼
          PhishingDetectionPipeline
                       │
        ┌──────────────┴──────────────┐
        │                             │
        ▼                             ▼
OrchestratorAgent            ThreatIntelligenceService
(Adaptive Policy / RL Policy) (VirusTotal, GSB, Local Cache)
        │
   ┌────┴─────────────────────────────┐
   ▼                                  ▼
Email Analysis Agent         URL Analysis Agent / Sender Agent
(Linear SVM 50k TF-IDF)      (Random Forest / Heuristics)
   │                                  │
   └──────────────┬───────────────────┘
                  │
                  ▼
      Phase 7 Evidence Synthesis
   (Grounded Reasoning & Conflict Resolution)
                  │
                  ▼
        Phase 4 Explainability
        (SHAP, LIME, Aggregation)
                  │
                  ▼
        Research-Grade UI (Streamlit)
(Decision Trail, Risk Gauges, Benchmark Analytics)
```

### Key Frontend Components

1. **`frontend/app.py`**:
   The primary entry point. Implements a research-grade dark theme (`#080D18`, `#111B2E`, `#4F8CFF`, `#38D9F5`, `#36D399`, `#FF5D73`) across four distinct views:
   - **Live Detection Workspace**: Real-time analysis with interactive risk gauges, agent intelligence cards, timeline traces, and XAI feature attributions.
   - **Empirical Evaluation & Benchmarks**: Real evaluation summaries loaded directly from project JSON artifacts (`rl_evaluation_summary.json` and `final_test_metrics.json`).
   - **System Architecture & Design**: Structural flowcharts, agent capability matrices, and implementation statuses across Phases 1 through 9.
   - **Research Contribution & Evaluator Guide**: Framing for university evaluations (DA2), technical defenses, and step-by-step walkthroughs.

2. **`frontend/services/detection_service.py`**:
   The bridge between the presentation layer and backend detection agents. It:
   - Translates raw input parameters into canonical representations.
   - Dynamically configures the orchestrator with either the Phase 5 Deterministic Policy, Phase 6 RL Policy, or Fixed Baseline.
   - Converts URLs into 30-dimensional UCI numerical vectors via `build_runtime_url_features`.
   - Executes the end-to-end detection pipeline with threat intelligence and grounded synthesis enabled.
   - Gathers SHAP and LIME feature attributions via `XAIService`.

3. **`frontend/services/xai_service.py`**:
   Wraps `ExplainabilityPipelineIntegrator` to generate per-agent and multi-agent aggregated explanations without crashing the UI if an individual explainer encounters an edge case.

---

## 3. Empirical Evaluation & Policy Comparison

All benchmark metrics displayed in the user interface are derived from genuine, frozen experimental artifacts:

### Multi-Agent Decision Policy Comparison (`rl_evaluation_summary.json`)

| Metric | Fixed Baseline (Static Threshold) | Phase 5 Deterministic Policy | Phase 6 RL Policy (DQN) |
| :--- | :---: | :---: | :---: |
| **Total Test Samples** | 100 | 100 | 100 |
| **Accuracy** | 94.0% | **99.0%** | 94.0% |
| **Precision** | 1.0000 | 1.0000 | 1.0000 |
| **Recall** | 0.8947 | **0.9825** | 0.8947 |
| **F1-Score** | 0.9444 | **0.9912** | 0.9444 |
| **False Positive Rate** | 0.00% | 0.00% | 0.00% |
| **False Negative Rate** | 10.53% | **1.75%** | 10.53% |
| **Avg Agents Executed** | 1.26 | **1.04** | 1.26 |
| **Average Latency (ms)** | 9.04 ms | **5.36 ms** | 8.66 ms |
| **Mean Reward** | 7.27 | **9.13** | 7.27 |

### Research Insight: Computational Savings of Adaptive Stopping
The Phase 5 deterministic adaptive policy executes **1.04 agents on average**, representing a **17.5% reduction in computational latency and agent execution costs** compared to the static baseline (1.26 agents). Blatant phishing and routine safe emails are halted after Agent 1 with >0.90 confidence, preserving compute resources while delivering superior F1 performance (0.9912).

### Baseline Model Performance (`final_test_metrics.json`)
- **Email Classifier (Linear SVM)**:
  - Test Accuracy: 98.93%
  - Precision: 98.54% | Recall: 99.45% | F1-Score: 0.9899
  - Out-of-sample Test Samples: 21,708
- **URL Classifier (Random Forest)**:
  - Test Accuracy: 95.74%
  - Precision: 94.85% | Recall: 96.43% | F1-Score: 0.9563
  - Evaluated on 30 UCI structural indicators

---

## 4. Preset Demonstration Scenarios

The frontend includes 4 pre-configured research scenarios accessible via 1-click preset buttons:

1. **High-Confidence Phishing (Early Stop)**:
   - *Input*: Deceptive banking urgency notification regarding unauthorized payroll access.
   - *Behavior*: Triggers high lexical urgency. Email Agent produces phishing probability ~0.97. The adaptive policy halts at Step 1 with an "Early Stop" status, executing only 1 agent.
2. **Suspicious URL Requiring Threat Intel Escalation**:
   - *Input*: Benign invoice text accompanied by an IP-hosted login script URL (`http://192.168.1.105/admin-login-verify/secure.php`).
   - *Behavior*: Text classifier confidence is insufficient. System escalates to URL analysis and Phase 8 Threat Intelligence, identifying suspicious hosting and escalating risk to Phishing.
3. **Modality Tension / Compromised Account (Conflict Resolution)**:
   - *Input*: Internal corporate executive sender (`executive-office@trusted-enterprise.org`) linking to a deceptive subdomain attacker redirect (`secure-login.trusted-enterprise.org.attacker-redirect.ru/sso`).
   - *Behavior*: Sender Agent flags legitimate (SPF pass), while URL Agent flags phishing. Phase 7 Evidence Synthesis detects cross-modality contradiction and applies security-first resolution to convict the payload.
4. **Clean Routine Corporate Notification**:
   - *Input*: Internal engineering all-hands meeting agenda from a trusted domain.
   - *Behavior*: Benign lexical tokens, clean domain alignment, low phishing probability (< 0.05). Adaptive policy terminates with low risk (Safe).

---

## 5. Deployment & Execution Instructions

### Local Application Launch
To launch the research-grade Streamlit frontend:
```powershell
cd C:\Users\HP\OneDrive\Desktop\adaptive-phishing-detection
.\.venv\Scripts\streamlit.exe run frontend/app.py
```
The application will open automatically at `http://localhost:8501`.

### Verification & Testing
To execute the complete test suite verifying all 9 project phases:
```powershell
cd C:\Users\HP\OneDrive\Desktop\adaptive-phishing-detection
.\.venv\Scripts\pytest.exe tests/ -v
```
All **538 tests** across all phases pass with 0 failures.
