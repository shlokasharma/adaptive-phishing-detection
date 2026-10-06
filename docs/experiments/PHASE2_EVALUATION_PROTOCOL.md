# Phase 2 — Baseline Detection Models
## Evaluation Protocol

**Project:** Adaptive Phishing Detection  
**Phase:** Phase 2 — Baseline Detection Models  
**Status:** Evaluation protocol  
**Purpose:** Define the experimental objectives, datasets, evaluation methodology, model-selection criteria, and reproducibility requirements before implementation of baseline detection models.

---

## 1. Purpose of Phase 2

Phase 2 establishes conventional machine-learning baselines for phishing detection before introducing specialized analysis agents, explainable AI, adaptive evidence orchestration, reinforcement learning, and LLM-based reasoning.

The purpose of this phase is to determine how effectively conventional machine-learning approaches can detect phishing using individual evidence modalities.

The results obtained in this phase will serve as the experimental baseline against which subsequent intelligent and adaptive components of the proposed system will be compared.

Phase 2 therefore focuses on two primary detection tasks:

1. Email-based phishing detection.
2. URL-based phishing detection.

The evaluation protocol is defined before model training to prevent changes in evaluation methodology from being introduced after observing experimental results.

---

## 2. Phase 2 Research Objectives

The objectives of Phase 2 are:

1. Establish reproducible conventional machine-learning baselines for phishing detection.
2. Evaluate multiple classical machine-learning algorithms for email phishing detection.
3. Evaluate multiple classical machine-learning algorithms for URL phishing detection.
4. Compare model performance using consistent evaluation metrics.
5. Select the strongest baseline model for each detection modality using validation data.
6. Evaluate the selected models on previously unseen held-out test data.
7. Analyze false-positive and false-negative behavior.
8. Measure model confidence and inference efficiency.
9. Establish reference performance for subsequent phases of the adaptive phishing detection system.

---

## 3. Detection Tasks

### 3.1 Email Phishing Detection

The email detection task treats an email as the primary input evidence.

The model input will consist of textual information derived from the email, primarily:

- Subject
- Body
- Combined email text

The prediction target is binary:

| Label | Meaning |
|---|---|
| `0` | Legitimate |
| `1` | Phishing |

The baseline email models will operate independently of the future agentic architecture.

The purpose is to establish how effectively conventional text-based machine-learning models perform when email content is considered as the primary evidence source.

---

### 3.2 URL Phishing Detection

The URL detection task evaluates whether a URL is legitimate or phishing.

The URL experiments will use the URL datasets and feature representations established during Phase 1.

The prediction target is:

| Label | Meaning |
|---|---|
| `0` | Legitimate |
| `1` | Phishing |

The URL baseline models will use structured URL-related features rather than relying on the future adaptive evidence-selection mechanism.

---

## 4. Phase 1 Data Boundary

Phase 2 must use the benchmark foundation established and frozen during Phase 1.

The Phase 1 train, validation, and test partitions must not be regenerated merely to improve Phase 2 results.

The following principle applies:

> Phase 2 models must consume the Phase 1 benchmark splits as fixed experimental inputs.

The datasets and split definitions are documented in:

```text
docs/research/PHASE1_BENCHMARK_FREEZE.md