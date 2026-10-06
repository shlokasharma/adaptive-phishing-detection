# Phase 4 — Explainable AI Architecture

## 1. Overview

Phase 4 introduces the Explainable AI (XAI) layer into the adaptive
phishing detection system.

The purpose of this phase is to make model and agent decisions
interpretable while preserving the modular architecture established
during Phases 2 and 3.

The XAI layer supports:

- SHAP explanations
- LIME explanations
- Native rule-based explanations
- Unified explanation representation
- SHAP/LIME comparison
- Multi-agent explanation aggregation
- Pipeline-level explainable predictions
- Quantitative XAI evaluation

The architecture is designed so that XAI remains an independent layer
between model/agent inference and final evidence synthesis.

---

## 2. High-Level Architecture

```text
                         INPUT
                           |
                           v
                  +------------------+
                  |  Detection Model |
                  +------------------+
                           |
                           v
                  +------------------+
                  |   Agent Result   |
                  +------------------+
                           |
              +------------+------------+
              |            |            |
              v            v            v
           SHAP           LIME       Native XAI
              |            |            |
              +------------+------------+
                           |
                           v
                  +------------------+
                  | Unified XAI      |
                  | Explanation      |
                  | Schema            |
                  +------------------+
                           |
                           v
                  +------------------+
                  | Multi-Agent      |
                  | Aggregation      |
                  +------------------+
                           |
                           v
                  +------------------+
                  | Explainable      |
                  | Pipeline Result  |
                  +------------------+
                           |
                           v
                    Final Decision