# Phase 3 Agent Architecture

## 1. Overview

Phase 3 introduces a modular specialized-agent architecture for
adaptive phishing detection.

The architecture separates:

1. input representation,
2. specialized analysis,
3. evidence generation,
4. confidence estimation,
5. adaptive orchestration,
6. decision aggregation,
7. end-to-end execution.

---

## 2. Architectural Principle

Each specialized agent performs one clearly defined analytical task.

Agents do not directly control other agents.

Agent selection is delegated to the Orchestrator Agent.

The end-to-end pipeline delegates orchestration to the Orchestrator
rather than duplicating orchestration logic.

---

## 3. Components

### BaseAgent

Defines the common interface for specialized agents.

### Evidence

Defines the standardized evidence representation.

### AgentResult

Defines the standardized output of each agent.

### EmailAnalysisAgent

Provides email-content phishing analysis.

### URLAnalysisAgent

Provides URL-based phishing analysis.

### SenderAnalysisAgent

Provides sender and header analysis.

### Confidence Layer

Provides confidence, uncertainty, and evidence-strength calculations.

### OrchestratorAgent

Selects which specialized agents should execute.

### PhishingDetectionPipeline

Provides the high-level end-to-end interface.

---

## 4. Execution Flow

```text
Input
  |
  v
PhishingDetectionPipeline
  |
  v
AgentContext
  |
  v
OrchestratorAgent
  |
  +----> EmailAnalysisAgent
  |
  +----> URLAnalysisAgent
  |
  +----> SenderAnalysisAgent
  |
  v
AgentResult[]
  |
  v
Evidence[]
  |
  v
Confidence / Uncertainty
  |
  v
Decision Trace
  |
  v
Final Decision