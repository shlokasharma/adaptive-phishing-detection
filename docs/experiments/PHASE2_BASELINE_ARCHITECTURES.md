# Phase 2 Baseline Model Architectures

## 1. Overview

Phase 2 establishes the machine-learning baseline layer of the
Adaptive Phishing Detection system.

The objective of this phase is to establish reproducible reference
models for two primary phishing-detection modalities:

1. Email-text phishing detection
2. URL-based phishing detection

These baseline models provide the quantitative foundation against which
the later adaptive evidence-orchestration, specialized-agent,
explainability, and reinforcement-learning components can be evaluated.

The Phase 2 models operate exclusively on the frozen datasets and
train/validation/test partitions produced during Phase 1.

---

## 2. Label Convention

A common binary label convention is used throughout the Phase 2 system:

| Label | Meaning |
|---|---|
| `0` | Legitimate |
| `1` | Phishing |

All models, evaluation utilities, and downstream components follow this
convention.

---

## 3. Email Detection Pipeline

### 3.1 Input Representation

The email detector operates on the `clean_text` representation produced
by the Phase 1 email preprocessing pipeline.

The canonical Phase 1 email split contains:

- `sample_id`
- `source_dataset`
- `label`
- `clean_text`

Only `clean_text` is used as the model input.

This separation prevents metadata fields such as dataset identity or
sample identifiers from becoming predictive features.

---

### 3.2 TF-IDF Feature Extraction

Email text is transformed into numerical features using Term
Frequency-Inverse Document Frequency (TF-IDF).

The Phase 2 configuration uses:

- Maximum features: `50,000`
- N-gram range: `(1, 2)`
- Minimum document frequency: `2`
- Maximum document frequency: `0.95`
- Sublinear term frequency: enabled
- Unicode accent stripping: enabled

The feature extractor is fitted exclusively on the training data.

Validation and test data are transformed using the already-fitted
training feature extractor.

This prevents information leakage from validation or test samples into
the learned vocabulary and inverse-document-frequency statistics.

---

### 3.3 Email Logistic Regression

The first email baseline is a linear Logistic Regression classifier
operating on TF-IDF features.

Architecture:

```text
Raw Email
    |
    v
Phase 1 clean_text
    |
    v
TF-IDF Vectorization
    |
    v
50,000-dimensional sparse feature representation
    |
    v
Logistic Regression
    |
    v
Phishing Probability
    |
    v
Binary Prediction