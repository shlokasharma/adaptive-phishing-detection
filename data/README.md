# Dataset Organization

This directory contains the datasets and derived data used by the
Adaptive Multi-Agent Phishing Detection System.

## Dataset Philosophy

The project does not combine every available dataset into a single
training table.

Instead, datasets are organized according to their research role:

1. Core training datasets
2. External validation datasets
3. Dynamic threat-intelligence sources
4. Future multimodal evidence
5. Independent benchmark datasets

This separation helps reduce data leakage and allows evaluation of
cross-dataset generalization.

## Directory Structure

```text
data/
├── raw/
│   ├── email/
│   ├── url/
│   ├── threat_intelligence/
│   └── benchmarks/
│
├── interim/
│   ├── email/
│   ├── url/
│   └── extracted/
│
├── processed/
│   ├── email/
│   ├── url/
│   └── master/
│
├── annotations/
├── splits/
└── metadata/