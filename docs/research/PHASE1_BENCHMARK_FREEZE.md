# Phase 1 Benchmark Freeze

## Adaptive Phishing Detection System

**Project:** Adaptive Phishing Detection  
**Phase:** Phase 1 — Data and Benchmark Foundation  
**Status:** COMPLETE  
**Integrity Status:** PASS  
**Mandatory Integrity Failures:** 0

---

## 1. Purpose

Phase 1 establishes the reproducible data and benchmark foundation for the
adaptive phishing detection system.

The objective of this phase was to acquire, inspect, normalize, validate,
clean, audit, and split the selected phishing-related datasets while
explicitly addressing data leakage and cross-split contamination.

Phase 1 is considered complete only after all mandatory integrity checks
successfully pass.

---

## 2. Phase 1 Scope

The following activities were completed during Phase 1:

1. Project structure creation
2. Dataset ecosystem definition
3. Dataset acquisition
4. Dataset inspection
5. Dataset normalization
6. Dataset validation and label auditing
7. Email cleaning and URL extraction
8. URL dataset acquisition
9. URL dataset inspection
10. URL normalization
11. URL validation
12. URL cleaning
13. Leakage discovery and correction
14. UCI phishing feature processing
15. UCI feature validation
16. UCI feature-vector-aware splitting
17. PhiUSIIL leakage auditing
18. PhiUSIIL domain-aware splitting
19. Email leakage auditing
20. Email text-group-disjoint splitting
21. Dataset metadata consolidation
22. Final Phase-1 integrity verification
23. Phase-1 benchmark freeze

---

## 3. Core Dataset Ecosystem

### 3.1 UCI Phishing Websites

Dataset:

`uci_phishing_websites`

Modality:

`url_features`

Role:

Core feature-based URL benchmark

Important characteristic:

The dataset contains engineered website/phishing features rather than
raw URL strings.

Therefore, it is treated as a feature-based phishing detection dataset and
is not used as a raw-URL dataset.

Project label convention:

- `0` = legitimate
- `1` = phishing

The processed dataset is stored as:

`data/processed/url/uci_phishing_websites_features.csv`

---

### 3.2 PhiUSIIL Phishing URL (Website)

Dataset:

`phiusiil`

Modality:

`url`

Role:

Core raw-URL benchmark

Important characteristic:

The dataset contains actual URL information and domain information.

Project label convention:

- `0` = legitimate
- `1` = phishing

The cleaned dataset is stored as:

`data/interim/url/phiusiil_cleaned.csv`

---

### 3.3 Email Dataset Collection

The email benchmark contains multiple publicly available email sources.

The normalized canonical schema includes:

- `sample_id`
- `source_dataset`
- `label`
- `subject`
- `body`
- `sender`
- `receiver`
- `timestamp`
- `raw_text`

Additional cleaned representation:

- `clean_text`
- extracted URL information
- URL count

Project label convention:

- `0` = legitimate
- `1` = phishing/spam/fraud according to source normalization

The email benchmark preserves the original dataset identity through the
`source_dataset` field.

---

## 4. Email Sources Processed

The following nine email datasets were successfully processed:

1. CEAS_08
2. Enron
3. Ling
4. Nazario
5. Nazario_5
6. Nigerian_5
7. Nigerian_Fraud
8. SpamAssasin
9. TREC_07

Two additional source files were retained but were previously found to be
unreadable during inspection:

- TREC_05
- TREC_06

These are documented exceptions and are not included in the final processed
email benchmark.

---

## 5. Data Cleaning

### Email Cleaning

Email preprocessing included:

- HTML unescaping
- HTML tag removal
- whitespace normalization
- construction of cleaned textual representations
- URL extraction
- URL counting

The cleaned email representation is retained separately from the original
source representation.

---

### URL Cleaning

URL preprocessing included:

- URL canonicalization
- extraction of domain information
- URL length calculation
- domain length calculation
- preservation of the original URL representation where applicable

---

## 6. Label Standardization

A project-wide binary label convention was established:

```text
0 = legitimate
1 = phishing