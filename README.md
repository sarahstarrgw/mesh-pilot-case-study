# MESH Pilot Case Study

Supporting materials for **“The MESH Environment: Multi-level Evaluation of Sociotechnical HAI-workflows,”** prepared for submission to *Systems Engineering* (Wiley).

## Overview

MESH provides a reference architecture for a multilevel evaluation environment for sociotechnical human–AI workflows. The paper uses a February technical pilot as a case study to demonstrate how the definition of the **system under test** affects which failures and behaviors are observable.

This repository brings together evaluation data for all three levels tested.

## Evaluation levels

| Level | Folder | Materials |
| --- | --- | --- |
| **Model-level benchmarking** | `01_model_benchmarking/` | Model-level evaluation data and correctness scores. |
| **Model field testing** | `02_model_field_testing/` | 78 observed interactions from field testing, anonymized as described below. |
| **System-level outcomes** | `03_system_outcomes/` | Final task submissions and system-level correctness assessments. |

## Organization

### `01_model_benchmarking/`

- **`code/`** contains model-level benchmarking and evaluation code.
- **`data/`** contains the curated test set, assessment criteria, raw model responses, and scoring outputs described below.

| Data file or naming pattern | Contents |
| --- | --- |
| `ai_advisor_test_set.csv` | Curated test set and assessment criteria used for model-level benchmarking. |
| `benchmark_results_TIMESTAMP` | Raw responses from independent, single-turn model-level benchmarking. |
| `eval_results_benchmark_results_TIMESTAMP` | Scorer results produced by the deterministic evaluation functions. |
| `judge_results_TIMESTAMP` | LLM-as-judge scoring results; these are **outside the scope of the paper**. |

`TIMESTAMP` represents the timestamp in each output's filename. The patterns above omit file extensions.

### `02_model_field_testing/`

Contains the 78 observed interactions from field testing. Identifying information and exact language have been removed for anonymization; these records are not verbatim transcripts.

### `03_system_outcomes/`

Contains final task submissions and system-level correctness assessments.

## Contact

For questions, contact Sarah Starr at [sarah.starr@gwu.edu](mailto:sarah.starr@gwu.edu).
