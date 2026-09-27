# PHM-XAI: Explainable and Uncertainty-Aware Prognostics for Predictive Maintenance

An end-to-end Prognostics and Health Management (PHM) research system for **Remaining Useful Life (RUL)** and **Health Index (HI)** estimation using the **NASA C-MAPSS turbofan engine dataset**.

The system combines deep-learning-based prognostics with **Explainable AI (XAI)**, **conformal uncertainty estimation**, **constraint-aware maintenance decision support**, and a **Human-in-the-Loop (HITL)** workflow.

> **Project Status:** Research implementation. The system is intended for academic experimentation and decision support and is not a production maintenance system.

---

## 1. Project Overview

Predictive maintenance aims to estimate the future health condition of equipment and identify when maintenance should be performed.

Traditional prognostic systems often provide a single prediction without explaining:

* how the prediction was generated,
* how uncertain the prediction is,
* which sensor observations influenced the prediction, or
* how the prediction should be converted into an operational maintenance decision.

PHM-XAI addresses these issues through an integrated pipeline:

```text
NASA C-MAPSS
     │
     ▼
Data Preprocessing
     │
     ▼
Sequence / Window Generation
     │
     ▼
Multi-Task Deep Learning
     │
     ├───────────────┐
     ▼               ▼
    RUL             HI
 Prediction      Prediction
     │               │
     └───────┬───────┘
             │
      ┌──────┴──────┐
      ▼             ▼
 Uncertainty       XAI
 Estimation       Analysis
      │             │
      │        ┌────┴─────────────┐
      │        ▼        ▼         ▼
      │       SHAP      IG    Temporal
      │                       Relevance
      │                           │
      │                           ▼
      │                          ERI
      │
      └────────────┬──────────────┘
                   ▼
          Maintenance Decision
              Engine
                   │
                   ▼
        Operational Constraints
                   │
                   ▼
       Maintenance Recommendation
                   │
                   ▼
            Human Review
                   │
                   ▼
          Expert Feedback
                   │
                   ▼
          Policy Refinement
```

The system therefore treats prognostics as a **decision-support chain**, rather than as a standalone RUL prediction model.

---

# 2. Research Objectives

The project has the following objectives:

1. Develop deep-learning models for **RUL prediction** of turbofan engines.
2. Estimate a continuous **Health Index (HI)** representing degradation.
3. Compare different sequence-learning architectures.
4. Quantify prediction uncertainty using conformal prediction methods.
5. Explain model predictions using multiple XAI techniques.
6. Estimate explanation reliability using the **Explainability Reliability Index (ERI)**.
7. Convert predictions, uncertainty and explanations into auditable maintenance recommendations.
8. Incorporate expert feedback through a Human-in-the-Loop workflow.
9. Maintain engine-disjoint evaluation to reduce data leakage.
10. Provide reproducible artifacts, reports and experiment outputs.

---

# 3. Dataset

The project uses the **NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation)** turbofan engine dataset.

Four operating subsets are supported:

* FD001
* FD002
* FD003
* FD004

Each subset contains simulated turbofan engine run-to-failure trajectories with:

* engine identifiers,
* operational settings,
* sensor measurements,
* training trajectories,
* test trajectories,
* ground-truth RUL values for the test set.

The raw files are expected under:

```text
data/raw/
├── train_FD001.txt
├── test_FD001.txt
├── RUL_FD001.txt
├── train_FD002.txt
├── test_FD002.txt
├── RUL_FD002.txt
├── train_FD003.txt
├── test_FD003.txt
├── RUL_FD003.txt
├── train_FD004.txt
├── test_FD004.txt
└── RUL_FD004.txt
```

The raw NASA C-MAPSS data is not committed to the repository and must be added locally.

---

# 4. System Architecture

The system consists of six major components:

### 4.1 Data Processing

Responsible for:

* loading C-MAPSS data,
* validating input data,
* identifying individual engines,
* generating RUL labels,
* selecting usable features,
* scaling sensor values,
* generating temporal windows.

### 4.2 Prognostic Models

The project supports multiple deep-learning architectures:

* LSTM
* GRU
* Transformer
* Hybrid architecture
* GRU-Att-Deg

The models perform joint prediction of:

* Remaining Useful Life (RUL)
* Health Index (HI)

### 4.3 Uncertainty Estimation

Prediction intervals are generated using conformal calibration approaches.

The repository currently implements:

* EARA-Conformal
* Adaptive Conformal
* Split Conformal
* CQR-style scale-based calibration
* Engine-Joint Conformal

### 4.4 Explainability

Model predictions are analyzed using:

* Integrated Gradients
* SHAP
* Temporal Relevance
* Explainability Reliability Index (ERI)

These methods provide information about the sensor and temporal evidence influencing predictions.

### 4.5 Decision Engine

The decision engine combines:

* predicted RUL,
* predicted HI,
* prediction interval,
* uncertainty level,
* explanation reliability,
* risk score,
* urgency,
* operational constraints.

It produces an auditable maintenance recommendation.

### 4.6 Human-in-the-Loop

Human experts can:

* review flagged decisions,
* approve AI recommendations,
* override recommendations,
* provide confidence and reasoning.

Expert feedback is accumulated and can be used for conservative **decision-policy refinement**.

> HITL currently does **not** retrain the underlying prognostic model.

---

# 5. End-to-End Data Flow

The complete pipeline is:

```text
NASA C-MAPSS
      │
      ▼
Data Ingestion
      │
      ▼
Validation
      │
      ▼
RUL Label Generation
      │
      ▼
Feature Selection
      │
      ▼
Scaling
      │
      ▼
Sliding-Window Generation
      │
      ▼
Train / Validation / Calibration / Test
      │
      ▼
Deep Learning Model
      │
      ├───────────────┐
      ▼               ▼
     RUL              HI
      │               │
      └───────┬───────┘
              ▼
       Uncertainty Layer
              │
              ▼
       Explainability Layer
              │
              ▼
       Decision Engine
              │
              ▼
    Operational Constraints
              │
              ▼
 Maintenance Recommendation
              │
              ▼
       Human Review
              │
              ▼
     Expert Feedback
              │
              ▼
      Policy Refinement
```

Two important data-integrity principles are maintained:

### Engine-disjoint splitting

Windows belonging to the same engine are kept within the same dataset split to reduce leakage caused by correlated observations from the same engine.

### Calibration isolation

Conformal calibration is performed using a calibration split rather than the final test targets. The test set is reserved for evaluating coverage and prediction performance.

---

# 6. Preprocessing Pipeline

The preprocessing stage converts raw C-MAPSS trajectories into model-ready temporal sequences.

## 6.1 Data Ingestion

The preprocessing pipeline reads the C-MAPSS text files and constructs structured records containing engine IDs, operating conditions and sensor measurements.

## 6.2 Validation

Input data is validated before further processing.

Validation includes:

* expected columns,
* engine identifiers,
* sensor values,
* trajectory structure,
* missing or invalid values,
* compatibility between training and test data.

## 6.3 RUL Label Generation

For a training trajectory, RUL is derived from the remaining number of operational cycles before the engine reaches the end of its trajectory.

The resulting target is used for supervised RUL prediction.

## 6.4 Feature Selection

The preprocessing stage selects the usable operational and sensor variables required by the model.

The feature-selection configuration is applied consistently when creating training and evaluation windows.

## 6.5 Scaling

Sensor and operating-condition variables are scaled before being provided to the neural network.

Scaling parameters are learned from the appropriate training data and reused during validation/test processing.

## 6.6 Sliding Windows

The continuous engine trajectories are converted into fixed-length temporal windows.

Conceptually:

```text
Engine trajectory:

t1 ─ t2 ─ t3 ─ t4 ─ t5 ─ t6 ─ t7 ─ ...

Window 1:
[t1 t2 t3 t4 ...]

Window 2:
   [t2 t3 t4 t5 ...]

Window 3:
      [t3 t4 t5 t6 ...]
```

Each window represents the recent operating history of an engine and becomes one model input sample.

---

# 7. Model Design

The repository contains five model families.

## 7.1 LSTM

Long Short-Term Memory networks are used to capture temporal dependencies in sensor sequences.

```text
Input Sequence
      │
      ▼
     LSTM
      │
      ▼
Temporal Representation
      │
 ┌────┴────┐
 ▼         ▼
RUL       HI
Head      Head
```

---

## 7.2 GRU

The Gated Recurrent Unit architecture provides recurrent temporal modeling with a simpler gating mechanism than LSTM.

```text
Input Sequence
      │
      ▼
     GRU
      │
      ▼
Sequence Representation
      │
 ┌────┴────┐
 ▼         ▼
RUL       HI
```

---

## 7.3 Transformer

The Transformer architecture uses attention mechanisms to model relationships between different time steps in the sensor sequence.

```text
Input Sequence
      │
      ▼
Feature / Positional Representation
      │
      ▼
Transformer Encoder
      │
      ▼
Temporal Representation
      │
 ┌────┴────┐
 ▼         ▼
RUL       HI
```

---

## 7.4 Hybrid Model

The hybrid architecture combines multiple sequence-processing mechanisms to capture complementary temporal characteristics.

The exact configuration is defined by the implementation under `src/models/`.

---

## 7.5 GRU-Att-Deg

The GRU-Att-Deg architecture combines:

* GRU-based temporal encoding,
* temporal attention,
* RUL prediction,
* degradation/health-index prediction.

Conceptually:

```text
Input Sensor Window
        │
        ▼
    GRU Encoder
        │
        ▼
Temporal Attention
        │
        ▼
 Context Vector
      │   │
      │   └─────────────► HI / Degradation Head
      │
      └─────────────────► RUL Head
```

This architecture provides a learned representation of the most relevant temporal information for prognostic prediction.

---

# 8. Multi-Task Prediction

The system jointly estimates two related outputs:

### Remaining Useful Life

RUL estimates the expected number of operational cycles remaining before failure/end-of-life.

### Health Index

HI provides a continuous representation of the degradation or health state of the engine.

The multi-task formulation can be represented as:

```text
                 Shared Representation
                         │
             ┌───────────┴───────────┐
             ▼                       ▼
        RUL Prediction          HI Prediction
             │                       │
             ▼                       ▼
          RUL Value               HI Value
```

Joint prediction allows the model to learn both remaining life and degradation behavior.

---

# 9. Training and Evaluation

The project uses PyTorch-based deep-learning models.

Training involves:

* sequence-based input data,
* supervised RUL/HI targets,
* configurable optimization,
* validation monitoring,
* checkpoint generation,
* evaluation on held-out engine trajectories.

Supported model names:

```text
lstm
gru
transformer
hybrid
gru_att_deg
```

Supported subsets:

```text
FD001
FD002
FD003
FD004
```

Example:

```bash
python scripts/train.py --model gru --subset FD001
```

Evaluation:

```bash
python scripts/evaluate.py --model gru --subset FD001
```

The project reports standard regression metrics such as:

* MAE
* RMSE
* R²
* NASA scoring metric

Results should be interpreted together with the corresponding model, subset, checkpoint and experimental configuration.

---

# 10. Explainable AI (XAI)

A key objective of PHM-XAI is to provide evidence for model predictions rather than treating the neural network as a black box.

The project uses multiple complementary explanation techniques.

## 10.1 Integrated Gradients

Integrated Gradients attributes the model output to input features by integrating gradients along a path from a baseline input to the observed input.

It provides feature-level attribution for RUL and/or HI predictions.

---

## 10.2 SHAP

SHAP is used to estimate the contribution of input features to the model output.

The resulting values can be used to identify sensors that have a strong positive or negative influence on the prediction.

---

## 10.3 Temporal Relevance

Temporal relevance analyzes which portions of the input sequence contribute most strongly to the prediction.

This is particularly useful for time-series prognostics because both:

* **which sensor**
* **when the sensor observation occurred**

can affect a prediction.

---

## 10.4 Explainability Reliability Index

The **Explainability Reliability Index (ERI)** summarizes agreement between multiple explanation methods.

The project compares evidence from:

* Integrated Gradients,
* SHAP,
* temporal relevance.

A higher agreement indicates stronger consistency between the explanation methods, while disagreement can be used as a warning signal for decision support.

Conceptually:

```text
              Model Prediction
                     │
        ┌────────────┼────────────┐
        ▼            ▼            ▼
 Integrated        SHAP       Temporal
 Gradients                    Relevance
        │            │            │
        └────────────┼────────────┘
                     ▼
                    ERI
                     │
                     ▼
        Explanation Reliability
```

XAI artifacts are stored under:

```text
outputs/xai/
```

---

# 11. Uncertainty Estimation

A point prediction alone does not indicate how confident the system should be in its estimate.

PHM-XAI therefore generates prediction intervals using conformal calibration.

For example:

```text
Predicted RUL = 35 cycles

Prediction interval:

Lower bound = 24 cycles
Prediction   = 35 cycles
Upper bound = 47 cycles
```

The interval provides an estimate of the range within which the true RUL is expected to fall at a specified calibration level.

## Implemented uncertainty variants

| Variant                  | Description                                       |
| ------------------------ | ------------------------------------------------- |
| `eara_conformal`         | Regime- and scale-adaptive conformal calibration  |
| `adaptive_conformal`     | Global scale-adaptive calibration                 |
| `conformal`              | Global split-conformal baseline                   |
| `cqr`                    | Scale-based conformalized interval implementation |
| `engine_joint_conformal` | Calibration using engine-level scores             |

Example:

```bash
python scripts/uncertainty.py \
    --model gru \
    --subset FD001 \
    --n_regimes 7
```

Coverage is evaluated at nominal:

* 80%
* 90%
* 95%

Reported uncertainty metrics include:

* empirical coverage,
* coverage error,
* Mean Prediction Interval Width (MPIW),
* normalized prediction interval width,
* per-regime coverage,
* engine-level coverage.

The uncertainty artifacts are stored under:

```text
outputs/results/uncertainty/
```

---

# 12. Maintenance Decision Workflow

The maintenance decision engine converts model outputs into operational recommendations.

The decision engine considers:

* RUL,
* HI,
* prediction interval,
* uncertainty,
* ERI,
* risk,
* urgency,
* operational constraints.

The workflow is:

```text
RUL + HI
   │
   ▼
Uncertainty Interval
   │
   ▼
ERI / Explanation Reliability
   │
   ▼
Risk Assessment
   │
   ▼
Health & Uncertainty Classification
   │
   ▼
Operational Constraints
   │
   ▼
Maintenance Recommendation
```

## Available actions

The system currently supports:

```text
CONTINUE_OPERATION
MONITOR
INSPECT
SCHEDULE_MAINTENANCE
URGENT_MAINTENANCE
```

The decision output contains information such as:

* health state,
* uncertainty level,
* explanation reliability,
* risk score,
* urgency index,
* risk components,
* reasons,
* constraint violations,
* human-review flag.

---

# 13. Operational Constraints

Maintenance decisions can be affected by real-world operational limitations.

Supported constraints include:

| Constraint              | Purpose                                                   |
| ----------------------- | --------------------------------------------------------- |
| Maintenance window      | Determines whether maintenance can currently be performed |
| Minimum safe RUL        | Minimum remaining life required to defer maintenance      |
| Maintenance lead time   | Required preparation time                                 |
| Maximum allowable delay | Maximum acceptable waiting period                         |
| Resource availability   | Availability of maintenance personnel and parts           |

Example:

```bash
python scripts/decision.py \
    --subset FD001 \
    --minimum-safe-rul 10 \
    --maintenance-lead-time 10 \
    --no-maintenance-window
```

Constraints are designed to make recommendations more conservative rather than less conservative.

---

# 14. Human-in-the-Loop Workflow

Some predictions can be flagged for human review when uncertainty, explanation reliability or operational risk requires additional assessment.

The workflow is:

```text
AI Recommendation
       │
       ▼
Human Review Required?
       │
   ┌───┴────┐
   │        │
  No       Yes
   │        │
   │        ▼
   │   Expert Review
   │        │
   │   ┌────┴────┐
   │   ▼         ▼
   │ APPROVE   OVERRIDE
   │             │
   └──────┬──────┘
          ▼
    Expert Feedback
          │
          ▼
   Policy Refinement
```

Example:

```bash
python scripts/hitl.py --subset FD001 --model gru
```

A reviewer can approve an AI recommendation or override it with an alternative action.

Expert feedback is stored in:

```text
outputs/hitl/feedback.jsonl
```

### Important implementation boundary

The current HITL implementation **does not retrain the prognostic neural network**.

Instead, expert feedback is used for conservative **decision-policy refinement**.

Model retraining remains a future extension.

---

# 15. Model Monitoring and Retraining

The current implementation provides decision-level monitoring through:

* prediction uncertainty,
* explanation reliability,
* decision risk,
* human review,
* expert feedback.

However, automated model retraining is currently outside the implemented HITL workflow.

A future production-oriented extension could introduce:

```text
Prediction Monitoring
        │
        ▼
Performance / Data Drift Detection
        │
        ▼
Retraining Trigger
        │
        ▼
Model Retraining
        │
        ▼
Validation
        │
        ▼
Model Approval
        │
        ▼
Deployment
```

This should be considered a **future system capability**, not a currently implemented feature.

---

# 16. Repository Structure

```text
phm_project/
│
├── ARCHITECTURE.md
├── README.md
├── main.py
├── requirements.txt
│
├── configs/
│   ├── fd001.yaml
│   ├── fd002.yaml
│   ├── fd003.yaml
│   └── fd004.yaml
│
├── data/
│   ├── external/
│   ├── raw/
│   └── processed/
│
├── scripts/
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   ├── uncertainty.py
│   ├── explain.py
│   ├── decision.py
│   ├── hitl.py
│   └── regime_sweep.py
│
├── src/
│   ├── preprocessing/
│   ├── models/
│   ├── uncertainty/
│   ├── explainability/
│   ├── decision_engine/
│   ├── hitl/
│   ├── evaluation/
│   └── utils/
│
├── tests/
│
└── outputs/
    ├── checkpoints/
    ├── figures/
    ├── hitl/
    ├── logs/
    ├── models/
    ├── reports/
    ├── results/
    └── xai/
```

---

# 17. Installation

## Requirements

* Python 3.9+
* PyTorch
* NumPy
* Pandas
* scikit-learn
* SciPy
* SHAP
* Captum
* PyYAML
* tqdm
* joblib
* Matplotlib
* Seaborn

Create a virtual environment:

```bash
python -m venv .venv
```

### Windows PowerShell

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

The repository requirements currently specify a CUDA-enabled PyTorch build. Users running CPU-only systems or a different CUDA version should install an appropriate PyTorch build before installing the remaining dependencies.

---

# 18. Running the Pipeline

## Step 1 — Exploratory Data Analysis

```bash
python main.py
```

The current EDA entry point operates on FD001.

## Step 2 — Preprocessing

For one subset:

```bash
python scripts/preprocess.py --subset FD001
```

For all subsets:

```bash
python scripts/preprocess.py --subset all
```

## Step 3 — Train a Model

```bash
python scripts/train.py \
    --model gru \
    --subset FD001
```

## Step 4 — Evaluate

```bash
python scripts/evaluate.py \
    --model gru \
    --subset FD001
```

## Step 5 — Generate Uncertainty Intervals

```bash
python scripts/uncertainty.py \
    --model gru \
    --subset FD001 \
    --n_regimes 7
```

## Step 6 — Generate Explanations

```bash
python scripts/explain.py \
    --model gru_att_deg \
    --subset FD001 \
    --target rul
```

For all supported models and subsets:

```bash
python scripts/explain.py \
    --model all \
    --subset all \
    --target all
```

## Step 7 — Generate Maintenance Decisions

```bash
python scripts/decision.py \
    --subset FD001 \
    --model gru
```

## Step 8 — Human-in-the-Loop Review

```bash
python scripts/hitl.py \
    --subset FD001 \
    --model gru
```

---

# 19. Experimental Results

Results are dependent on:

* dataset subset,
* preprocessing configuration,
* model architecture,
* checkpoint,
* random seed,
* training configuration.

Therefore, reported values should be reproduced before being used as formal benchmark results.

Example results from project runs include:

| Experiment                | Result                                                          |
| ------------------------- | --------------------------------------------------------------- |
| FD001 — GRU-Att-Deg       | MAE: 10.05, RMSE: 13.64, R²: 0.884, NASA Score: 308.63          |
| FD004 — Transformer       | MAE: 13.56, R²: 0.808                                           |
| GRU split conformal — 90% | Coverage evaluated across FD001–FD004                           |
| Mean ERI                  | Approximately 0.88 for HI and 0.87 for RUL across reported runs |

These values are **experiment-specific** and should not be interpreted as universal model performance.

---

# 20. Current Implementation Status

| Component                           | Status          |
| ----------------------------------- | --------------- |
| Data preprocessing and windowing    | Complete        |
| Joint RUL/HI prediction             | Implemented     |
| LSTM                                | Implemented     |
| GRU                                 | Implemented     |
| Transformer                         | Implemented     |
| Hybrid model                        | Implemented     |
| GRU-Att-Deg                         | Implemented     |
| Conformal uncertainty               | Implemented     |
| EARA-Conformal                      | Implemented     |
| Integrated Gradients                | Implemented     |
| SHAP                                | Implemented     |
| Temporal relevance                  | Implemented     |
| ERI                                 | Implemented     |
| Decision engine                     | Implemented     |
| Operational constraints             | Implemented     |
| HITL review                         | Implemented     |
| Decision-policy refinement          | Implemented     |
| Automated neural-network retraining | Not implemented |
| Multi-seed uncertainty validation   | Pending         |
| XAI faithfulness validation         | Pending         |
| Full evaluation/ablation framework  | Pending         |
| Configuration-file loading          | Not implemented |

---

# 21. Known Limitations

The current research implementation has several limitations.

### Configuration Management

The YAML files under `configs/` currently describe intended configurations but are not fully connected to the execution pipeline. Some scripts use module-level constants.

### Uncertainty Validation

Multi-seed and engine-level validation of adaptive uncertainty approaches remains an area for further experimentation.

### Existing Checkpoints

Previously trained checkpoints may not contain a properly trained scale head required by adaptive uncertainty estimation. Adaptive interval results should therefore be regenerated using checkpoints trained with the current loss configuration.

### Evaluation Framework

Some components under `src/evaluation/` remain placeholders. Core metrics are implemented elsewhere in the model evaluation code.

### CQR Terminology

The current `cqr` implementation is a scale-based conformalized interval approach and does not implement conventional trained lower/upper quantile heads.

### Production Deployment

The project is an academic research implementation and does not currently provide production-grade deployment, live sensor ingestion, automated model retraining or safety certification.

---

# 22. Future Scope

Future development may include:

* automated configuration management,
* automated model retraining,
* data and concept drift detection,
* continuous monitoring,
* automated experiment tracking,
* multi-seed benchmarking,
* XAI faithfulness evaluation,
* ablation studies,
* robustness testing,
* real-time sensor integration,
* production deployment,
* model versioning,
* CI/CD integration,
* automated model validation and approval.

A future production workflow could be:

```text
Live Sensor Data
       │
       ▼
Data Quality Monitoring
       │
       ▼
PHM Model
       │
       ├──────────► RUL / HI
       │
       ├──────────► Uncertainty
       │
       └──────────► XAI / ERI
                    │
                    ▼
             Decision Engine
                    │
                    ▼
             Human Oversight
                    │
                    ▼
          Monitoring / Feedback
                    │
                    ▼
              Drift Detection
                    │
                    ▼
            Retraining Pipeline
                    │
                    ▼
             Model Validation
                    │
                    ▼
              Model Deployment
```

---

# 23. Testing

Run the test suite using:

```bash
python -m pytest -q
```

The main regression tests cover decision and uncertainty functionality.

| Test                          | Purpose                                                     |
| ----------------------------- | ----------------------------------------------------------- |
| `tests/test_decision.py`      | Decision rules, constraints, risk scoring and HITL behavior |
| `tests/test_uncertainty.py`   | Conformal quantiles, interval clipping and coverage         |
| `tests/test_preprocessing.py` | Preprocessing and short-engine handling                     |
| `tests/test_models.py`        | Model-related tests; currently limited                      |
| `tests/test_xai.py`           | XAI-related tests; currently limited                        |

---

# 24. Generated Artifacts

Generated outputs are stored under:

```text
outputs/
├── checkpoints/     # trained model checkpoints
├── figures/         # EDA and experiment figures
├── hitl/            # expert feedback
├── logs/            # execution logs
├── models/          # fitted scalers
├── reports/         # HTML reports and summaries
├── results/         # predictions, metrics, intervals and decisions
└── xai/             # XAI outputs and ERI reports
```

These outputs provide traceability between:

```text
Input Data
    ↓
Model
    ↓
Prediction
    ↓
Uncertainty
    ↓
Explanation
    ↓
Decision
    ↓
Human Feedback
```

---

# 25. Research Contribution

The project combines several components into a unified prognostics decision-support workflow:

1. **Joint RUL and Health Index prediction**
2. **Multiple deep-learning sequence architectures**
3. **Engine-disjoint experimental methodology**
4. **Conformal uncertainty estimation**
5. **Multi-method Explainable AI**
6. **Explainability Reliability Index**
7. **Constraint-aware maintenance decisions**
8. **Human-in-the-Loop decision review**
9. **Auditable decision and explanation artifacts**

The overall objective is not only to predict when an engine may fail, but to provide a more transparent decision-support process:

```text
Prediction
    +
Uncertainty
    +
Explanation
    +
Operational Context
    +
Human Oversight
    =
Trustworthy Prognostic Decision Support
```

---

# 26. Documentation

The repository contains additional internal technical documentation:

* `README.md` — project overview, setup and execution
* `ARCHITECTURE.md` — detailed system architecture, module responsibilities, artifact contracts, uncertainty methodology, decision policy and implementation gaps

The README provides the high-level project documentation, while `ARCHITECTURE.md` provides deeper implementation details.

---

# 27. License and Academic Use

This repository is intended for academic and research purposes.

When using the project, dataset, models or results in academic work, ensure that the original NASA C-MAPSS dataset and any referenced research methods are properly cited.

---

## Project Status

**PHM-XAI is an ongoing academic research project.**

The current implementation demonstrates an integrated workflow from:

**NASA C-MAPSS → Preprocessing → Deep Learning → RUL/HI → Uncertainty → XAI → Maintenance Decision → Human Review**

while clearly separating currently implemented functionality from future research extensions.
