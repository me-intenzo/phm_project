# PHM-XAI System Architecture

## 1. Purpose

This document describes the technical architecture of the **PHM-XAI** research system.

The system is designed to provide an end-to-end workflow for **Prognostics and Health Management (PHM)** using the NASA C-MAPSS turbofan engine dataset.

The architecture integrates:

* Data preprocessing
* Temporal sequence generation
* Deep-learning-based RUL prediction
* Health Index estimation
* Uncertainty quantification
* Explainable AI
* Explanation reliability assessment
* Maintenance decision support
* Operational constraints
* Human-in-the-Loop review
* Decision-policy refinement

The architecture is designed as a modular research system so that individual components can be independently developed, evaluated and replaced.

---

# 2. High-Level Architecture

The overall system follows the pipeline:

```text
┌───────────────────────────────┐
│       NASA C-MAPSS Data       │
│   FD001 / FD002 / FD003 /    │
│            FD004              │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│      Data Preprocessing       │
│                               │
│ • Data ingestion              │
│ • Validation                  │
│ • RUL labeling                │
│ • Feature selection           │
│ • Scaling                     │
│ • Sliding-window generation  │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│     Temporal Deep Model       │
│                               │
│ • LSTM                        │
│ • GRU                         │
│ • Transformer                 │
│ • Hybrid                      │
│ • GRU-Att-Deg                 │
└───────────────┬───────────────┘
                │
        ┌───────┴────────┐
        │                │
        ▼                ▼
┌─────────────┐   ┌─────────────┐
│ RUL Head    │   │ HI Head     │
│             │   │             │
│ RUL value   │   │ Health      │
│             │   │ Index       │
└──────┬──────┘   └──────┬──────┘
       │                 │
       └────────┬────────┘
                │
        ┌───────┴──────────┐
        │                  │
        ▼                  ▼
┌────────────────┐  ┌────────────────────┐
│   Uncertainty  │  │       XAI          │
│   Estimation   │  │                    │
│                │  │ • SHAP             │
│ • Conformal    │  │ • Integrated       │
│   prediction   │  │   Gradients        │
│ • Adaptive     │  │ • Temporal         │
│   intervals    │  │   Relevance        │
└───────┬────────┘  └─────────┬──────────┘
        │                     │
        │                     ▼
        │              ┌──────────────┐
        │              │     ERI      │
        │              │ Explanation  │
        │              │ Reliability  │
        │              └──────┬───────┘
        │                     │
        └──────────┬──────────┘
                   ▼
        ┌───────────────────────┐
        │   Decision Engine     │
        │                       │
        │ • Risk                │
        │ • Urgency             │
        │ • Health state        │
        │ • Uncertainty         │
        │ • ERI                 │
        │ • Constraints         │
        └───────────┬───────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │ Maintenance Decision  │
        │                       │
        │ • Continue Operation  │
        │ • Monitor             │
        │ • Inspect             │
        │ • Schedule Maintenance│
        │ • Urgent Maintenance  │
        └───────────┬───────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │ Human-in-the-Loop     │
        │       Review          │
        └───────────┬───────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │ Expert Feedback       │
        │                       │
        │ Decision Policy       │
        │ Refinement            │
        └───────────────────────┘
```

---

# 3. Architectural Principles

The system follows several architectural principles.

## 3.1 Modularity

Major functionality is separated into modules:

```text
src/
├── preprocessing/
├── models/
├── uncertainty/
├── explainability/
├── decision_engine/
├── hitl/
├── evaluation/
└── utils/
```

Each module has a specific responsibility.

---

## 3.2 Separation of Prediction and Decision

The prognostic model does not directly determine the final maintenance action.

Instead:

```text
Model
  ↓
Prediction
  ↓
Uncertainty + Explanation
  ↓
Decision Engine
  ↓
Maintenance Recommendation
```

This separation allows the predictive model and operational decision logic to be evaluated independently.

---

## 3.3 Engine-Disjoint Evaluation

C-MAPSS contains multiple observations from the same engine.

Because windows generated from the same engine are correlated, random window-level splitting can result in information leakage.

The architecture therefore uses **engine-disjoint splitting**.

Conceptually:

```text
Engine IDs
    │
    ├── Training Engines
    │
    ├── Validation Engines
    │
    ├── Calibration Engines
    │
    └── Test Engines
```

Windows generated from a particular engine remain in the corresponding split.

This is particularly important for reliable evaluation of uncertainty and generalization.

---

# 4. Data Layer

## 4.1 Input

The primary input is the NASA C-MAPSS turbofan dataset.

Supported subsets:

```text
FD001
FD002
FD003
FD004
```

Each trajectory contains:

* Engine ID
* Operational cycle
* Operational settings
* Sensor measurements

Training trajectories contain run-to-failure information from which RUL targets can be generated.

---

# 5. Preprocessing Architecture

The preprocessing pipeline converts raw trajectories into model-ready tensors.

```text
Raw C-MAPSS
     │
     ▼
Data Loader
     │
     ▼
Validation
     │
     ▼
Feature Selection
     │
     ▼
RUL Label Generation
     │
     ▼
Scaling
     │
     ▼
Sliding Windows
     │
     ▼
Dataset Splitting
     │
     ▼
Model Input
```

---

## 5.1 Data Ingestion

Raw C-MAPSS text files are loaded into structured tabular representations.

The loader identifies:

* engine ID,
* cycle,
* operating conditions,
* sensor variables.

---

## 5.2 Data Validation

Validation is performed before model training.

The preprocessing layer checks the structural consistency of the dataset and ensures that the required variables are available.

---

## 5.3 RUL Generation

RUL is derived from the remaining operational cycles of an engine trajectory.

Conceptually:

```text
RUL = Final Cycle - Current Cycle
```

For example:

```text
Engine final cycle = 150

Current cycle = 120

RUL = 150 - 120
    = 30 cycles
```

The generated RUL becomes the supervised target for the prediction model.

---

## 5.4 Feature Selection

The preprocessing layer selects the sensor and operational variables used by the model.

Feature selection is kept consistent between training and inference.

---

## 5.5 Scaling

Input features are normalized/scaled before being passed to the neural network.

Scaling parameters are fitted using the appropriate training data and subsequently reused for validation, calibration and testing.

This prevents information from the test set from influencing preprocessing.

---

# 6. Temporal Windowing

C-MAPSS provides continuous engine trajectories.

Neural networks require fixed-size temporal inputs.

Therefore, the preprocessing pipeline converts each trajectory into sliding windows.

Example:

```text
Original trajectory:

t1 ─ t2 ─ t3 ─ t4 ─ t5 ─ t6 ─ t7

Window size = 4

Window 1:
[t1 t2 t3 t4]

Window 2:
   [t2 t3 t4 t5]

Window 3:
      [t3 t4 t5 t6]

Window 4:
         [t4 t5 t6 t7]
```

Each window contains a sequence of sensor observations.

The resulting tensor can be represented as:

```text
(batch_size, sequence_length, number_of_features)
```

---

# 7. Dataset Split Architecture

The system separates data according to engine identity.

A conceptual split is:

```text
                  C-MAPSS
                     │
                     ▼
              Engine IDs
                     │
       ┌─────────────┼─────────────┐
       │             │             │
       ▼             ▼             ▼
    Training      Validation    Calibration
       │             │             │
       └─────────────┬─────────────┘
                     │
                     ▼
                   Test
```

The test set remains isolated until final evaluation.

The calibration set is used for uncertainty estimation and is not treated as the final test set.

---

# 8. Model Layer

The model layer provides multiple temporal architectures.

```text
src/models/
```

Supported architectures include:

* LSTM
* GRU
* Transformer
* Hybrid
* GRU-Att-Deg

---

# 9. LSTM Architecture

LSTM provides recurrent sequence modeling with memory cells and gating mechanisms.

```text
Input Sequence
      │
      ▼
┌─────────────┐
│    LSTM     │
└──────┬──────┘
       │
       ▼
Temporal Representation
       │
  ┌────┴────┐
  ▼         ▼
 RUL        HI
Head       Head
```

The shared temporal representation can be used for both prognostic outputs.

---

# 10. GRU Architecture

GRU provides recurrent temporal modeling using update and reset gates.

```text
Input Sequence
      │
      ▼
┌─────────────┐
│     GRU     │
└──────┬──────┘
       │
       ▼
Temporal Representation
       │
  ┌────┴────┐
  ▼         ▼
 RUL        HI
```

GRU provides a relatively compact recurrent architecture compared with LSTM.

---

# 11. Transformer Architecture

The Transformer architecture uses attention mechanisms to model temporal dependencies.

```text
Input Sequence
      │
      ▼
Feature Representation
      │
      ▼
Positional Representation
      │
      ▼
Transformer Encoder
      │
      ▼
Temporal Representation
      │
  ┌───┴────┐
  ▼        ▼
 RUL       HI
```

Attention allows the model to identify relationships between different time steps in the sequence.

---

# 12. Hybrid Architecture

The hybrid architecture combines complementary temporal modeling mechanisms.

Its implementation-specific details are defined in:

```text
src/models/
```

The purpose is to capture temporal characteristics that may not be adequately represented by a single sequence architecture.

---

# 13. GRU-Att-Deg Architecture

The GRU-Att-Deg architecture combines:

* GRU encoding,
* temporal attention,
* RUL prediction,
* degradation/HI prediction.

```text
Input Sequence
      │
      ▼
GRU Encoder
      │
      ▼
Temporal Attention
      │
      ▼
Context Vector
      │
 ┌────┴─────────┐
 ▼              ▼
RUL Head     HI/Deg Head
 │              │
 ▼              ▼
RUL            HI
```

The attention mechanism provides a mechanism for identifying important temporal regions of the input sequence.

---

# 14. Multi-Task Learning

The system performs joint prediction.

Instead of training separate models for RUL and HI, the architecture can learn a shared temporal representation.

```text
                Input Sequence
                      │
                      ▼
             Shared Encoder
                      │
             ┌────────┴────────┐
             ▼                 ▼
         RUL Head           HI Head
             │                 │
             ▼                 ▼
          RUL Output       HI Output
```

This enables the model to learn representations associated with both remaining life and degradation state.

---

# 15. Training Architecture

The training pipeline consists of:

```text
Preprocessed Dataset
        │
        ▼
Batch Generation
        │
        ▼
Model Forward Pass
        │
        ▼
RUL / HI Predictions
        │
        ▼
Loss Calculation
        │
        ▼
Backpropagation
        │
        ▼
Optimizer Update
        │
        ▼
Validation
        │
        ▼
Checkpoint
```

The training implementation is primarily based on PyTorch.

Training configuration includes model-specific parameters such as:

* learning rate,
* batch size,
* hidden dimensions,
* number of layers,
* dropout,
* number of epochs,
* optimizer,
* loss configuration.

The exact values used by an experiment should be obtained from the corresponding training script/configuration rather than assumed from this architecture document.

---

# 16. Prediction Layer

After training, the model produces:

```text
Input Window
     │
     ▼
Neural Network
     │
     ├──────────────┐
     ▼              ▼
    RUL             HI
Prediction       Prediction
```

These outputs are passed to the uncertainty and explainability components.

---

# 17. Uncertainty Architecture

A key design principle is that the system does not treat the point prediction as the complete output.

Instead:

```text
Point Prediction
      │
      ▼
Conformal Calibration
      │
      ▼
Prediction Interval
```

The system supports several uncertainty approaches:

```text
src/uncertainty/
```

Including:

* Split Conformal
* Adaptive Conformal
* EARA-Conformal
* CQR-style scale-based calibration
* Engine-Joint Conformal

---

# 18. Conformal Prediction Flow

```text
Training Data
     │
     ▼
Train Model
     │
     ▼
Calibration Data
     │
     ▼
Calculate Nonconformity Scores
     │
     ▼
Estimate Calibration Quantile
     │
     ▼
Apply to New Prediction
     │
     ▼
Prediction Interval
```

A prediction can therefore be represented as:

```text
RUL = 35

Interval = [24, 47]
```

The interval width provides information about prediction uncertainty.

---

# 19. Uncertainty Variants

## 19.1 Split Conformal

Provides a global calibration baseline using calibration residuals/nonconformity scores.

---

## 19.2 Adaptive Conformal

Adjusts interval scale based on model uncertainty or predicted scale.

---

## 19.3 EARA-Conformal

Provides regime- and scale-aware conformal calibration.

The goal is to account for differences between operating regimes and prediction difficulty.

---

## 19.4 CQR-Style Calibration

The current implementation uses a scale-based conformalized interval approach.

It should not be interpreted as conventional CQR with independently trained lower and upper quantile prediction heads.

---

## 19.5 Engine-Joint Conformal

Provides engine-level calibration using engine-associated nonconformity information.

---

# 20. Explainability Architecture

The explainability layer analyzes why the model produced a prediction.

```text
                Model
                  │
                  ▼
              Prediction
                  │
       ┌──────────┼───────────┐
       │          │           │
       ▼          ▼           ▼
      SHAP       IG      Temporal
                          Relevance
       │          │           │
       └──────────┼───────────┘
                  ▼
                 ERI
                  │
                  ▼
       Explanation Reliability
```

---

# 21. Integrated Gradients

Integrated Gradients uses gradients of the model output with respect to input features.

The output provides feature-level attribution.

For a time-series model, attribution can be represented as:

```text
Sensor × Time Step
```

This allows analysis of both:

* which sensor contributed,
* when that sensor contributed.

---

# 22. SHAP

SHAP estimates the contribution of features to a prediction relative to a reference/background distribution.

The output can identify features that strongly influence the RUL or HI prediction.

---

# 23. Temporal Relevance

Temporal relevance focuses on the contribution of different positions in the input sequence.

Example:

```text
Older observations ───────── Recent observations

       low              medium          high
       relevance        relevance       relevance
```

This provides an additional temporal perspective that complements feature-level attribution.

---

# 24. Explainability Reliability Index

ERI provides a summary of agreement between different explanation methods.

The conceptual flow is:

```text
Integrated Gradients
        │
        ├──────────┐
        │          │
        ▼          ▼
       SHAP    Temporal Relevance
        │          │
        └────┬─────┘
             ▼
            ERI
```

ERI is intended to identify whether multiple explanation methods provide consistent evidence.

A disagreement between explanation methods can be surfaced as an additional caution signal for downstream decisions.

---

# 25. Decision Engine

The decision engine acts as the bridge between machine-learning outputs and operational recommendations.

Inputs include:

```text
RUL
HI
Prediction Interval
Uncertainty
ERI
Risk
Operational Constraints
```

The decision engine then produces an action.

```text
Prediction
    │
    ▼
Risk Assessment
    │
    ▼
Health State
    │
    ▼
Uncertainty Assessment
    │
    ▼
Explanation Reliability
    │
    ▼
Operational Constraints
    │
    ▼
Decision Policy
    │
    ▼
Maintenance Action
```

---

# 26. Maintenance Actions

The current decision engine supports:

```text
CONTINUE_OPERATION
MONITOR
INSPECT
SCHEDULE_MAINTENANCE
URGENT_MAINTENANCE
```

The decision output may include:

* selected action,
* health state,
* uncertainty level,
* ERI,
* risk score,
* urgency,
* reasons,
* constraint violations,
* human-review requirement.

---

# 27. Operational Constraints

The decision engine can incorporate operational constraints such as:

* maintenance window,
* minimum safe RUL,
* maintenance lead time,
* maximum allowable delay,
* resource availability.

This creates a distinction between:

```text
"What does the model predict?"
```

and:

```text
"What maintenance action is operationally feasible?"
```

The final recommendation is therefore generated by combining model evidence with decision constraints.

---

# 28. Human-in-the-Loop Architecture

The HITL subsystem allows an expert to review selected decisions.

```text
AI Recommendation
        │
        ▼
Review Required?
   │          │
  No         Yes
   │          │
   │          ▼
   │     Expert Review
   │          │
   │     ┌────┴─────┐
   │     ▼          ▼
   │   Approve    Override
   │     │          │
   └─────┴──────────┘
          │
          ▼
     Expert Feedback
          │
          ▼
    Policy Refinement
```

Expert feedback can contain:

* AI recommendation,
* expert action,
* confidence,
* risk,
* urgency,
* health state,
* uncertainty,
* explanation reliability,
* reasoning.

---

# 29. Decision Policy Refinement

The current system uses HITL feedback for **decision-policy refinement**.

The conceptual workflow is:

```text
AI Decision
    │
    ▼
Expert Decision
    │
    ▼
Compare AI vs Expert
    │
    ▼
Record Feedback
    │
    ▼
Aggregate Feedback
    │
    ▼
Conservative Policy Update
```

The current implementation does not use this feedback to retrain the neural-network prognostic model.

---

# 30. Model Monitoring

The current architecture provides decision-level monitoring through:

* RUL prediction,
* HI prediction,
* prediction intervals,
* uncertainty,
* ERI,
* risk,
* human-review flags,
* expert feedback.

A future monitoring system can extend this with:

```text
Model Predictions
       │
       ▼
Performance Monitoring
       │
       ├── Error Drift
       ├── Coverage Drift
       ├── Feature Drift
       └── Explanation Drift
```

---

# 31. Future Retraining Architecture

Automated model retraining is currently outside the implemented HITL workflow.

A future architecture could be:

```text
Monitoring
    │
    ▼
Drift Detection
    │
    ▼
Retraining Trigger
    │
    ▼
Dataset Update
    │
    ▼
Model Retraining
    │
    ▼
Validation
    │
    ├── Performance
    ├── Uncertainty
    ├── XAI
    └── Decision Safety
    │
    ▼
Model Approval
    │
    ▼
Deployment
```

This is a future extension rather than a currently implemented automated pipeline.

---

# 32. Artifact Flow

The system generates several categories of artifacts.

```text
                    Experiment
                        │
        ┌───────────────┼────────────────┐
        │               │                │
        ▼               ▼                ▼
   Checkpoints       Results           Logs
        │               │
        │       ┌───────┼────────┐
        │       │       │        │
        │       ▼       ▼        ▼
        │     Metrics Predictions
        │
        └─────────────────────────┐
                                  ▼
                            XAI Outputs
                                  │
                                  ▼
                           Decision Outputs
                                  │
                                  ▼
                           HITL Feedback
```

Typical output directories include:

```text
outputs/
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

# 33. Module Responsibilities

| Module                 | Responsibility                                       |
| ---------------------- | ---------------------------------------------------- |
| `src/preprocessing/`   | Data loading, preprocessing and window generation    |
| `src/models/`          | Neural network architectures                         |
| `src/uncertainty/`     | Prediction interval and conformal methods            |
| `src/explainability/`  | SHAP, Integrated Gradients and temporal explanations |
| `src/decision_engine/` | Risk assessment and maintenance decisions            |
| `src/hitl/`            | Human review and feedback handling                   |
| `src/evaluation/`      | Evaluation-related functionality                     |
| `src/utils/`           | Shared utilities                                     |
| `scripts/`             | Experiment and pipeline entry points                 |
| `configs/`             | Experiment configuration definitions                 |
| `tests/`               | Automated tests                                      |

---

# 34. End-to-End Execution Sequence

A complete experiment can be represented as:

```text
1. Load C-MAPSS
        │
        ▼
2. Validate Dataset
        │
        ▼
3. Generate RUL Labels
        │
        ▼
4. Select Features
        │
        ▼
5. Scale Features
        │
        ▼
6. Create Temporal Windows
        │
        ▼
7. Split by Engine
        │
        ▼
8. Train Model
        │
        ▼
9. Generate RUL + HI
        │
        ▼
10. Evaluate Predictions
        │
        ▼
11. Calibrate Uncertainty
        │
        ▼
12. Generate XAI
        │
        ▼
13. Calculate ERI
        │
        ▼
14. Run Decision Engine
        │
        ▼
15. Apply Operational Constraints
        │
        ▼
16. Generate Maintenance Action
        │
        ▼
17. Human Review
        │
        ▼
18. Store Expert Feedback
        │
        ▼
19. Refine Decision Policy
```

---

# 35. Reproducibility

Experiments should record:

* dataset subset,
* model architecture,
* random seed,
* preprocessing configuration,
* window size,
* model hyperparameters,
* training checkpoint,
* uncertainty method,
* calibration level,
* decision configuration.

A result should therefore be interpreted together with its complete experiment configuration.

For example:

```text
Dataset: FD001
Model: GRU-Att-Deg
Target: RUL
Uncertainty: EARA-Conformal
Coverage: 90%
Seed: <experiment seed>
Checkpoint: <checkpoint identifier>
```

---

# 36. Implementation Gaps

The following items are important when interpreting the current architecture.

## 36.1 Configuration Loading

Configuration YAML files exist in the repository, but configuration loading is not fully wired into all execution paths.

Some scripts still rely on module-level constants.

---

## 36.2 Automated Retraining

The current HITL implementation does not automatically retrain the underlying model.

Feedback is used for decision-policy refinement.

---

## 36.3 Multi-Seed Validation

Some uncertainty experiments require additional multi-seed and engine-level validation before their results should be treated as robust research conclusions.

---

## 36.4 XAI Faithfulness

Explanation-method agreement through ERI does not by itself prove explanation faithfulness.

Additional faithfulness and sensitivity experiments are required.

---

## 36.5 Evaluation Components

Some evaluation components remain incomplete or are implemented outside the dedicated evaluation package.

---

# 37. Architectural Summary

PHM-XAI is organized around the principle:

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
                     │
                     ▼
       Trustworthy Decision Support
```

The architecture therefore separates the system into four conceptual layers:

```text
┌───────────────────────────────────────┐
│          DATA & PREPROCESSING         │
│       C-MAPSS → Model-ready Data     │
└──────────────────────┬────────────────┘
                       │
                       ▼
┌───────────────────────────────────────┐
│          PROGNOSTIC MODELING          │
│       RUL + HI Prediction             │
└──────────────────────┬────────────────┘
                       │
                       ▼
┌───────────────────────────────────────┐
│       TRUST & INTERPRETABILITY        │
│   Uncertainty + XAI + ERI             │
└──────────────────────┬────────────────┘
                       │
                       ▼
┌───────────────────────────────────────┐
│        DECISION SUPPORT               │
│ Decision Engine + Constraints + HITL  │
└───────────────────────────────────────┘
```

The architecture provides a foundation for extending the research system toward continuous monitoring, automated drift detection, model retraining and production deployment while maintaining separation between **prediction**, **uncertainty**, **explanation**, and **maintenance decision logic**.
