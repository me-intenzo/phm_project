# Uncertainty Estimation

## 1. Overview

The PHM-XAI system does not treat a Remaining Useful Life (RUL) prediction as a single deterministic value.

A neural network may predict:

```text
RUL = 35 cycles
```

However, the actual remaining useful life may differ from this point estimate.

Therefore, the system supplements the prediction with an uncertainty interval:

```text
Predicted RUL = 35 cycles

Prediction Interval:
[24, 47] cycles
```

The interval provides information about the range of plausible RUL values at a specified calibration level.

The uncertainty estimation module is designed to:

* quantify prediction uncertainty,
* generate prediction intervals,
* evaluate empirical coverage,
* measure interval width,
* analyze uncertainty across operating regimes,
* provide uncertainty information to the maintenance decision engine.

The current implementation primarily uses **conformal prediction-based methods**.

---

# 2. Why Uncertainty is Required

RUL prediction is inherently uncertain because:

* sensor measurements can be noisy,
* degradation trajectories can vary between engines,
* operating conditions can change,
* different engines may degrade at different rates,
* the model may encounter conditions that are difficult to predict,
* limited training data may not represent every possible degradation pattern.

A point prediction does not communicate this uncertainty.

For example:

```text
Model A:

RUL = 35
```

and

```text
Model B:

RUL = 35
Interval = [34, 36]
```

both have the same point prediction, but the second output provides additional information about the expected prediction range.

Therefore, PHM-XAI represents the model output as:

```text
Prediction + Prediction Interval
```

rather than only:

```text
Prediction
```

---

# 3. Uncertainty Architecture

The uncertainty layer receives predictions from the trained prognostic model and calibrates them using a separate calibration dataset.

```text
                Trained Model
                     │
                     ▼
              Test Input Window
                     │
                     ▼
                Point RUL
                Prediction
                     │
                     ▼
            Calibration Module
                     │
                     ▼
          Nonconformity Score
                     │
                     ▼
        Calibration Quantile
                     │
                     ▼
          Prediction Interval
                     │
                     ▼
       ┌─────────────────────────┐
       │ Lower Bound             │
       │ Point Prediction        │
       │ Upper Bound             │
       └────────────┬────────────┘
                    │
                    ▼
             Decision Engine
```

The uncertainty layer is logically separated from the neural network.

This allows different calibration techniques to be evaluated without changing the underlying prognostic model.

---

# 4. Current Uncertainty Methods

The repository contains implementations for several conformal uncertainty approaches:

```text
src/uncertainty/
```

The supported variants include:

1. Split Conformal
2. Adaptive Conformal
3. EARA-Conformal
4. CQR-style scale-based calibration
5. Engine-Joint Conformal

These methods provide different approaches to constructing prediction intervals.

---

# 5. Conformal Prediction

## 5.1 Basic Concept

Conformal prediction uses a calibration dataset to estimate how large the model's prediction errors can be.

The model is first trained using training data.

A separate calibration set is then used to calculate nonconformity scores.

Conceptually:

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
Calculate Prediction Errors
     │
     ▼
Nonconformity Scores
     │
     ▼
Calibration Quantile
     │
     ▼
Prediction Interval
```

The calibration set is separate from the final test set.

---

# 6. Split Conformal Prediction

Split conformal prediction is used as a baseline uncertainty method.

For a regression problem, the nonconformity score can be defined as the absolute prediction error:

```text
s_i = |y_i - ŷ_i|
```

where:

* `y_i` = observed target,
* `ŷ_i` = model prediction,
* `s_i` = nonconformity score.

The calibration scores are then used to obtain an appropriate quantile.

For a new prediction:

```text
ŷ = predicted RUL
q = calibrated quantile
```

the prediction interval can be represented as:

```text
[ŷ - q, ŷ + q]
```

For example:

```text
Prediction = 35
Calibration quantile = 11

Interval = [24, 46]
```

The exact interval values depend on the calibration data and selected coverage level.

---

# 7. Prediction Coverage

A central objective of conformal prediction is **coverage**.

If the target coverage is 90%, the desired behavior is approximately:

```text
90% of future observations
        ↓
should fall inside
        ↓
the generated prediction intervals
```

Empirical coverage is calculated as:

```text
Coverage =
Number of targets inside interval
---------------------------------
Total number of evaluated targets
```

For example:

```text
100 test observations

92 contained within their intervals

Empirical Coverage = 92%
```

The observed coverage may differ from the nominal coverage in a finite sample.

Therefore, both nominal and empirical coverage should be reported.

---

# 8. Coverage Levels

The uncertainty evaluation uses multiple nominal coverage levels.

Current target levels include:

```text
80%
90%
95%
```

These correspond approximately to:

```text
80% → narrower interval
90% → wider interval
95% → widest interval
```

Increasing the requested coverage generally increases the prediction interval width.

---

# 9. Prediction Interval Width

Coverage alone is not sufficient.

An interval could achieve high coverage simply by becoming excessively wide.

Therefore, interval width is also measured.

For an interval:

```text
[L, U]
```

the width is:

```text
Width = U - L
```

For example:

```text
Interval = [20, 50]

Width = 50 - 20
      = 30 cycles
```

The system evaluates interval width together with coverage.

---

# 10. Mean Prediction Interval Width

The **Mean Prediction Interval Width (MPIW)** is the average width across evaluated samples.

```text
MPIW = mean(U_i - L_i)
```

where:

* `U_i` = upper interval bound,
* `L_i` = lower interval bound.

A useful uncertainty system should provide adequate coverage without producing unnecessarily wide intervals.

Therefore, uncertainty evaluation considers both:

```text
Coverage
+
Interval Width
```

---

# 11. Normalized Prediction Interval Width

Raw interval width can be difficult to compare across datasets or prediction ranges.

The system therefore also supports normalized interval-width analysis.

A normalized width can be expressed conceptually as:

```text
Normalized Width =
Interval Width
----------------
Reference Scale
```

The exact normalization depends on the evaluation implementation.

This metric helps compare interval sharpness across experiments.

---

# 12. Adaptive Conformal Prediction

A global conformal interval may not be equally informative for every prediction.

For example:

```text
Easy Prediction
      │
      ▼
Small Expected Error
      │
      ▼
Narrow Interval
```

while:

```text
Difficult Prediction
      │
      ▼
Large Expected Error
      │
      ▼
Wider Interval
```

Adaptive conformal approaches attempt to account for varying prediction difficulty.

The current implementation includes scale-adaptive conformal calibration.

Conceptually:

```text
Model Prediction
      │
      ▼
Estimated Prediction Difficulty / Scale
      │
      ▼
Adaptive Nonconformity
      │
      ▼
Calibration
      │
      ▼
Adaptive Prediction Interval
```

---

# 13. EARA-Conformal

The repository includes an **EARA-Conformal** implementation intended to provide more adaptive uncertainty estimation.

The approach incorporates:

* operating-regime information,
* prediction scale,
* calibration errors.

The conceptual workflow is:

```text
Calibration Data
      │
      ▼
Identify / Assign Operating Regimes
      │
      ▼
Estimate Prediction Error
      │
      ▼
Account for Prediction Scale
      │
      ▼
Regime / Scale-Aware Calibration
      │
      ▼
Prediction Interval
```

This is useful for C-MAPSS because the dataset contains different operating conditions and degradation trajectories.

---

# 14. Operating Regimes

Operating regimes represent different regions of operating behavior.

The uncertainty system can evaluate interval performance across regimes rather than reporting only a single aggregate value.

Conceptually:

```text
Engine Observations
       │
       ▼
Operating Regime
       │
 ┌─────┼─────┐
 ▼     ▼     ▼
Regime 1  Regime 2  Regime 3
 │          │          │
 ▼          ▼          ▼
Coverage   Coverage   Coverage
```

This enables analysis of whether uncertainty calibration behaves consistently under different operating conditions.

---

# 15. Engine-Joint Conformal

C-MAPSS contains multiple temporal observations from individual engines.

Randomly treating every window as completely independent can hide engine-level variation.

The repository therefore includes an engine-joint conformal approach.

Conceptually:

```text
Engine 1 ──┐
Engine 2 ──┤
Engine 3 ──┤
Engine 4 ──┤
           ▼
   Engine-Level Scores
           │
           ▼
     Calibration
           │
           ▼
 Engine-Aware Intervals
```

The objective is to evaluate uncertainty at a level that respects the grouping of observations by engine.

---

# 16. CQR-Style Calibration

The repository includes a `cqr` uncertainty option.

The current implementation should be described as a:

> **scale-based conformalized interval approach**

rather than conventional trained **Conformalized Quantile Regression (CQR)** with independently trained lower and upper quantile heads.

The conceptual flow is:

```text
Point Prediction
      │
      ▼
Estimated Scale
      │
      ▼
Conformal Calibration
      │
      ▼
Prediction Interval
```

This distinction is important for accurate academic documentation.

---

# 17. Calibration and Test Isolation

The calibration dataset must remain separate from the final test dataset.

The intended workflow is:

```text
                 Dataset
                    │
          ┌─────────┼─────────┐
          │         │         │
          ▼         ▼         ▼
       Training  Validation Calibration
                              │
                              ▼
                     Calibrate Interval
                              │
                              ▼
                           Test
                              │
                              ▼
                     Final Evaluation
```

The test targets should not be used to determine calibration parameters.

This separation reduces the risk of optimistic uncertainty estimates.

---

# 18. Engine-Disjoint Calibration

Where applicable, calibration should respect engine identity.

For example:

```text
Training Engines
     │
     ├── Engine 1
     ├── Engine 2
     └── Engine 3

Calibration Engines
     │
     ├── Engine 4
     ├── Engine 5
     └── Engine 6

Test Engines
     │
     ├── Engine 7
     ├── Engine 8
     └── Engine 9
```

Windows from the same engine should not be arbitrarily distributed across these groups.

This is important because adjacent windows from the same engine can be highly correlated.

---

# 19. Uncertainty Output

The uncertainty module produces interval information associated with model predictions.

A conceptual output record is:

```text
{
    prediction: 35,
    lower_bound: 24,
    upper_bound: 47,
    coverage_level: 0.90
}
```

The actual output structure depends on the implementation used by the experiment.

These outputs can be stored under:

```text
outputs/results/uncertainty/
```

---

# 20. Uncertainty Metrics

The uncertainty evaluation considers multiple metrics.

## 20.1 Empirical Coverage

Measures the proportion of observed targets contained inside the prediction intervals.

```text
Coverage =
Count(y_i ∈ [L_i, U_i])
-------------------------
N
```

---

## 20.2 Coverage Error

Measures the difference between nominal and empirical coverage.

```text
Coverage Error =
|Empirical Coverage - Nominal Coverage|
```

For example:

```text
Nominal = 90%
Empirical = 87%

Coverage Error = 3 percentage points
```

---

## 20.3 MPIW

Mean Prediction Interval Width measures the average interval size.

```text
MPIW = mean(U_i - L_i)
```

---

## 20.4 Normalized Width

Normalized width allows interval size to be compared across different scales.

---

## 20.5 Per-Regime Coverage

Coverage can be reported separately for different operating regimes.

This helps identify cases where aggregate coverage appears acceptable while specific regimes are poorly calibrated.

---

## 20.6 Engine-Level Coverage

Engine-level analysis evaluates whether intervals provide adequate coverage across complete engine trajectories rather than only individual windows.

---

# 21. Coverage vs Sharpness

Uncertainty quality involves a trade-off between:

```text
Coverage
    ↕
Reliability
```

and:

```text
Interval Width
    ↕
Sharpness
```

Conceptually:

```text
Very Narrow Interval
        │
        ▼
     Low Width
        │
        └──────► May have insufficient coverage


Very Wide Interval
        │
        ▼
    High Coverage
        │
        └──────► May provide limited practical information
```

The goal is therefore not simply to maximize coverage.

The objective is to obtain appropriate coverage with reasonably sharp intervals.

---

# 22. Uncertainty and Maintenance Decisions

Uncertainty is passed to the maintenance decision engine.

The decision system does not rely exclusively on the point RUL prediction.

Instead:

```text
RUL
 │
 ├───────────────┐
 ▼               ▼
Point Value    Interval
 │               │
 └───────┬───────┘
         ▼
   Risk Assessment
         │
         ▼
 Maintenance Decision
```

For example:

```text
RUL = 40 cycles
Interval = [37, 43]
```

may represent a relatively concentrated prediction.

In contrast:

```text
RUL = 40 cycles
Interval = [15, 70]
```

represents a much wider range of possible outcomes.

The decision engine can use this difference when determining whether additional monitoring or human review is appropriate.

---

# 23. Uncertainty-Based Decision Logic

The conceptual decision process is:

```text
                 RUL Prediction
                      │
                      ▼
              Prediction Interval
                      │
                      ▼
             Estimate Uncertainty
                      │
             ┌────────┴─────────┐
             ▼                  ▼
       Lower Uncertainty    Higher Uncertainty
             │                  │
             ▼                  ▼
       Normal Decision      Additional Review
                                  │
                                  ▼
                            Human-in-the-Loop
```

The exact thresholds and decision rules are defined by the decision-engine configuration and implementation.

They should not be inferred from uncertainty metrics alone.

---

# 24. Relationship with XAI

Uncertainty and explainability are complementary.

Uncertainty answers:

> **How certain is the prediction?**

XAI answers:

> **What input evidence contributed to the prediction?**

The combined architecture is:

```text
                 Model
                   │
          ┌────────┴────────┐
          ▼                 ▼
     Prediction        Prediction
     Value             Uncertainty
          │                 │
          │                 │
          ▼                 ▼
        XAI             Interval
          │                 │
          └────────┬────────┘
                   ▼
              Decision Engine
```

The PHM-XAI system additionally calculates ERI to assess agreement among explanation methods.

Therefore:

```text
Prediction
+
Uncertainty
+
Explanation
+
Explanation Reliability
```

can be considered together during decision support.

---

# 25. Uncertainty Workflow

The complete uncertainty workflow is:

```text
1. Train Prognostic Model
          │
          ▼
2. Generate Calibration Predictions
          │
          ▼
3. Calculate Nonconformity Scores
          │
          ▼
4. Estimate Calibration Quantile
          │
          ▼
5. Generate Test Predictions
          │
          ▼
6. Construct Prediction Intervals
          │
          ▼
7. Calculate Empirical Coverage
          │
          ▼
8. Calculate Interval Width
          │
          ▼
9. Evaluate Per-Regime Performance
          │
          ▼
10. Evaluate Engine-Level Behavior
          │
          ▼
11. Pass Uncertainty to Decision Engine
```

---

# 26. Example

Consider an engine for which the model predicts:

```text
RUL = 32 cycles
```

After calibration, the system produces:

```text
90% Prediction Interval

Lower Bound = 21
Prediction  = 32
Upper Bound = 46
```

The output can be represented as:

```text
RUL = 32
PI90 = [21, 46]
```

The interval width is:

```text
46 - 21 = 25 cycles
```

The decision engine can then combine this information with:

* HI,
* ERI,
* risk,
* urgency,
* operational constraints.

The uncertainty estimate therefore becomes part of the downstream decision-support process.

---

# 27. Command-Line Usage

The uncertainty pipeline can be executed using:

```bash
python scripts/uncertainty.py \
    --model gru \
    --subset FD001 \
    --n_regimes 7
```

The exact command-line options may vary depending on the current implementation.

The uncertainty module supports experiments across different:

* models,
* C-MAPSS subsets,
* regime configurations,
* calibration methods,
* nominal coverage levels.

---

# 28. Output Artifacts

Uncertainty-related outputs are organized under:

```text
outputs/
└── results/
    └── uncertainty/
```

Typical artifacts can include:

```text
Prediction intervals
Coverage metrics
Interval-width metrics
Regime-level results
Engine-level results
Evaluation reports
```

The exact filenames depend on the experiment configuration.

---

# 29. Experimental Comparison

Uncertainty methods should be compared using a common evaluation protocol.

A recommended experiment table is:

| Method             | Nominal Coverage | Empirical Coverage | Coverage Error | MPIW | Normalized Width |
| ------------------ | ---------------: | -----------------: | -------------: | ---: | ---------------: |
| Split Conformal    |              80% |                  — |              — |    — |                — |
| Split Conformal    |              90% |                  — |              — |    — |                — |
| Split Conformal    |              95% |                  — |              — |    — |                — |
| Adaptive Conformal |              90% |                  — |              — |    — |                — |
| EARA-Conformal     |              90% |                  — |              — |    — |                — |
| Engine-Joint       |              90% |                  — |              — |    — |                — |

The values should be populated from actual experimental runs.

No method should be declared superior solely because it has a narrower interval; coverage must also be considered.

---

# 30. Current Implementation Status

| Component                         | Status      |
| --------------------------------- | ----------- |
| Split Conformal                   | Implemented |
| Adaptive Conformal                | Implemented |
| EARA-Conformal                    | Implemented |
| CQR-style scale-based calibration | Implemented |
| Engine-Joint Conformal            | Implemented |
| 80% coverage evaluation           | Supported   |
| 90% coverage evaluation           | Supported   |
| 95% coverage evaluation           | Supported   |
| Empirical coverage                | Implemented |
| Coverage error                    | Implemented |
| MPIW                              | Implemented |
| Normalized interval width         | Implemented |
| Per-regime evaluation             | Implemented |
| Engine-level analysis             | Implemented |
| Integration with decision engine  | Implemented |
| Multi-seed robustness analysis    | Pending     |
| Full uncertainty ablation study   | Pending     |

---

# 31. Important Implementation Caveats

## 31.1 Existing Checkpoints

Previously trained model checkpoints may not contain the scale-related outputs required by some adaptive uncertainty methods.

Therefore, adaptive uncertainty experiments should use checkpoints trained with the current uncertainty-aware model/loss configuration where required.

---

## 31.2 Configuration Files

The repository contains YAML configuration files, but configuration loading is not fully connected to every execution path.

Some uncertainty experiments may therefore depend on parameters defined directly in the relevant Python modules.

---

## 31.3 Finite-Sample Variation

Empirical coverage does not necessarily equal nominal coverage exactly.

For example:

```text
Nominal Coverage = 90%
Observed Coverage = 88%
```

does not automatically indicate that the method is invalid.

Coverage should be interpreted using:

* sample size,
* calibration size,
* engine-level grouping,
* experimental repetition.

---

## 31.4 Regime Calibration

Regime-specific calibration can reduce the number of samples available for individual regimes.

Very small regimes may produce unstable estimates.

Therefore, regime-level results should be interpreted together with the number of calibration samples per regime.

---

# 32. Limitations

The current uncertainty framework has several limitations.

### Dataset Limitation

C-MAPSS is a simulated turbofan dataset and may not represent the complete variability of real industrial systems.

### Calibration Distribution

Conformal guarantees depend on assumptions concerning the relationship between calibration and future data.

Distribution shifts may reduce practical coverage.

### Operating-Regime Generalization

Regime-aware methods depend on how operating regimes are defined and represented.

### Sample Size

Small calibration groups can produce less stable estimates.

### Model Dependence

Prediction intervals are still based on the underlying model's predictions and errors.

Uncertainty calibration does not automatically correct a fundamentally unsuitable predictive model.

### No Automated Drift Response

The current system does not automatically detect production drift and retrain the model.

---

# 33. Future Work

Future uncertainty-related research can include:

1. Multi-seed uncertainty experiments.
2. Bootstrap-based confidence analysis.
3. Deep ensembles.
4. Monte Carlo Dropout.
5. Comparison of conformal and probabilistic uncertainty.
6. Conditional coverage analysis.
7. More detailed engine-level calibration.
8. Drift-aware uncertainty calibration.
9. Distribution-shift experiments.
10. Uncertainty-aware retraining triggers.
11. Calibration diagnostics and reliability curves.
12. Statistical significance testing across repeated experiments.

A future uncertainty architecture could be:

```text
Live Sensor Data
       │
       ▼
Prognostic Model
       │
       ▼
Prediction
       │
 ┌─────┴─────────────┐
 ▼                   ▼
Aleatoric          Epistemic
Uncertainty        Uncertainty
 │                   │
 └─────────┬─────────┘
           ▼
     Calibration
           │
           ▼
 Prediction Interval
           │
           ▼
 Decision Engine
```

This would extend the current conformal approach toward a broader uncertainty framework.

---

# 34. Summary

The uncertainty module transforms a deterministic prognostic prediction into a prediction accompanied by an uncertainty interval.

The core workflow is:

```text
C-MAPSS
   │
   ▼
Preprocessing
   │
   ▼
Deep Learning Model
   │
   ▼
RUL / HI Prediction
   │
   ▼
Calibration Data
   │
   ▼
Conformal Calibration
   │
   ▼
Prediction Interval
   │
   ├───────────────┐
   ▼               ▼
Coverage        Interval Width
   │               │
   └───────┬───────┘
           ▼
      Uncertainty
           │
           ▼
    Decision Engine
           │
           ▼
 Maintenance Decision
```

The purpose of uncertainty estimation in PHM-XAI is therefore not simply to generate wider or narrower intervals. It is to provide **calibrated information about prediction reliability** that can be combined with RUL, HI, XAI and operational constraints for maintenance decision support.
