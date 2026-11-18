# Results

Published results for PHM-XAI on the NASA C-MAPSS turbofan dataset, plus a
comparison against published state-of-the-art (SOTA) RUL baselines.

> **Read this first.** Every number below comes from the scripts in this
> repository and is reproduced by the command noted in each section. The
> headline point-prediction and interval numbers are **mean ± sample std over
> five independent seeds (42–46)** of the deliverable model
> `gru_att_deg_v2`, each trained from scratch and evaluated from its own
> checkpoint. They are *not* a leaderboard claim: a published number is
> single-run by construction, so differences smaller than the seed band are
> not evidence of a ranking (see [Caveats](#7-caveats)). Re-run before citing.

---

## Contents

- [0. System Architecture](#0-system-architecture)
- [1. Point prediction - published baselines](#1-point-prediction--published-baselines)
- [2. Point prediction - this repository](#2-point-prediction--this-repository)
- [2.1 The health-index task](#21-the-health-index-task)
- [3. Conformal interval benchmark (multi-seed)](#3-conformal-interval-benchmark-multi-seed)
- [4. Explainability](#4-explainability)
- [4.1 The downstream chain on the same checkpoints](#41-the-downstream-chain-on-the-same-checkpoints)
- [4.2 The decision policy](#42-the-decision-policy)
- [4.3 Sensitivity of the decision to its own weights](#43-sensitivity-of-the-decision-to-its-own-weights)
- [4.4 Operational-constraint scenarios](#44-operational-constraint-scenarios)
- [5. Key findings](#5-key-findings)
- [6. How to reproduce](#6-how-to-reproduce)
- [7. Caveats](#7-caveats)

---

## 0. System Architecture

A step-by-step walkthrough of what happens from raw sensor data to an auditable maintenance recommendation.

```
Step 1  Raw data
        NASA C-MAPSS text files (train/test/RUL for FD001–FD004).
        Each row is one engine-cycle with 21 sensor readings and
        3 operational settings.

Step 2  Preprocessing  [scripts/preprocess.py]
        a. Validate schema, check for gaps and out-of-range values.
        b. Regress each sensor on the 3 operational settings; keep
           the residual so that operating-condition variance does not
           dominate the degradation signal (critical for FD002/FD004).
        c. Build the piecewise-linear RUL label (capped at max_rul=125).
        d. Build the sensor-derived Health Index (HI):
             - per-engine baseline from first 30 cycles,
             - one-component PLS direction fitted against training RUL,
             - projection mapped onto [0,1] with training quantiles.
        e. Select the 14 informative sensors (variance/MI filter).
        f. Z-score normalise per engine; persist the scaler.
        g. Slide a 40-cycle window across every engine; write .npy arrays
           and a metadata sidecar with hi_thresholds for the decision layer.
        Invariant: train/validation and calibration/test splits are
        grouped by engine ID — no engine appears in two splits.

Step 3  Multi-task model training  [scripts/train.py]
        Architecture: GRU-Att-Deg-v2 (GRU trunk + degradation-aware
        attention + three output heads).
        Each forward pass returns (pred_rul, pred_hi, pred_scale).
        Loss = RUL_MSE + 0.4 * HI_MSE * max_rul^2 + 0.1 * scale_loss.
        The HI term is scaled by max_rul^2 so it is numerically on the
        same order as the RUL term.
        Training: AdamW, cosine LR with 5-epoch warm-up, gradient clip
        1.0, weight EMA decay 0.995, early stopping patience 30.
        The checkpoint stores model_config, seed, and a SHA-256 stamp.

Step 4  Point-prediction evaluation  [scripts/evaluate.py]
        Rebuilds the architecture from the checkpoint's own model_config
        (never from the YAML), runs inference on the test set, and writes
        predictions + provenance stamp to .npz.
        Metrics: RUL MAE / RMSE / R² / NASA Score; HI MAE / RMSE / R².

Step 5  Conformal uncertainty intervals  [scripts/uncertainty.py]
        Engine-disjoint calibration split (carved from training engines).
        Four methods available (--method flag):
          split          global ŷ ± q̂
          cqr            conformalized quantile regression
          adaptive       ŷ ± q̂·s(x), scale from training residuals
          engine_joint   EARA-Conformal: scale from k-NN engine profiles
                         conditioned on degradation regime; global q̂
        Coverage evaluated at 80/90/95% nominal.
        Invariant: calibration quantiles are estimated on calibration
        engines only; the test set is used solely to measure coverage.

Step 6  Explainability  [scripts/explain.py]
        Three attribution methods run on the same eval-guarded forward pass:
          Integrated Gradients (IG)  — path-integrated sensor attributions
          SHAP                       — kernel SHAP sensor importances
          Temporal relevance         — per-timestep gradient relevance
        The Explainability Reliability Index (ERI) measures cross-method
        agreement:
          ERI = 0.4·(ρ(IG,Attn)+1)/2 + 0.4·(ρ(IG,SHAP)+1)/2 + 0.2·top-K consensus
        ERI ∈ [0,1]; an uncorrelated pair contributes 0.5 to its term.
        Outputs: attribution arrays, eri_rul.json, eri_hi.json, HTML report.

Step 7  Decision engine  [scripts/decision.py]
        Pure function: state dict → action. No torch, no numpy arrays.
        a. Classify health state from RUL and HI using calibrated thresholds:
             HEALTHY / DEGRADING / AT_RISK / CRITICAL
        b. Compute risk score R = 0.40·r_rul + 0.30·r_hi + 0.15·r_u + 0.15·r_eri
        c. Determine planning_state from the LOWER bound of the 90% interval
           (the worst credible case, not the point estimate).
        d. Map planning_state to a base action on the severity ladder:
             CONTINUE_OPERATION < MONITOR < INSPECT <
             SCHEDULE_MAINTENANCE < URGENT_MAINTENANCE
        e. Escalate by +1 step if ERI is LOW (< 0.60).
        f. Override to URGENT_MAINTENANCE if risk score > 0.85.
        g. Apply operational constraints (maintenance window, lead time,
           minimum safe RUL, resource availability) — constraints can only
           escalate, never de-escalate, except the window substitution.
        h. Flag for human review if uncertainty is HIGH or ERI is LOW.
        Outputs: decision JSON + HTML report per engine.

Step 8  Human-in-the-loop review  [scripts/hitl.py]
        Consumes the decision JSON; creates review items for records
        flagged human_review=true.
        Expert feedback is appended to an append-only feedback.jsonl log.
        Conservative policy refinement: an alternative action is adopted
        only with ≥3 consistent observations, weighted support ≥2.0,
        consensus ≥0.70. URGENT_MAINTENANCE is never automatically weakened.
        HITL refines the decision policy only — the model is never retrained.
```

**Three invariants hold at every stage:**

1. Engine-disjoint splits — windows from one engine are correlated; no engine appears in two splits.
2. Calibration never sees test targets — conformal quantiles are estimated on calibration engines only.
3. Monotone policy — more uncertainty or a less reliable explanation can never produce a less severe action.

---

## 1. Point prediction — published baselines

Representative, widely-cited C-MAPSS RUL baselines. RMSE and the NASA
asymmetric Score are the two standard metrics; lower is better for both.

| Model | Reference | FD001 RMSE | FD001 Score | FD002 RMSE | FD003 RMSE | FD004 RMSE | FD004 Score |
|---|---|---|---|---|---|---|---|
| CNN | Li et al. 2018 | 12.61 | 273.7 | 22.36 | 12.64 | 28.17 | 12466 |
| LSTM | Zheng et al. 2017 | 16.14 | 338 | 24.49 | 16.18 | 28.17 | 5550 |
| LSTM-F (FSGRI) | Fu et al. 2024 | 10.98 | — | 14.60 | 9.38 | 15.00 | — |
| CNN-GRU | Fu et al. 2024 | 11.66 | — | 14.93 | 9.85 | 14.86 | — |
| BTSAM | Fu et al. 2024 | 10.62 | — | 14.18 | 9.25 | 14.29 | — |
| Dual-Mixer (SOTA) | Fu et al. 2024 | 10.41 | — | 13.90 | 8.87 | 12.15 | — |
| DRLSTM-DA | Wu et al. 2024 | 11.55 | 234.2 | 13.74 | 11.25 | 14.19 | 537.7 |
| 1D-CNN-LSTM | Bektas & Kara 2023 | 16.10 | 437.2 | — | — | — | — |
| KESTF | Applied Sciences 2026 | — | — | — | — | — | — |
| TTSNet | Sensors 2025 | — | — | — | — | — | — |

**Sources**

- Li, Ding & Sun (2018), *Remaining useful life estimation in prognostics using deep convolution neural networks*, RESS.
- Zheng et al. (2017), *Long short-term memory network for RUL estimation*.
- Fu et al. (2024), *Supervised Contrastive Learning based Dual-Mixer Model for RUL Prediction*, arXiv:2401.16462.
- Wu et al. (2024), *DRLSTM-DA: a novel aeroengine remaining useful life prediction method*, Measurement Science and Technology.
- Bektas & Kara (2023), *RUL Estimation of Turbofan Engines with Change-Point Detection Based Labeling*, Applied Sciences 13(21):11893.

> **Score metric warning.** NASA Score values are extremely sensitive to a handful of late predictions and are not comparable across papers unless the same RUL clipping and test-endpoint definition are used. Treat cross-paper Score columns as indicative, not exact.

---

## 2. Point prediction — this repository

**Certified headline.** Five independent seeds (42–46) of `gru_att_deg_v2`,
trained from scratch with deterministic kernels, each evaluated on the full
test set. Values are mean ± sample std **across seeds** (n = 5).
Source: `results/multiseed/summary_mean_std.csv`.

| Subset | RUL MAE | RUL RMSE | R² | NASA Score | RMSE range (5 seeds) |
|---|---|---|---|---|---|
| FD001 | 9.17 ± 0.23 | **12.63 ± 0.23** | 0.901 | 259 ± 22 | 12.36 – 12.98 |
| FD002 | 10.17 ± 0.16 | **13.86 ± 0.17** | 0.896 | 927 ± 54 | 13.62 – 14.04 |
| FD003 | 8.11 ± 0.29 | **11.61 ± 0.29** | 0.912 | 235 ± 26 | 11.11 – 11.77 |
| FD004 | 10.95 ± 0.18 | **15.02 ± 0.22** | 0.878 | 1035 ± 50 | 14.78 – 15.35 |
| **FD001–FD004 mean** | **9.60** | **13.282 ± 0.140** | **0.896** | **614** | 13.21 – 13.53 |

Per-seed four-subset aggregates: 13.533 / 13.209 / 13.214 / 13.225 / 13.230 (seeds 42–46).

**Sensor-derived HI vs legacy label — effect on RUL RMSE:**

| | legacy `HI = RUL / max_rul` | sensor-derived HI | Δ |
|---|---|---|---|
| FD001 | 12.383 ± 0.275 | 12.630 ± 0.231 | +0.247 |
| FD002 | 13.923 ± 0.433 | 13.865 ± 0.173 | −0.058 |
| FD003 | 11.974 ± 0.261 | 11.611 ± 0.285 | −0.363 |
| FD004 | 15.166 ± 0.168 | 15.023 ± 0.221 | −0.143 |
| **Four-subset mean** | **13.361 ± 0.136** | **13.282 ± 0.140** | **−0.079** |

The aggregate is 0.079 RMSE better than the legacy run — well inside the seed band (±0.14) and therefore not a claimed improvement — but it confirms that making the health index a real task did **not** cost RUL accuracy.

**Where PHM-XAI sits** (mean vs published single runs):

- **FD001 12.63** — marginally ahead of the classic CNN (12.61); behind every 2024 model.
- **FD002 13.86** — ahead of CNN-GRU (14.93), BTSAM (14.18), LSTM-F (14.60); within seed std of DRLSTM-DA (13.74) and Dual-Mixer (13.90).
- **FD003 11.61** — behind DRLSTM-DA (11.25) and the 9.85/8.87 band.
- **FD004 15.02** — behind DRLSTM-DA (14.19), BTSAM (14.29), CNN-GRU (14.86), LSTM-F (15.00).
- **Four-subset mean 13.282** — 0.452 behind KESTF (12.83), which exceeds the seed std (0.140) and is a real deficit. Statistically indistinguishable from TTSNet (13.40); ahead of Cau-BiMamba-LSTM (13.62), MLEAN (13.71), DAST (14.09), ATCN (14.11), DBA-DCRN (14.28), CNN-LSTM-Att (15.25).

This repository's contribution is **not** to beat those RMSE numbers. The value-add is that every point prediction is wrapped in calibrated, engine-disjoint uncertainty intervals and auditable maintenance decisions — see Sections 3–4.

---

## 2.1 The health-index task

The health index is the model's second output. It is built from the **sensor signal**, not from the RUL label. `corr(HI, RUL)` on the training split is 0.890 (FD001), 0.674 (FD002), 0.797 (FD003) and 0.705 (FD004) — not 1.000000.

**Certified health-index metrics** (five seeds, same checkpoints as §2):

| Subset | HI RMSE | HI MAE | HI R² |
|---|---|---|---|
| FD001 | 0.0756 ± 0.0014 | 0.0577 ± 0.0012 | 0.900 ± 0.004 |
| FD002 | 0.0793 ± 0.0012 | 0.0650 ± 0.0013 | 0.869 ± 0.004 |
| FD003 | 0.0654 ± 0.0013 | 0.0513 ± 0.0013 | 0.931 ± 0.003 |
| FD004 | 0.0815 ± 0.0011 | 0.0650 ± 0.0013 | 0.863 ± 0.004 |

The sensor-derived index is easier to predict than the legacy label because it is a function of the window itself, not of a per-engine normalisation the model cannot see.

---

## 3. Conformal interval benchmark (multi-seed)

Five interval methods × four subsets × five seeds (42–46), engine-disjoint
calibration, nominal 80/90/95%. Every number is computed from the **real
checkpoint** of that (subset, seed). Source: `results/conformal_benchmark/summary_mean_std.csv`.

```powershell
python scripts/multiseed_conformal_benchmark.py --subsets all --seeds 42 43 44 45 46
```

Two calibration units: **`window`** (primary — one score per calibration window, tracks nominal) and **`engine`** (conservative — one max-residual per calibration engine, over-covers by design).

### 3.1 Coverage at the 90% nominal level (window unit)

Mean empirical coverage ± std across seeds; width in cycles (narrower is better at a given coverage).

| Subset | Method | Emp. coverage | Std | Mean width |
|---|---|---|---|---|
| FD001 | split_conformal | 0.878 | 0.033 | 35.4 |
| FD001 | cqr | 0.894 | 0.034 | 32.2 |
| FD001 | adaptive | 0.902 | 0.040 | 31.8 |
| FD001 | nex_scp | 0.892 | 0.023 | **31.0** |
| FD001 | eara_conformal | 0.890 | 0.034 | 32.4 |
| FD002 | split_conformal | 0.906 | 0.006 | 43.1 |
| FD002 | cqr | 0.921 | 0.015 | 39.2 |
| FD002 | adaptive | 0.900 | 0.013 | 37.4 |
| FD002 | nex_scp | 0.896 | 0.004 | **37.7** |
| FD002 | eara_conformal | 0.912 | 0.012 | 39.3 |
| FD003 | split_conformal | 0.898 | 0.008 | 34.3 |
| FD003 | cqr | 0.902 | 0.013 | 30.5 |
| FD003 | adaptive | 0.876 | 0.026 | 28.4 |
| FD003 | nex_scp | 0.884 | 0.013 | 30.0 |
| FD003 | eara_conformal | 0.852 | 0.040 | **28.4** |
| FD004 | split_conformal | 0.904 | 0.026 | 44.5 |
| FD004 | cqr | 0.915 | 0.018 | 42.3 |
| FD004 | adaptive | 0.892 | 0.018 | **37.5** |
| FD004 | nex_scp | 0.896 | 0.003 | 39.9 |
| FD004 | eara_conformal | 0.908 | 0.030 | 42.4 |

**Key findings from the benchmark:**

- Every method lands within ±5 points of the 90% target — the intervals are calibrated in practice.
- `nex_scp` has the most stable coverage (std 0.003–0.023) and the best coverage/width trade-off overall.
- `split_conformal` is the widest on every subset (34.3–44.5 cycles) but the most consistently calibrated.
- `eara_conformal` under-covers on FD003 (0.852, −4.8 pts) while being the narrowest; its k-NN scale under-states uncertainty on last-window test queries.
- `adaptive` under-covers on FD003 (0.876) and FD004 (0.892).
- `cqr` over-covers on FD002 (0.921) but is competitive elsewhere.
- No method ranking is statistically established at five seeds (smallest attainable p = 0.0625 with exact sign-flip test).

### 3.2 Coverage across all nominal levels (window unit)

| Subset | Method | 80% | 90% | 95% |
|---|---|---|---|---|
| FD001 | split_conformal | 0.778 | 0.878 | 0.936 |
| FD001 | cqr | 0.792 | 0.894 | 0.938 |
| FD001 | adaptive | 0.784 | 0.902 | 0.942 |
| FD001 | nex_scp | 0.790 | 0.892 | 0.940 |
| FD001 | eara_conformal | 0.788 | 0.890 | 0.934 |
| FD002 | split_conformal | 0.832 | 0.906 | 0.968 |
| FD002 | cqr | 0.835 | 0.921 | 0.963 |
| FD002 | adaptive | 0.809 | 0.900 | 0.941 |
| FD002 | nex_scp | 0.798 | 0.896 | 0.946 |
| FD002 | eara_conformal | 0.815 | 0.912 | 0.949 |
| FD003 | split_conformal | 0.800 | 0.898 | 0.942 |
| FD003 | cqr | 0.798 | 0.902 | 0.936 |
| FD003 | adaptive | 0.768 | 0.876 | 0.916 |
| FD003 | nex_scp | 0.784 | 0.884 | 0.932 |
| FD003 | eara_conformal | 0.768 | 0.852 | 0.916 |
| FD004 | split_conformal | 0.814 | 0.904 | 0.954 |
| FD004 | cqr | 0.812 | 0.915 | 0.954 |
| FD004 | adaptive | 0.792 | 0.892 | 0.944 |
| FD004 | nex_scp | 0.806 | 0.896 | 0.944 |
| FD004 | eara_conformal | 0.814 | 0.908 | 0.956 |

`adaptive` and `eara_conformal` systematically under-cover at 95% on FD003 (both 0.916). `nex_scp` and `split_conformal` are the two closest to nominal at every level on every subset.

### 3.3 The engine unit (conservative reading)

Same checkpoints, calibration reduced to one maximum score per calibration engine. Sits 7.9–14.5 points above the window unit at the same nominal level by design — a lifetime maximum is much larger than any single test window. Do not compare widths across units.

| Subset | Method | Emp. coverage | Mean width |
|---|---|---|---|
| FD001 | split_conformal | 0.996 | 66.0 |
| FD001 | nex_scp | 0.988 | 61.2 |
| FD002 | split_conformal | 0.999 | 76.0 |
| FD002 | nex_scp | 0.975 | 64.3 |
| FD003 | split_conformal | 1.000 | 68.8 |
| FD003 | nex_scp | 0.992 | 58.1 |
| FD004 | split_conformal | 0.996 | 78.1 |
| FD004 | nex_scp | 0.977 | 68.8 |

> **Which method to ship?** `nex_scp` is the safest default when coverage must be guaranteed: it tracks nominal at all three levels on all four subsets and never has the widest interval. `split_conformal` is the simplest and almost equally well calibrated. `eara_conformal` and `adaptive` produce narrower intervals but under-cover by 3–5 points on FD003/FD004.

---

## 4. Explainability

The Explainability Reliability Index (ERI) measures cross-method agreement between Integrated Gradients, SHAP and temporal relevance; an uncorrelated pair scores 0.5.

**Certified (five seeds × four subsets, `gru_att_deg_v2`).** Source: `results/multiseed/downstream_summary.csv`.

| Subset | ERI (RUL) | ERI (HI) | Review rate | Mean interval width (cycles) | Constraint violations |
|---|---|---|---|---|---|
| FD001 | 0.925 ± 0.044 | 0.899 ± 0.039 | 0.642 ± 0.086 | 37.1 ± 4.2 | 64.4 ± 1.5 |
| FD002 | 0.979 ± 0.020 | 0.936 ± 0.043 | 0.706 ± 0.043 | 45.4 ± 1.8 | 156.4 ± 5.7 |
| FD003 | 0.879 ± 0.056 | 0.912 ± 0.032 | 0.660 ± 0.083 | 33.2 ± 2.4 | 68.6 ± 3.8 |
| FD004 | 0.886 ± 0.018 | 0.897 ± 0.020 | 0.764 ± 0.039 | 49.0 ± 4.1 | 119.0 ± 4.8 |
| **ALL** | **0.917 ± 0.026** | **0.911 ± 0.018** | 0.693 ± 0.031 | 41.2 ± 1.2 | 102.1 ± 2.2 |

The review rate (0.64–0.76) is higher than the legacy reading (0.370) because the policy is now planned on the conservative end of the interval, and the certified runs use the `degraded_operations` constraint scenario which forces human review of every constrained action. Constraint violations are a real measurement: 38–42% of decisions in each subset violate at least one operational constraint.

> **ERI measures agreement, not faithfulness.** A high ERI means the three methods agree with each other, not that they identify the physically important sensor. Faithfulness validation is pending.

---

## 4.1 The downstream chain on the same checkpoints

`scripts/multiseed_pipeline.py` runs uncertainty → XAI → decision → HITL for every (subset, seed) pair and verifies artifact consistency. Source: `results/multiseed/pipeline_manifest.json`.

| Check | Result |
|---|---|
| Runs executed (4 subsets × 5 seeds) | **20** |
| Runs with no stage error | **20 / 20** |
| Wall-clock for the four stages | 1587 s |
| Cross-stage prediction check | point predictions agree to within 0.1 RUL cycles, Pearson r > 0.9999, for every run |
| `prognostic_model_updated` | `false` on every run — HITL refines the decision policy only |

**Action mix** (share of decisions, pooled over five seeds):

| Subset | Continue | Monitor | Inspect | Schedule maint. | Urgent maint. |
|---|---|---|---|---|---|
| FD001 | 56 % | 5 % | 24 % | 0 % | 15 % |
| FD002 | 56 % | 5 % | 23 % | 0 % | 16 % |
| FD003 | 56 % | 5 % | 24 % | 0 % | 15 % |
| FD004 | 56 % | 2 % | 26 % | 0 % | 16 % |

`SCHEDULE_MAINTENANCE` is absorbed by the no-maintenance-window substitution of the `degraded_operations` scenario. The policy did not adapt in any of the 20 runs (`adaptations_applied` = 0 everywhere) — the three historical feedback records in the log do not reach the adoption threshold.

---

## 4.2 The decision policy

The certified decision mix, with the policy planned on the conservative end of the interval and the `degraded_operations` constraint scenario applied.

| Subset | Decisions | Review rate | Constraint violations |
|---|---|---|---|
| FD001 | 100 | 0.64 | 64.4 |
| FD002 | 259 | 0.71 | 156.4 |
| FD003 | 100 | 0.66 | 68.6 |
| FD004 | 248 | 0.76 | 119.0 |
| **ALL** | 177 | 0.69 | 102.1 |

The action mix is stable across subsets — the policy is dominated by the planning state (a function of the interval), not the seed.

---

## 4.3 Sensitivity of the decision to its own weights

`scripts/decision_sensitivity.py` re-runs the decision engine over a weight space (one-at-a-time + Dirichlet sweep, 200 draws per run). Source: `outputs/reports/decision_sensitivity.json`.

| Weight | Max action-mix TV | Max risk-band switch rate |
|---|---|---|
| `w_rul` | 0.020 | 0.25 |
| `w_hi` | 0.000 | 0.00 |
| `w_u` | 0.016 | 0.23 |
| `w_eri` | 0.000 | 0.00 |

**The decision mix is essentially insensitive to the fusion weights.** The action is determined by the planning state and the ERI band, not by the continuous risk score. The risk ranking of the engines is invariant (Spearman r ≥ 0.999 across the whole sweep).

---

## 4.4 Operational-constraint scenarios

The certified runs use `degraded_operations`, so the constraint layer is exercised by the published results. Source: `outputs/reports/decision_constraint_scenarios.csv`.

| Scenario | Constraint violations | Review rate | Mean severity |
|---|---|---|---|
| `unconstrained` | 0 | 0.49 | 1.13 |
| `no_maintenance_window` | 15 | 0.49 | 0.98 |
| `tight_safety_margin` | 11 | 0.57 | 1.28 |
| `lead_time_pressure` | 10 | 0.49 | 1.23 |
| `resource_shortage` | 28 | 0.49 | 1.13 |
| `degraded_operations` | 65 | 0.57 | 1.13 |

Every rule fires. The maintenance-window rule is the only one that substitutes an action (SCHEDULE_MAINTENANCE → INSPECT) rather than escalating it, which is why `no_maintenance_window` is the only scenario that lowers mean severity.

---

## 5. Key findings

1. **Sensor-derived HI does not hurt RUL accuracy.** Replacing the trivial `HI = RUL / max_rul` label with a genuine sensor-derived index improved the four-subset mean RMSE by 0.079 (13.361 → 13.282), well inside the seed band. The HI head now solves a real second task.

2. **`nex_scp` is the safest interval method.** It tracks the 90% nominal level to within ±2 points on all four subsets, has the most stable coverage across seeds (std 0.003–0.023), and is never the widest interval. No method ranking is statistically established at five seeds.

3. **`eara_conformal` and `adaptive` trade coverage for width.** They are 2–5 points below nominal on FD003/FD004 at 90/95%. Use them when width matters more than the last few points of coverage, and say so.

4. **The decision mix is robust to weight perturbation.** One-at-a-time and Dirichlet sweeps over the risk-score weights move the action-mix TV by at most 0.020. The risk ranking of engines is invariant (Spearman r ≥ 0.999).

5. **Operational constraints are exercised, not bypassed.** Under `degraded_operations`, 38–42% of decisions per subset violate at least one constraint. Every constraint rule fires at least once across the 20 certified runs.

6. **The pipeline is fully reproducible across 20 runs.** All 20 (subset, seed) pairs complete without error; cross-stage prediction agreement is within 0.1 RUL cycles with r > 0.9999.

7. **The four-subset mean RMSE (13.282) is 0.452 behind KESTF (12.83).** This gap exceeds the seed std (0.140) and is a real deficit on the accuracy dimension. The repository's contribution is the calibrated uncertainty + auditable decision layer, not SOTA accuracy.

8. **ERI agreement is high (0.91–0.92) but measures cross-method consistency, not faithfulness.** The IG-vs-Attention term is inflated because temporal relevance re-weights IG attributions rather than being fully independent. Only the IG-vs-SHAP term (weight 0.4) compares genuinely independent methods.

---

## 6. How to reproduce

```powershell
# 1. Place NASA C-MAPSS files in data/raw/ (see README.md -> Dataset)

# 2. Multi-seed certification (Section 2 numbers)
python scripts/multiseed_train.py  --model gru_att_deg_v2 --subsets all --seeds 42 43 44 45 46
python scripts/multiseed_report.py

# 3. Interval benchmark (Section 3)
python scripts/multiseed_conformal_benchmark.py --subsets all --seeds 42 43 44 45 46

# 4. Downstream chain — uncertainty, XAI, decision, HITL (Sections 4–4.4)
python scripts/multiseed_pipeline.py --subsets all --seeds 42 43 44 45 46 --stages all
python scripts/multiseed_downstream.py --model gru_att_deg_v2

# 5. Decision sensitivity (Section 4.3)
python scripts/decision_sensitivity.py --subsets all --seeds 42 43 44 45 46

# 6. Single-run legacy artifacts
python scripts/preprocess.py --subset FD001
python scripts/train.py      --model gru_att_deg_v2 --subset FD001
python scripts/evaluate.py   --model gru_att_deg_v2 --subset FD001
python scripts/summarize.py
python scripts/sota_benchmark.py
```

All benchmark outputs land in `results/` (tracked); all other generated artifacts land in `outputs/` (not tracked).

---

## 7. Caveats

- **Point-prediction numbers are certified over five seeds.** The seed band is ±0.14 on the four-subset mean and ±0.17–0.43 on individual subsets. A single-run comparison to a published number is not evidence of a ranking.
- **The 0.452 gap to KESTF is real.** It exceeds the seed std (0.140). The 0.118 gap to TTSNet is inside the seed band and is not a ranking.
- **Coverage is reported for two calibration units.** The `engine` unit over-covers by design (0.97–1.00 at an 80% target). The `window` unit is the primary number and tracks nominal to within a few points.
- **The finite-sample rank is never clamped.** `conformal_quantile` returns `+inf` when `ceil((n+1)(1−α)) > n`; the interval builder clips an infinite radius to the RUL domain and reports it as a full-width interval.
- **`eara_conformal` and `adaptive` under-cover slightly at 90/95% on some subsets.** Their scales are estimated from training-engine windows, which under-state the uncertainty of a last-window test query. The split and nex_scp intervals are the safer default when coverage must be guaranteed.
- **No interval-method ranking is statistically established.** The exact sign-flip test at n = 5 has a smallest attainable p-value of 0.0625; every Holm-adjusted comparison is ≥ 0.25.
- **Neither interval method has been stress-tested under distribution shift.** All calibration and test engines come from the same C-MAPSS simulation.
- **The ERI measures agreement, not faithfulness.** The IG-vs-Attention term is inflated (§7.5 of ARCHITECTURE.md). Only the IG-vs-SHAP term compares genuinely independent methods.
- **The HITL policy did not adapt in the certified runs.** All 20 runs reuse the same three historical feedback records; no alternative action reaches the adoption threshold. The HITL evidence is "the workflow runs and routes reviews correctly", not "the policy learned from expert feedback".
- **Score-metric values in Section 1 are not directly comparable** across papers without identical RUL clipping and test-endpoint definitions.

See [docs/ARCHITECTURE.md § 7](docs/ARCHITECTURE.md#7-known-gaps) for the full list of known gaps.
