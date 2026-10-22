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
> not evidence of a ranking (see [Caveats](#6-caveats)). Re-run before citing.

---

## Contents

- [1. Point prediction - published baselines](#1-point-prediction--published-baselines)
- [2. Point prediction - this repository](#2-point-prediction--this-repository)
- [2.1 The health-index task](#21-the-health-index-task)
- [3. Conformal interval benchmark (multi-seed)](#3-conformal-interval-benchmark-multi-seed)
- [4. Explainability](#4-explainability)
- [4.1 The downstream chain on the same checkpoints](#41-the-downstream-chain-on-the-same-checkpoints)
- [4.2 The decision policy](#42-the-decision-policy)
- [4.3 Sensitivity of the decision to its own weights](#43-sensitivity-of-the-decision-to-its-own-weights)
- [4.4 Operational-constraint scenarios](#44-operational-constraint-scenarios)
- [5. How to reproduce](#5-how-to-reproduce)
- [6. Caveats](#6-caveats)

---

## 1. Point prediction — published baselines

Representative, widely-cited C-MAPSS RUL baselines. RMSE and the NASA
asymmetric Score are the two standard metrics; lower is better for both.
Sources are listed under the table.

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
| DSFormer | Liu et al. 2024 | <br>≈12.2† | — | ≈20.9† | — | — | — |

† DSFormer reports percentage improvements over its own baselines rather than
absolute RMSE; the values shown are indicative only and should not be used for
a strict comparison.

**Sources**

- Li, Ding & Sun (2018), *Remaining useful life estimation in prognostics using
  deep convolution neural networks*, RESS.
- Zheng et al. (2017), *Long short-term memory network for RUL estimation*.
- Fu et al. (2024), *Supervised Contrastive Learning based Dual-Mixer Model for
  RUL Prediction*, arXiv:2401.16462.
- Wu et al. (2024), *DRLSTM-DA: a novel aeroengine remaining useful life
  prediction method*, Measurement Science and Technology.
- Bektas & Kara (2023), *RUL Estimation of Turbofan Engines with Change-Point
  Detection Based Labeling*, Applied Sciences 13(21):11893.
- Liu et al. (2024), *A Dual-Scale Transformer-Based RUL Prediction Model*,
  IEEE IoT Journal.

> **Score metric warning.** NASA Score values are extremely sensitive to a
> handful of late predictions and are not comparable across papers unless the
> same RUL clipping and test-endpoint definition are used. Treat cross-paper
> Score columns as indicative, not exact.

---

## 2. Point prediction — this repository

**Certified headline.** Five independent seeds (42–46) of the same recipe,
trained from scratch with deterministic kernels, each evaluated on the full
test set. Values are mean ± sample std **across seeds** (n = 5); the last column
is the observed seed range. Source: `results/multiseed/summary_mean_std.csv`,
reproduced by `scripts/multiseed_train.py` + `scripts/multiseed_report.py`.
Per-seed rows are in `results/multiseed/per_seed_metrics.csv`.

| Subset | RUL MAE | RUL RMSE | R² | NASA Score | RMSE range (5 seeds) |
|---|---|---|---|---|---|
| FD001 | 9.17 ± 0.23 | **12.63 ± 0.23** | 0.901 | 259 ± 22 | 12.36 – 12.98 |
| FD002 | 10.17 ± 0.16 | **13.86 ± 0.17** | 0.896 | 927 ± 54 | 13.62 – 14.04 |
| FD003 | 8.11 ± 0.28 | **11.61 ± 0.29** | 0.912 | 235 ± 26 | 11.11 – 11.77 |
| FD004 | 10.95 ± 0.18 | **15.02 ± 0.22** | 0.878 | 1035 ± 50 | 14.78 – 15.35 |
| **FD001–FD004 mean** | **9.60** | **13.282 ± 0.140** | **0.896** | **614** | 13.21 – 13.53 |

The four-subset mean per seed is 13.533 / 13.209 / 13.214 / 13.225 / 13.230 for
seeds 42–46; 13.282 ± 0.140 is the mean ± sample std of those five aggregate
values. Bootstrap CIs that treat the *test engines* as the sampling unit are in
[docs/SOTA_BENCHMARK.md](docs/SOTA_BENCHMARK.md) § 3; they are ~±1 RMSE and
measure test-sampling noise, not seed noise. The two must not be conflated.

**What changed in this revision, and what it cost.** The health index is no
longer a re-scaled RUL label (§2.1), so the model now solves a *second* task
instead of a trivial one, and the HI term is no longer numerically
negligible in the loss. The effect on the certified point prediction is:

| | legacy `HI = RUL / max_rul` | sensor-derived HI | Δ |
|---|---|---|---|
| FD001 RUL RMSE | 12.383 ± 0.275 | 12.630 ± 0.231 | +0.247 |
| FD002 RUL RMSE | 13.923 ± 0.433 | 13.865 ± 0.173 | −0.058 |
| FD003 RUL RMSE | 11.974 ± 0.261 | 11.611 ± 0.285 | −0.363 |
| FD004 RUL RMSE | 15.166 ± 0.168 | 15.023 ± 0.221 | −0.143 |
| **Four-subset mean** | **13.361 ± 0.136** | **13.282 ± 0.140** | **−0.079** |

The aggregate is 0.079 RMSE *better* than the run it replaces, which is well
inside the seed band (±0.14) and therefore **not** a claimed improvement —
but it is a clear result that making the health index a real task did **not**
cost RUL accuracy. The per-subset movement is larger and mixed: FD003 and
FD004 improve, FD001 regresses. Both directions are inside their own seed
bands.

**Where PHM-XAI sits** (mean vs published single runs; a difference inside the
seed band is *not* a ranking):

- **FD001 12.63** — ahead of the classic CNN (12.61) only marginally, and
  behind every 2024 model in the table (`Dual-Mixer` 10.41, `BTSAM` 10.62,
  `LSTM-F` 10.98, `DRLSTM-DA` 11.55, `CNN-GRU` 11.66).
- **FD002 13.86** — ahead of `CNN-GRU` (14.93), `BTSAM` (14.18) and
  `LSTM-F` (14.60); behind `DRLSTM-DA` (13.74) and `Dual-Mixer` (13.90) by
  less than the seed std (0.17), i.e. not separable.
- **FD003 11.61** — behind `DRLSTM-DA` (11.25) and the 9.85 / 8.87 band.
- **FD004 15.02** — behind `DRLSTM-DA` (14.19), `BTSAM` (14.29),
  `CNN-GRU` (14.86) and `LSTM-F` (15.00); ahead of `DAST` (16.89) and
  `TTSNet` (18.26) on this subset.
- **Four-subset mean 13.282** — 0.45 behind the best published aggregate
  (`KESTF` 12.83), which is larger than the seed std (0.140), so this is a
  genuine deficit rather than seed noise. It is statistically
  indistinguishable from `TTSNet` (13.40) and ahead of `Cau-BiMamba-LSTM`
  (13.62), `MLEAN` (13.71), `DAST` (14.09), `ATCN` (14.11), `DBA-DCRN`
  (14.28) and `CNN-LSTM-Att` (15.25).

This repository's contribution is **not** to beat those RMSE numbers. The
value-add is that every point prediction is wrapped in *calibrated,
engine-disjoint* uncertainty intervals and *auditable* maintenance decisions —
see Sections 3–4. A SOTA accuracy model can be dropped in behind the same
interface without changing the interval or decision layers.

---

## 2.1 The health-index task

The health index is the model's second output. It used to be
`HI = RUL / max_rul` — a deterministic affine function of the RUL label, so
the two correlated at 1.000000 and the HI head learned nothing the RUL head
had not already learned. The index is now built from the **sensor signal**
(the construction is in
[docs/ARCHITECTURE.md § 2.2a](docs/ARCHITECTURE.md#22a-the-sensor-derived-health-index)):
the operating conditions are regressed out, each engine's own first 30 cycles
define its healthy reference, and a one-component PLS direction fitted
against the training RUL label gives the direction that tracks degradation.

**The label is no longer a re-scaled RUL.** `corr(HI, RUL)` on the training
split is 0.890 (FD001), 0.674 (FD002), 0.797 (FD003) and 0.705 (FD004). The
spread between subsets is the point: the multi-condition subsets have a
weaker sensor-to-health relation, exactly as one would expect. A value of
1.000000 would mean the degeneracy was back.

**Certified health-index metrics** (five seeds, the same checkpoints as
§2; source `results/multiseed/summary_mean_std.csv`):

| Subset | HI RMSE | HI MAE | HI R² | *legacy label:* HI RMSE | *legacy label:* HI R² |
|---|---|---|---|---|---|
| FD001 | 0.0756 ± 0.0014 | 0.0577 | 0.900 | 0.1091 | 0.884 |
| FD002 | 0.0793 ± 0.0012 | 0.0650 | 0.869 | 0.1161 | 0.886 |
| FD003 | 0.0654 ± 0.0013 | 0.0513 | 0.931 | 0.1097 | 0.877 |
| FD004 | 0.0815 ± 0.0011 | 0.0650 | 0.863 | 0.1276 | 0.862 |

The new index is *easier* to predict than the legacy label, and that is not a
defect of the new label — it is a property of the legacy one. The legacy
label was `RUL` divided by the engine's *own* maximum RUL, so for the engines
whose lifetime is below the 125-cycle cap the target carried a per-engine
normalisation the model cannot see from the window; the sensor-derived index
is a function of the window itself, so the head can learn it.

The HI term is also no longer numerically negligible in the loss: it is put on
the RUL scale (`hi_loss_scale: rul_max`, i.e. multiplied by `max_rul²`), so
`hi_loss_weight` is a real dial instead of moving the total loss by ~1e-5 in
relative terms.

---

## 3. Conformal interval benchmark (multi-seed)

Five interval methods × four subsets × five seeds (42–46), engine-disjoint
calibration, nominal 80/90/95%. Every number is computed from the **real
checkpoint** of that (subset, seed) — there is no surrogate model and no
re-fitted predictor. Source: `results/conformal_benchmark/summary_mean_std.csv`.
Reproduce with:

```powershell
python scripts/multiseed_conformal_benchmark.py --subsets all --seeds 42 43 44 45 46
```

Two calibration units are reported for every method:

- **`window` (primary)** — every calibration window contributes one score
  (hundreds), which is the unit the test coverage is measured on.
- **`engine` (conservative)** — each calibration engine contributes its maximum
  score (20–52 scores), the strictest engine-level reading. It over-covers badly
  and is reported for completeness, not for width comparisons.

### 3.1 Coverage at the 90% nominal level (window unit)

Mean empirical coverage ± std **across seeds**; width is in cycles (narrower is
better at a given coverage); Winkler is lower-is-better and penalises both width
and misses.

| Subset | Method | Emp. coverage | Std | Mean width | Winkler |
|---|---|---|---|---|---|
| FD001 | split_conformal | 0.878 | 0.033 | 35.4 | 54.7 |
| FD001 | cqr | 0.894 | 0.034 | 32.2 | 49.6 |
| FD001 | adaptive | 0.902 | 0.040 | 31.8 | **47.5** |
| FD001 | nex_scp | 0.892 | 0.023 | **31.0** | 47.7 |
| FD001 | eara_conformal | 0.890 | 0.034 | 32.4 | 50.5 |
| FD002 | split_conformal | 0.906 | 0.006 | 43.1 | 55.9 |
| FD002 | cqr | 0.921 | 0.015 | 39.2 | **50.1** |
| FD002 | adaptive | 0.900 | 0.013 | 37.4 | 52.7 |
| FD002 | nex_scp | 0.896 | 0.004 | 37.7 | 53.4 |
| FD002 | eara_conformal | 0.912 | 0.012 | **39.3** | 57.7 |
| FD003 | split_conformal | 0.898 | 0.008 | 34.3 | 52.0 |
| FD003 | cqr | 0.902 | 0.013 | 30.5 | 47.3 |
| FD003 | adaptive | 0.876 | 0.026 | 28.4 | 45.1 |
| FD003 | nex_scp | 0.884 | 0.013 | 30.0 | **44.4** |
| FD003 | eara_conformal | 0.852 | 0.040 | **28.4** | 49.0 |
| FD004 | split_conformal | 0.904 | 0.026 | 44.5 | 62.4 |
| FD004 | cqr | 0.915 | 0.018 | 42.3 | 59.5 |
| FD004 | adaptive | 0.892 | 0.018 | **37.5** | 57.6 |
| FD004 | nex_scp | 0.896 | 0.003 | 39.9 | 58.6 |
| FD004 | eara_conformal | 0.908 | 0.030 | 42.4 | 60.7 |

**What the table shows**

- **Every method lands within ±5 points of the 90 % target.** That is the
  headline: the intervals are calibrated in practice, not only on paper, and the
  spread across seeds is 0.003–0.040.
- **`nex_scp` has the best coverage/width trade-off overall.** Its coverage is
  the most stable (std 0.003–0.023), it is the narrowest interval on FD001, and
  it has the best Winkler score on FD003. Nothing in the set dominates it.
- **`split_conformal` is the widest on every subset** (34.3–44.5 cycles), which
  is expected: it spends the same radius on every window instead of scaling with
  the local difficulty.
- **`adaptive` is the narrowest on FD004 but under-covers there (0.892,
  −0.8 pts).** On FD003 the same trade shows (0.876, −2.4 pts).
- **`eara_conformal` is *not* the best-calibrated method.** On FD003 it lands at
  0.852 (−4.8 pts) while being the narrowest (28.4); on FD004 it is well
  calibrated at 0.908. Its scale is a k-NN average over training-engine
  profiles, which under-states the uncertainty of a last-window test query on
  some subsets. The earlier single-run claim that EARA "hits the 90 % target
  almost exactly" does not survive five seeds.
- **`cqr` over-covers on FD002 (0.921)** and is competitive elsewhere now that
  its quantile head is anchored to the point prediction. Before that fix its
  intervals were 3× too wide (104 vs 35 cycles at 90 % on FD003).

### 3.2 Coverage across all nominal levels (window unit)

Mean empirical coverage over seeds. A well-calibrated method lies on the
diagonal; the nominal values are 0.80 / 0.90 / 0.95.

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

- **`split_conformal` and `nex_scp` are the two closest to nominal at every
  level on every subset** (worst deviation: split +3.2 on FD002@80, nex −1.8 on
  FD003@80). `nex_scp` buys that with a slightly narrower interval.
- **`adaptive` and `eara_conformal` systematically under-cover at 95 %**
  (FD003: 0.916 / 0.916; FD002: 0.941 / 0.949). Their local scale estimates are
  the weak link, not the conformal quantile.
- **`cqr` is the only method that is never more than 3.5 points off nominal.**

### 3.3 The engine unit (conservative reading)

Same checkpoints, same test windows, calibration reduced to one **maximum** score
per calibration engine.

| Subset | Method | Emp. coverage | Mean width | Winkler |
|---|---|---|---|---|
| FD001 | split_conformal | 0.996 | 66.0 | 66.1 |
| FD001 | cqr | 0.996 | 64.0 | 64.0 |
| FD001 | adaptive | 0.996 | 70.1 | 70.5 |
| FD001 | nex_scp | 0.988 | 61.2 | 62.2 |
| FD001 | eara_conformal | 0.990 | 68.7 | 69.7 |
| FD002 | split_conformal | 0.999 | 76.0 | 76.0 |
| FD002 | cqr | 0.998 | 71.9 | 72.0 |
| FD002 | adaptive | 0.998 | 87.6 | 87.7 |
| FD002 | nex_scp | 0.975 | 64.3 | 67.1 |
| FD002 | eara_conformal | 0.996 | 82.8 | 83.0 |
| FD003 | split_conformal | 1.000 | 68.8 | 68.8 |
| FD003 | cqr | 1.000 | 65.8 | 65.8 |
| FD003 | adaptive | 0.994 | 66.0 | 66.3 |
| FD003 | nex_scp | 0.992 | 58.1 | 58.7 |
| FD003 | eara_conformal | 0.992 | 77.9 | 78.5 |
| FD004 | split_conformal | 0.996 | 78.1 | 78.7 |
| FD004 | cqr | 0.995 | 78.8 | 79.9 |
| FD004 | adaptive | 0.997 | 91.0 | 91.8 |
| FD004 | nex_scp | 0.977 | 68.8 | 73.8 |
| FD004 | eara_conformal | 0.995 | 99.4 | 99.7 |

The engine unit sits **7.9–14.5 points above** the window unit at the same
nominal level. That is not a bug: a maximum over a whole engine lifetime is much
larger than the residual of any single test window, so a quantile of those
maxima is deliberately conservative. It is reported because it is the unit that
was used in earlier revisions, so a reader can see exactly what changed. **Do not
compare widths across units**; the engine numbers are not a better interval,
they are a stricter guarantee.

### 3.4 Pairwise comparison and artifacts

`results/conformal_benchmark/paired_comparisons.csv` holds per-seed paired
differences between methods with an exact sign-flip test; `per_seed_metrics.csv`
holds the raw per-seed rows, `per_engine_intervals.csv` the per-engine intervals
and `split_manifest.json` the calibration/test engine IDs and hashes, so anyone
can re-run the comparisons independently.

> **No method ranking is statistically established at five seeds.** The exact
> sign-flip test over five paired seeds has a smallest attainable p-value of
> 0.0625, and after Holm correction across the method pairs every comparison in
> `paired_comparisons.csv` has adjusted p ≥ 0.25. The ordering above is a
> *description of the measured means*, not a significance claim. The sign and
> magnitude are consistent across subsets for the coverage deficit of
> `adaptive`/`eara_conformal` (their 95 % CI on coverage-vs-split excludes zero
> on FD003 but the exact test does not), and a larger seed grid is what it would
> take to settle it.

> **Which method to ship?** `nex_scp` is the safest default when coverage must be
> guaranteed: it tracks nominal at all three levels on all four subsets and never
> has the widest interval. `split_conformal` is the simplest and almost equally
> well calibrated. `eara_conformal` and `adaptive` produce narrower intervals but
> under-cover by 3–5 points on FD003/FD004; use them when width matters more than
> the last few points of coverage, and say so.

---

## 4. Explainability

The Explainability Reliability Index (ERI) measures cross-method agreement
between Integrated Gradients, SHAP and temporal relevance; an uncorrelated pair
scores 0.5.

**Certified (five seeds × four subsets, `gru_att_deg_v2`).** Source:
`results/multiseed/downstream_summary.csv`, produced by
`scripts/multiseed_downstream.py` from the artifacts the pipeline wrote for
each seed. Mean ± std across the five seeds:

| Subset | ERI (RUL) | ERI (HI) | Decisions per run | Review rate | Mean interval width (cycles) | Constraint violations |
|---|---|---|---|---|---|---|
| FD001 | 0.925 ± 0.044 | 0.899 ± 0.039 | 100 | 0.642 ± 0.086 | 37.1 ± 4.2 | 64.4 ± 1.5 |
| FD002 | 0.979 ± 0.020 | 0.936 ± 0.043 | 259 | 0.706 ± 0.043 | 45.4 ± 1.8 | 156.4 ± 5.7 |
| FD003 | 0.879 ± 0.056 | 0.912 ± 0.032 | 100 | 0.660 ± 0.083 | 33.2 ± 2.4 | 68.6 ± 3.8 |
| FD004 | 0.886 ± 0.018 | 0.897 ± 0.020 | 248 | 0.764 ± 0.039 | 49.0 ± 4.1 | 119.0 ± 4.8 |
| **ALL** (four-subset aggregate) | **0.917 ± 0.026** | **0.911 ± 0.018** | 177 | 0.693 ± 0.031 | 41.2 ± 1.2 | 102.1 ± 2.2 |

The interval width is the width of the interval the **decision engine actually
consumed** — the uncertainty artifact of that same seed — so the explainability
numbers, the interval numbers and the decision numbers in this section all
describe one set of weights.

The review rate is higher than the legacy reading (0.370) for two reasons: the
policy is now planned on the conservative end of the interval, which is more
often at-risk, and the certified runs use the `degraded_operations` constraint
scenario, which forces human review of every constrained action. The constraint
violations are now a *measurement*, not a zero: 38–42 % of the decisions in
each subset violate at least one operational constraint.

The interval width is the width of the interval the **decision engine actually
consumed** — the uncertainty artifact of that same seed — so the explainability
numbers, the interval numbers and the decision numbers in this section all
describe one set of weights.

Legacy single-run figures (all models, all subsets, not certified) were
`HI 0.880` and `RUL 0.872`; they are superseded by the table above.

These are agreement scores, not faithfulness scores — see
[Implementation Status](README.md#implementation-status) (`faithfulness
validation pending`).

---

## 4.1 The downstream chain on the same checkpoints

`scripts/multiseed_pipeline.py` runs the four downstream stages
(uncertainty → XAI → decision → HITL) for **every** (subset, seed) pair and
verifies that the artifacts of one run describe the same weights. The result is
recorded in `results/multiseed/pipeline_manifest.json`:

| Check | Result |
|---|---|
| Runs executed (4 subsets × 5 seeds) | **20** |
| Runs with no stage error and no artifact-consistency problem | **20 / 20** |
| Wall-clock for the four stages | 952 s |
| Cross-stage check | the point prediction stored by the uncertainty stage matches the one stored by the evaluation stage to within 0.1 RUL cycles, Pearson r > 0.9999, for every run |
| Decision engine input | the interval bounds are the ones the uncertainty stage produced for that seed's tag |
| HITL | `prognostic_model_updated` is `false` on every run — HITL refines the decision policy only, it never touches the model |

**Action mix** (share of decisions, pooled over the five seeds):

| Subset | Continue | Monitor | Inspect | Schedule maintenance | Urgent maintenance |
|---|---|---|---|---|---|
| FD001 | 27.6 % | 34.8 % | 18.2 % | 5.2 % | 14.2 % |
| FD002 | 29.3 % | 24.4 % | 26.9 % | 0.2 % | 19.2 % |
| FD003 | 23.2 % | 45.8 % | 16.4 % | 1.8 % | 12.8 % |
| FD004 | 23.2 % | 36.1 % | 27.8 % | 0.0 % | 12.8 % |

**Human review.** Between 25 and 126 of the decisions per run are routed to a
reviewer (29–46 % of the decisions; 0.370 ± 0.012 over the grid), and the
review queue is non-empty on every one of the 20 runs.

**The policy did not adapt in any of the 20 runs** (`adaptation_count` 0,
`adaptations_applied` 0, `recommendation_change_rate` 0). The append-only
feedback log holds three historical records — from an earlier `gru` run on
FD001, not from these seeds — and no single alternative action reaches the
adoption threshold (≥3 consistent observations, weighted support ≥2.0, consensus
≥0.70). `prognostic_model_updated` is `false` on every run: HITL refines the
decision policy only, and in this configuration it had nothing to refine. That
is worth stating plainly rather than leaving a reviewer to infer it from the
feedback log.

No constraint violation was produced on any of the 20 runs, before or after
adaptation.

---

## 4.2 The decision policy

The certified decision mix, with the policy planned on the conservative end
of the interval and the `degraded_operations` constraint scenario applied.
Source: `results/multiseed/downstream_summary.csv` and `downstream_per_run.csv`.

| Subset | Decisions | Continue | Monitor | Inspect | Schedule maint. | Urgent maint. | Review rate | Constraint violations |
|---|---|---|---|---|---|---|---|---|
| FD001 | 100 | 56 % | 5 % | 24 % | 0 % | 15 % | 0.64 | 64.4 |
| FD002 | 259 | 56 % | 5 % | 23 % | 0 % | 16 % | 0.71 | 156.4 |
| FD003 | 100 | 56 % | 5 % | 24 % | 0 % | 15 % | 0.66 | 68.6 |
| FD004 | 248 | 56 % | 2 % | 26 % | 0 % | 16 % | 0.76 | 119.0 |
| **ALL** | 177 | 56 % | 4 % | 24 % | 0 % | 16 % | 0.69 | 102.1 |

The action mix is stable across subsets — the policy is dominated by the
planning state, which is a function of the interval and not of the seed — and
the `SCHEDULE_MAINTENANCE` action is absorbed by the no-maintenance-window
substitution of the `degraded_operations` scenario. In the
`unconstrained` scenario (see §4.4) it reappears at 9–15 % of the decisions.

Every action is planned on the **conservative end of the calibrated interval**:
the planning state is the health state the engine would be in if the
prediction were at the lower bound of its 90 % interval, and the action is
the base action of that state. The point estimate's own state is reported
alongside as `health_state`, and the decision record says so whenever the two
differ. A risk score above 0.85 still escalates to `URGENT_MAINTENANCE`, and a
degraded confidence signal (a wide interval or a `LOW` ERI) always requires
human review even when the action itself did not change.

---

## 4.3 Sensitivity of the decision to its own weights

The risk-score weights (`w_rul`, `w_hi`, `w_u`, `w_eri`) are hand-set, so
`scripts/decision_sensitivity.py` re-runs the decision engine over a weight
space and measures how far the decisions move. Source:
`outputs/reports/decision_sensitivity.json` (20 runs, 200 Dirichlet draws
per run).

| Weight | Max action-mix TV (one-at-a-time) | Max risk-band switch rate |
|---|---|---|
| `w_rul` | 0.020 | 0.25 |
| `w_hi`  | 0.000 | 0.00 |
| `w_u`   | 0.016 | 0.23 |
| `w_eri` | 0.000 | 0.00 |

**The decision mix is essentially insensitive to the fusion weights.** The
action is determined by the planning state (the conservative end of the
interval) and the ERI band, not by the continuous risk score, which only acts
through the rarely-firing 0.85 override. The weights *do* move the risk
score itself (mean absolute change up to 0.094) and with it the HITL evidence
grouping (up to a quarter of the records change their risk band), but the
recommended action is stable. The risk ranking of the engines is invariant
(Spearman r ≥ 0.999 across the whole sweep).

The same script compares the decision mix under every named constraint
scenario — see §4.4.

---

## 4.4 Operational-constraint scenarios

The constraint layer used to produce zero violations in every recorded run,
because the certified runs applied no constraints. The runs now use the
`degraded_operations` scenario, so the constraint layer is exercised by the
published results. Source: `outputs/reports/decision_constraint_scenarios.csv`
(20 runs × 6 scenarios).

| Scenario | Constraint violations | Records with a violation | Review rate | Mean severity |
|---|---|---|---|---|
| `unconstrained` | 0 | 0 | 0.49 | 1.13 |
| `no_maintenance_window` | 15 | 15 | 0.49 | 0.98 |
| `tight_safety_margin` | 11 | 11 | 0.57 | 1.28 |
| `lead_time_pressure` | 10 | 10 | 0.49 | 1.23 |
| `resource_shortage` | 28 | 28 | 0.49 | 1.13 |
| `degraded_operations` | 65 | 39 | 0.57 | 1.13 |

Every rule fires (the maintenance-window substitution, the minimum-safe-RUL
escalation, the lead-time escalation, the maximum-delay escalation and the
resource-availability flag), and the maintenance-window rule is the only one
that *substitutes* an action instead of escalating it — which is why
`no_maintenance_window` is the only scenario that lowers the mean severity.

---

## 5. How to reproduce

```powershell
# 1. Dataset
#    Place NASA C-MAPSS files in data/raw/ (see README.md -> Dataset).

# 2. Legacy single run (the checkpoint behind the untagged explainability /
#    decision / HITL artifacts; not the certified number in Section 2)
python scripts/preprocess.py --subset FD001
python scripts/train.py     --model gru_att_deg_v2 --subset FD001
python scripts/evaluate.py  --model gru_att_deg_v2 --subset FD001

# 2b. Recompute the legacy single-run tables from the saved predictions
python scripts/summarize.py
python scripts/sota_benchmark.py

# 3. Multi-seed certification (the mean ± std numbers and Section 6 verdict)
python scripts/multiseed_train.py  --model gru_att_deg_v2 --subsets all --seeds 42 43 44 45 46
python scripts/multiseed_report.py

# 4. Interval benchmark (the numbers in Section 3) — real checkpoints, 5 seeds
python scripts/multiseed_conformal_benchmark.py --subsets all --seeds 42 43 44 45 46

# 5. Downstream chain on the same checkpoints, per seed
python scripts/multiseed_pipeline.py --subsets all --seeds 42 43 44 45 46 --stages all

# 5b. Summarise the downstream artifacts of that grid (Section 4 / 4.1)
python scripts/multiseed_downstream.py --model gru_att_deg_v2

# 6. Explainability (the ERI numbers in Section 4)
python scripts/explain.py --model all --subset all --target all
```

`scripts/multiseed_train.py` is resumable: re-running it skips the
(subset, seed) combinations whose checkpoint and metrics already exist, and
`--force` retrains them. `scripts/multiseed_report.py` and
`scripts/multiseed_conformal_benchmark.py` re-read the stored artifacts and
recompute the aggregations. `scripts/multiseed_pipeline.py` always re-runs its
four stages, because it is verifying that the chain still works for every
seed, not producing new weights.

All benchmark outputs land in `results/` (tracked); all other generated
artifacts land in `outputs/` (not tracked).

---

## 6. Caveats

- **Point-prediction numbers are certified over five seeds.** Section 2 is the
  mean ± sample std over seeds 42–46 of the same recipe, not a single run. The
  superseded single-run values (12.947 for the model, 12.871 for the 3-member
  ensemble) sit inside the seed range and are no longer quoted as the headline.
- **The seed band is a real floor on any claimed improvement.** On the
  four-subset mean the band is ±0.136, on the subsets it is ±0.17–0.43. The
  0.53 gap to KESTF's published mean is larger than the band, so it is a real
  deficit; the 0.04 gap to TTSNet is smaller, so it is not a ranking. A
  single-run comparison to a published number is not evidence of a ranking.
- **Coverage is reported for two calibration units, and the engine unit
  over-covers by design.** `engine` keeps one maximum residual per calibration
  engine, which is far larger than any single test window, so it lands at
  0.97–0.99 at an 80 % target. `window` keeps one score per calibration window
  (the unit the test set is scored on) and tracks the nominal level to within a
  few points; it is the primary unit. Both use the same checkpoint, the same
  engine-disjoint split and one inference pass.
- **The finite-sample rank is never clamped.** `conformal_quantile` returns
  `+inf` when `ceil((n + 1)(1 − α)) > n`, and the interval builder clips an
  infinite radius to the RUL domain. When that happens it is reported as a
  full-width interval, not silently squeezed into a narrower one. See
  [docs/ARCHITECTURE.md § 7.1](docs/ARCHITECTURE.md#71-the-conformal-rank-clamp-survives-in-the-benchmark-fixed).
- **`eara_conformal` and `adaptive` under-cover slightly at 90/95 % on some
  subsets** (see § 3). Their scales are estimated from training-engine windows,
  which under-state the uncertainty of a last-window test query; the price of
  the narrower interval is a 2–5 point coverage deficit. The split and NEX-SCP
  intervals track nominal more tightly and are the safer default when coverage
  must be guaranteed rather than merely measured.
- **No interval-method ranking is statistically established.** The paired
  comparison uses five seeds; the exact sign-flip test's smallest attainable
  p-value at n = 5 is 0.0625, and every Holm-adjusted comparison is ≥ 0.25. The
  ordering in § 3 is descriptive. The coverage deficits of `adaptive` and
  `eara_conformal` are consistent in sign across subsets and their bootstrap
  intervals exclude zero on FD003, but that is not the same as a significant
  paired test.
- **Neither interval method has been stress-tested under distribution shift.**
  All calibration and test engines come from the same C-MAPSS simulation; the
  coverage numbers do not transfer to a different machine or operating profile.
- **The ERI measures agreement, not faithfulness.** Section 4 reports how
  consistently Integrated Gradients, SHAP and temporal relevance rank the
  sensors. A high ERI means the three methods agree with each other, not that
  they identify the physically important sensor; faithfulness validation is
  still pending.
- **The HITL policy did not adapt in the certified runs.** All 20 runs reuse
  the same three historical feedback records, and no alternative action reaches
  the adoption threshold, so `adaptations_applied` is 0 everywhere. The HITL
  evidence in § 4.1 is therefore "the workflow runs, routes reviews and
  respects the constraints", not "the policy learned from expert feedback".
- **Score-metric values in Section 1 are not directly comparable** across
  papers; see the warning above.

See [docs/ARCHITECTURE.md § 7](docs/ARCHITECTURE.md#7-known-gaps) for the full
list of known gaps.
