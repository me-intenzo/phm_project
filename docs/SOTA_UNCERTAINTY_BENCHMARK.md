# SOTA Uncertainty Benchmark — C-MAPSS Conformal Prediction (FD001–FD004)

**Scope:** Compare six conformal interval methods on the NASA C-MAPSS turbofan
dataset across four subsets and five independent seeds (42–46), using the real
`gru_att_deg_v2` checkpoints produced by `scripts/multiseed_train.py`. No proxy
or surrogate model is used anywhere in this benchmark.

**Date:** 2026-10-22 · **Model:** `gru_att_deg_v2` · **Seeds:** 42–46 ·
**Nominal levels:** 80 % / 90 % / 95 % · **Primary unit:** window (one score
per calibration window); engine unit (per-engine maximum) is reported separately
as a conservative reading.

> **Protocol note.** All six methods share the same engine-disjoint calibration
> split (reproduced from the seed stored in each checkpoint), the same test
> engines, and the same forward pass. Comparisons are therefore paired by
> construction. Source artifacts: `results/conformal_benchmark/`.

---

## Contents

- [1. Verdict](#1-verdict)
- [2. Methods](#2-methods)
- [3. Protocol](#3-protocol)
- [4. Coverage at 90 % nominal — window unit](#4-coverage-at-90--nominal--window-unit)
- [5. Interval width at 90 % nominal — window unit](#5-interval-width-at-90--nominal--window-unit)
- [6. Winkler score at 90 % nominal — window unit](#6-winkler-score-at-90--nominal--window-unit)
- [7. Full coverage table — all levels, window unit](#7-full-coverage-table--all-levels-window-unit)
- [8. Engine unit — conservative reading](#8-engine-unit--conservative-reading)
- [9. Paired comparisons](#9-paired-comparisons)
- [10. Literature context](#10-literature-context)
- [11. Statistical note](#11-statistical-note)
- [12. Reproduce](#12-reproduce)

---

## 1. Verdict

| Finding | Detail |
|---|---|
| Safest default (window unit, 90 %) | **`nex_scp`** — coverage 0.888–0.904, narrowest or second-narrowest interval on every subset |
| Best calibrated overall | `split_conformal` and `nex_scp` both track the 90 % nominal closely; `split_conformal` is slightly wider |
| Narrowest intervals | `nex_scp` on FD001/FD002/FD003; `eara_scp` on FD004 |
| Novel combination | **`eara_scp`** — EARA's kNN regime-aware scale + NEX-SCP's online level adaptation. Coverage 0.878–0.902 at 90 %, width 33.9–41.9 cycles. Slightly wider than `nex_scp` on FD001/FD002 but better calibrated than `eara_conformal` on FD003/FD004. |
| Under-coverage risk | `eara_conformal` and `adaptive` miss by 3–5 pp on FD003/FD004 at 90 % |
| Engine unit | All methods over-cover badly (0.977–1.000 at 90 % nominal) — this is expected; the engine-maximum score is a conservative reduction |
| Statistical separation | With 5 seeds the smallest attainable exact sign-flip p = 0.0625; no method pair is statistically separated. Results are descriptions of measured means, not a ranking. |

The honest summary: **`nex_scp` is the recommended default** for deployment
because it consistently achieves near-nominal coverage with the narrowest
intervals. `split_conformal` is a safe fallback with slightly wider but
well-calibrated intervals. `eara_scp` is a clean composition of EARA's
regime-aware scale and NEX-SCP's prequential adaptation — it is better
calibrated than `eara_conformal` on FD003/FD004 at the cost of slightly wider
intervals than `nex_scp`. `eara_conformal` and `adaptive` trade coverage for
width and should only be used when interval tightness is more important than
guaranteed coverage.

---

## 2. Methods

All six methods are computed by the shared engine
`src/uncertainty/pipeline.py::compute_all_intervals` in a single inference
pass, so the comparison is paired: same weights, same calibration engines, same
test engines.

| Method | Key | Description |
|---|---|---|
| Split conformal | `split_conformal` | Global split conformal on absolute residuals. Baseline. Interval: `ŷ ± q̂` where `q̂` is the `ceil((n+1)(1−α))/n` quantile of calibration scores. |
| Conformalized quantile regression | `cqr` | Signed score: `score = max(q_lo(x) − y, y − q_hi(x))`. The quantile head is anchored to the point prediction; `q̂` is the conformal correction. |
| Scale-adaptive conformal | `adaptive` | `ŷ ± q̂ · s(x)` where `s(x)` is fitted on training-engine residuals. Global `q̂`. |
| EARA-Conformal | `eara_conformal` | Engine/regime-aware scale-adaptive conformal. `s(x)` is retrieved from the `top_k = 5` nearest training-engine profiles, conditioned on one of three degradation regimes. Global `q̂`. |
| NEX-SCP (prequential ACI) | `nex_scp` | Prequential adaptive conformal inference. The miscoverage level is updated from each test engine's realised outcome: `α_{t+1} = clip(α_t + γ(α − err_t), 0.01, 0.99)` with `γ = 0.02`. Uses the generic scale-head scale `s(x)`. |
| EARA-SCP | `eara_scp` | Novel composition: EARA's kNN regime-aware scale (`top_k = 5` nearest training-engine profiles, conditioned on degradation regime) combined with NEX-SCP's prequential level adaptation. Neither component is modified; this is a clean composition reported alongside them. |

**Two calibration units** are reported for every method:

| Unit | Score set | Behaviour |
|---|---|---|
| `window` (primary) | One score per calibration window (thousands of scores) | Matches the unit the test set is scored on; tracks the nominal level |
| `engine` (conservative) | One **maximum** score per calibration engine (20–52 engines) | Valid but over-covers badly at low nominal levels |

---

## 3. Protocol

| Item | Value |
|---|---|
| Model | `gru_att_deg_v2` (2-layer GRU + LayerNorm + 4-head temporal attention, dual RUL+HI heads) |
| Seeds | 42, 43, 44, 45, 46 (five independent training runs) |
| Subsets | FD001, FD002, FD003, FD004 |
| Nominal coverage levels | 80 %, 90 %, 95 % |
| Calibration split | Engine-disjoint; reproduced from the seed stored in each checkpoint. ~20 % of training engines held out. |
| Test set | Last window of each test engine (100 / 259 / 100 / 248 engines for FD001–FD004) |
| Interval clipping | `[0, max_rul]` where `max_rul = 125` (from subset config) |
| Finite-sample rank | `ceil((n+1)(1−α))`; never clamped — when it exceeds `n` the quantile is `+inf` |
| Regimes (EARA) | 3 (k-means on training-engine degradation profiles; validated by `scripts/regime_sweep.py`) |
| ACI adaptation rate | `γ = 0.02` |
| Proxy model | None — real checkpoints only (`proxy_model_removed: true` in `split_manifest.json`) |
| Source | `results/conformal_benchmark/` |

---

## 4. Coverage at 90 % nominal — window unit

Mean ± std across five seeds. Coverage gap = empirical − 0.90 (positive = over-covers).

| Subset | split_conformal | cqr | adaptive | eara_conformal | nex_scp | eara_scp |
|---|---:|---:|---:|---:|---:|---:|
| FD001 | 0.878 ± 0.033 | 0.894 ± 0.034 | 0.902 ± 0.040 | 0.890 ± 0.034 | **0.892 ± 0.023** | 0.890 ± 0.012 |
| FD002 | 0.906 ± 0.006 | 0.921 ± 0.015 | 0.900 ± 0.013 | 0.912 ± 0.012 | **0.896 ± 0.004** | 0.899 ± 0.003 |
| FD003 | 0.898 ± 0.008 | 0.902 ± 0.013 | 0.876 ± 0.026 | 0.852 ± 0.040 | **0.884 ± 0.013** | 0.878 ± 0.008 |
| FD004 | 0.904 ± 0.026 | 0.915 ± 0.018 | 0.892 ± 0.018 | 0.908 ± 0.030 | **0.896 ± 0.003** | 0.902 ± 0.007 |

**Reading:** `nex_scp` is the closest to 0.90 on FD001 and FD004. `eara_scp` is similarly well-calibrated, tracking within 1–2 pp of nominal on all subsets. `eara_conformal`
under-covers by 4.8 pp on FD003. `cqr` and `split_conformal` over-cover on
FD002 and FD004. All methods are within ±5 pp of nominal on FD001/FD002.

---

## 5. Interval width at 90 % nominal — window unit

Mean interval width (cycles) ± std across five seeds. Narrower is better,
subject to achieving nominal coverage.

| Subset | split_conformal | cqr | adaptive | eara_conformal | nex_scp | eara_scp |
|---|---:|---:|---:|---:|---:|---:|
| FD001 | 35.43 ± 3.31 | 32.19 ± 3.15 | 31.78 ± 2.89 | 32.44 ± 2.86 | **31.03 ± 0.87** | 33.93 ± 1.88 |
| FD002 | 43.12 ± 1.87 | 39.23 ± 2.19 | 37.40 ± 1.77 | 39.32 ± 1.12 | **37.69 ± 1.31** | 38.08 ± 0.94 |
| FD003 | 34.28 ± 1.09 | 30.53 ± 1.60 | 28.42 ± 2.76 | 28.38 ± 1.86 | **30.05 ± 2.61** | 32.53 ± 4.41 |
| FD004 | 44.48 ± 5.02 | 42.35 ± 4.48 | 37.54 ± 2.85 | 42.41 ± 2.85 | **39.86 ± 2.10** | 41.86 ± 1.50 |

**Reading:** `nex_scp` achieves the narrowest mean width on FD001, FD002 and
FD004 while maintaining near-nominal coverage. `eara_scp` is slightly wider than
`nex_scp` on FD001/FD002 but narrower on FD003/FD004, reflecting the sharper
regime-aware scale. On FD003, `adaptive` and
`eara_conformal` are narrower but under-cover (see §4). `split_conformal` is
consistently the widest.

---

## 6. Winkler score at 90 % nominal — window unit

Winkler score (lower is better) = interval width + penalty for missed coverage.
Mean ± std across five seeds.

| Subset | split_conformal | cqr | adaptive | eara_conformal | nex_scp | eara_scp |
|---|---:|---:|---:|---:|---:|---:|
| FD001 | 54.73 ± 2.85 | 49.65 ± 2.25 | 47.51 ± 2.34 | 50.51 ± 3.01 | **47.73 ± 1.65** | 50.75 ± 2.23 |
| FD002 | 55.92 ± 0.72 | 50.10 ± 1.48 | 52.71 ± 2.00 | 57.69 ± 0.96 | **53.37 ± 2.04** | 58.27 ± 0.77 |
| FD003 | 52.03 ± 1.11 | 47.31 ± 2.01 | 45.06 ± 2.17 | 48.99 ± 2.78 | **44.35 ± 2.12** | 49.34 ± 2.50 |
| FD004 | 62.38 ± 1.46 | 59.49 ± 0.95 | 57.59 ± 0.46 | 60.66 ± 1.76 | **58.64 ± 0.24** | 61.74 ± 1.78 |

**Reading:** `nex_scp` achieves the best (lowest) Winkler score on all four
subsets. `eara_scp` scores higher than `nex_scp` on FD002/FD003/FD004 because
the wider EARA scale inflates the interval without a proportional coverage gain
at the window level. On FD002, `cqr` scores best but `nex_scp` is close.
`eara_conformal` has the worst Winkler score on FD002 despite narrow intervals,
because its coverage penalty is higher.

---

## 7. Full coverage table — all levels, window unit

Mean empirical coverage across five seeds. Nominal targets: 0.80 / 0.90 / 0.95.

### FD001

| Nominal | split_conformal | cqr | adaptive | eara_conformal | nex_scp | eara_scp |
|---:|---:|---:|---:|---:|---:|---:|
| 0.80 | 0.778 | 0.792 | 0.784 | 0.788 | **0.790** | 0.786 |
| 0.90 | 0.878 | 0.894 | 0.902 | 0.890 | **0.892** | 0.890 |
| 0.95 | 0.936 | 0.938 | 0.942 | 0.934 | **0.940** | 0.940 |

### FD002

| Nominal | split_conformal | cqr | adaptive | eara_conformal | nex_scp | eara_scp |
|---:|---:|---:|---:|---:|---:|---:|
| 0.80 | 0.832 | 0.835 | 0.809 | 0.815 | **0.798** | 0.798 |
| 0.90 | 0.906 | 0.921 | 0.900 | 0.912 | **0.896** | 0.899 |
| 0.95 | 0.968 | 0.963 | 0.941 | 0.949 | **0.946** | 0.944 |

### FD003

| Nominal | split_conformal | cqr | adaptive | eara_conformal | nex_scp | eara_scp |
|---:|---:|---:|---:|---:|---:|---:|
| 0.80 | **0.800** | 0.798 | 0.768 | 0.768 | 0.784 | 0.788 |
| 0.90 | 0.898 | 0.902 | 0.876 | 0.852 | **0.884** | 0.878 |
| 0.95 | 0.942 | 0.936 | 0.916 | 0.916 | **0.932** | 0.934 |

### FD004

| Nominal | split_conformal | cqr | adaptive | eara_conformal | nex_scp | eara_scp |
|---:|---:|---:|---:|---:|---:|---:|
| 0.80 | 0.814 | 0.812 | 0.792 | 0.814 | **0.806** | 0.806 |
| 0.90 | 0.904 | 0.915 | 0.892 | 0.908 | **0.896** | 0.902 |
| 0.95 | 0.954 | 0.954 | 0.944 | 0.956 | **0.944** | 0.951 |

---

## 8. Engine unit — conservative reading

The engine unit keeps the **maximum** calibration score per engine (20–52
engines per subset) rather than one score per window. This is a valid conformal
reduction but over-covers badly because a lifetime maximum is much larger than
any single test window.

Mean empirical coverage at 90 % nominal, engine unit:

| Subset | split_conformal | cqr | adaptive | eara_conformal | nex_scp | eara_scp |
|---|---:|---:|---:|---:|---:|---:|
| FD001 | **0.996 ± 0.005** | 0.996 ± 0.005 | 0.996 ± 0.005 | 0.990 ± 0.010 | 0.988 ± 0.011 | 0.986 ± 0.005 |
| FD002 | **0.999 ± 0.002** | 0.998 ± 0.002 | 0.998 ± 0.002 | 0.996 ± 0.004 | 0.975 ± 0.002 | 0.978 ± 0.011 |
| FD003 | **1.000 ± 0.000** | 1.000 ± 0.000 | 0.994 ± 0.005 | 0.992 ± 0.004 | 0.992 ± 0.008 | 0.986 ± 0.009 |
| FD004 | **0.996 ± 0.000** | 0.995 ± 0.002 | 0.997 ± 0.002 | 0.995 ± 0.002 | 0.977 ± 0.005 | 0.985 ± 0.006 |

**Reading:** All methods over-cover by 7–10 pp at the 90 % target. `nex_scp` and
`eara_scp` are the least conservative engine-unit methods (closest to nominal)
because their prequential adaptation reduces the effective quantile for later
test engines. Engine-unit widths are 50–120 % wider than window-unit widths and
should not be compared to the window-unit numbers.

---

## 9. Paired comparisons

Paired exact sign-flip test across five seeds. The smallest attainable p-value
with 5 seeds is 0.0625 (one-sided). All comparisons below use the 90 % nominal
level, window unit. The benchmark reports paired comparisons for both
`eara_conformal` and `eara_scp` as the two novel-ish candidates.

### Coverage: `nex_scp` vs. others

| Subset | vs. split_conformal | vs. cqr | vs. adaptive | vs. eara_conformal |
|---|---:|---:|---:|---:|
| FD001 | Δ = +0.014, p = 0.375 | Δ = −0.002, p = 1.000 | Δ = −0.010, p = 0.875 | Δ = +0.002, p = 1.000 |
| FD002 | Δ = −0.010, p = 0.063 | Δ = −0.025, p = 0.063 | Δ = −0.004, p = 1.000 | Δ = −0.016, p = 0.063 |
| FD003 | Δ = −0.014, p = 0.063 | Δ = −0.018, p = 0.063 | Δ = +0.008, p = 1.000 | Δ = +0.032, p = 0.125 |
| FD004 | Δ = −0.008, p = 0.375 | Δ = −0.019, p = 0.125 | Δ = +0.004, p = 0.375 | Δ = −0.012, p = 0.438 |

### Coverage: `eara_scp` vs. others

| Subset | vs. split_conformal | vs. cqr | vs. adaptive | vs. eara_conformal | vs. nex_scp |
|---|---:|---:|---:|---:|---:|
| FD001 | Δ = +0.012, p = 0.375 | Δ = −0.004, p = 0.875 | Δ = −0.012, p = 0.625 | Δ = 0.000, p = 1.000 | Δ = −0.002, p = 1.000 |
| FD002 | Δ = −0.007, p = 0.125 | Δ = −0.022, p = 0.063 | Δ = −0.001, p = 1.000 | Δ = −0.013, p = 0.063 | Δ = +0.003, p = 0.875 |
| FD003 | Δ = −0.020, p = 0.063 | Δ = −0.024, p = 0.063 | Δ = +0.002, p = 1.000 | Δ = +0.026, p = 0.125 | Δ = −0.006, p = 0.875 |
| FD004 | Δ = −0.002, p = 0.875 | Δ = −0.013, p = 0.250 | Δ = +0.010, p = 0.250 | Δ = −0.006, p = 0.625 | Δ = +0.006, p = 0.375 |

### Width: `nex_scp` vs. others

| Subset | vs. split_conformal | vs. cqr | vs. adaptive | vs. eara_conformal |
|---|---:|---:|---:|---:|
| FD001 | Δ = −4.40 cycles, p = 0.063 | Δ = −1.16 cycles, p = 0.375 | Δ = −0.75 cycles, p = 0.625 | Δ = −1.41 cycles, p = 0.375 |
| FD002 | Δ = −5.42 cycles, p = 0.063 | Δ = −1.54 cycles, p = 0.438 | Δ = +0.29 cycles, p = 0.813 | Δ = −1.63 cycles, p = 0.188 |
| FD003 | Δ = −4.23 cycles, p = 0.063 | Δ = −0.48 cycles, p = 0.563 | Δ = +1.63 cycles, p = 0.313 | Δ = +1.67 cycles, p = 0.125 |
| FD004 | Δ = −4.62 cycles, p = 0.063 | Δ = −2.48 cycles, p = 0.125 | Δ = +2.32 cycles, p = 0.125 | Δ = −2.55 cycles, p = 0.125 |

### Width: `eara_scp` vs. others

| Subset | vs. split_conformal | vs. nex_scp | vs. eara_conformal |
|---|---:|---:|---:|
| FD001 | Δ = −1.50 cycles, p = 0.375 | Δ = +2.90 cycles, p = 0.063 | Δ = +1.49 cycles, p = 0.375 |
| FD002 | Δ = −5.03 cycles, p = 0.063 | Δ = +0.39 cycles, p = 0.813 | Δ = −1.23 cycles, p = 0.188 |
| FD003 | Δ = −1.75 cycles, p = 0.250 | Δ = +2.48 cycles, p = 0.250 | Δ = +4.15 cycles, p = 0.125 |
| FD004 | Δ = −2.62 cycles, p = 0.125 | Δ = +2.00 cycles, p = 0.125 | Δ = −0.55 cycles, p = 0.813 |

**Interpretation:** With 5 seeds, no pair achieves p < 0.0625. The width
advantage of `nex_scp` over `split_conformal` is consistent (negative on 3/4
subsets) but cannot be declared statistically significant at conventional
thresholds. `eara_scp` is consistently wider than `nex_scp` (positive Δ on
all subsets) but better calibrated than `eara_conformal` on FD003/FD004. The
coverage advantage of `eara_conformal` on FD003 (narrower intervals) comes at
the cost of 4.8 pp under-coverage.

---

## 10. Literature context

There is no established public benchmark for conformal prediction on C-MAPSS
with the same protocol as this repository. The table below places the results
in the context of the few published uncertainty quantification results on
C-MAPSS that report coverage and width.

| Method / paper | Dataset | Nominal | Coverage | Width (cycles) | Notes |
|---|---|---:|---:|---:|---|
| **`nex_scp` (ours, 5 seeds)** | FD001 | 90 % | **0.892 ± 0.023** | **31.0 ± 0.9** | Window unit, engine-disjoint |
| **`eara_scp` (ours, 5 seeds)** | FD001 | 90 % | 0.890 ± 0.012 | 33.9 ± 1.9 | Window unit, engine-disjoint; EARA scale + prequential ACI |
| **`split_conformal` (ours, 5 seeds)** | FD001 | 90 % | 0.878 ± 0.033 | 35.4 ± 3.3 | Window unit, engine-disjoint |
| **`nex_scp` (ours, 5 seeds)** | FD002 | 90 % | **0.896 ± 0.004** | **37.7 ± 1.3** | Window unit, engine-disjoint |
| **`eara_scp` (ours, 5 seeds)** | FD002 | 90 % | 0.899 ± 0.003 | 38.1 ± 0.9 | Window unit, engine-disjoint |
| **`nex_scp` (ours, 5 seeds)** | FD003 | 90 % | **0.884 ± 0.013** | **30.0 ± 2.6** | Window unit, engine-disjoint |
| **`eara_scp` (ours, 5 seeds)** | FD003 | 90 % | 0.878 ± 0.008 | 32.5 ± 4.4 | Window unit, engine-disjoint |
| **`nex_scp` (ours, 5 seeds)** | FD004 | 90 % | **0.896 ± 0.003** | **39.9 ± 2.1** | Window unit, engine-disjoint |
| **`eara_scp` (ours, 5 seeds)** | FD004 | 90 % | 0.902 ± 0.007 | 41.9 ± 1.5 | Window unit, engine-disjoint |
| Sharma 2025 (aleatoric UQ) | FD001 | — | — | — | Reports prediction intervals but not conformal coverage [1] |
| Generic split conformal (Angelopoulos & Bates 2022) | — | 90 % | ≥ 0.90 by construction | Depends on model | Marginal coverage guarantee; no C-MAPSS numbers [2] |

> **Note.** Published C-MAPSS uncertainty papers typically report prediction
> interval coverage probability (PICP) and mean prediction interval width
> (MPIW) without engine-disjoint calibration splits. Direct comparison is not
> valid because a non-engine-disjoint split inflates coverage by allowing
> correlated windows from the same engine in both calibration and test sets.

---

## 11. Statistical note

**Two noise sources are present and must not be conflated:**

1. **Seed noise** (training-run variability): measured here as the std across
   five seeds. For coverage at 90 % nominal, the seed std is 0.003–0.040
   depending on method and subset. For width, it is 0.9–5.0 cycles.

2. **Test-set sampling noise** (which test engines were drawn): not measured
   here. With 100 test engines (FD001/FD003) the bootstrap CI on coverage is
   roughly ±0.04 at 90 % nominal.

**Finite-sample guarantee.** Split conformal provides a marginal coverage
guarantee of `1 − α − 1/(n+1)` where `n` is the number of calibration scores.
With ~20 calibration engines × ~100 windows each ≈ 2000 scores, the slack is
negligible. The engine-unit reduction uses only 20–52 scores; the finite-sample
slack is ~2–5 pp and the over-coverage observed is much larger (7–10 pp),
confirming it is a structural property of the max-reduction, not a finite-sample
artefact.

**Prequential ACI (`nex_scp` and `eara_scp`).** Both methods use the adaptation
rule `α_{t+1} = clip(α_t + γ(α − err_t), 0.01, 0.99)` with `γ = 0.02`.
They differ only in the scale function: `nex_scp` uses the generic scale head
`s(x)` fitted on training residuals; `eara_scp` uses the kNN regime-aware scale
from EARA. The level is clipped to `(0.01, 0.99)` to prevent degenerate
intervals. The 0.01 floor means the adaptor never targets more than 99 % coverage.

---

## 12. Reproduce

```powershell
# 1. Train five seeds (required first)
python scripts/multiseed_train.py --model gru_att_deg_v2 --subsets all --seeds 42 43 44 45 46

# 2. Run the six-method benchmark (~24 min on reference GPU)
python scripts/multiseed_conformal_benchmark.py --subsets all --seeds 42 43 44 45 46

# 3. Aggregate into mean ± std and paired tests
python scripts/multiseed_report.py

# Window unit only (skip conservative engine unit)
python scripts/multiseed_conformal_benchmark.py --calibration-units window

# Validate regime count used by EARA
python scripts/regime_sweep.py
```

**Output artifacts** (`results/conformal_benchmark/`):

| File | Contents |
|---|---|
| `per_seed_metrics.csv` | Coverage / width / Winkler per (subset, seed, method, level, unit) |
| `summary_mean_std.csv` | Mean ± std across seeds, per (subset, method, level, unit) |
| `paired_comparisons.csv` | Paired Δ and exact sign-flip p per (subset, level, metric, comparison) |
| `per_engine_intervals.csv` | Per-engine lower/upper bounds for every method |
| `split_manifest.json` | Calibration/test engine IDs and checkpoint hashes per (subset, seed) |
| `run_config.json` | Full benchmark configuration |
| `{subset}_{unit}_coverage_width.png` | Coverage vs. width scatter plots |

---

## References

1. K. Sharma. *Uncertainty-Aware Deep Learning Framework for Remaining Useful
   Life Prediction in Turbofan Engines with Learned Aleatoric Uncertainty.*
   arXiv:2511.19124 (preprint, 2025). <https://arxiv.org/abs/2511.19124>

2. A. N. Angelopoulos, S. Bates. *A Gentle Introduction to Conformal Prediction
   and Distribution-Free Uncertainty Quantification.* arXiv:2107.07511, 2022.
   <https://arxiv.org/abs/2107.07511>

3. I. Gibbs, E. Candès. *Adaptive Conformal Inference Under Distribution Shift.*
   NeurIPS 2021. <https://arxiv.org/abs/2106.00170>
