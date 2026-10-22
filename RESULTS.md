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

- [1. Point prediction — published baselines](#1-point-prediction--published-baselines)
- [2. Point prediction — this repository](#2-point-prediction--this-repository)
- [3. Conformal interval benchmark (multi-seed)](#3-conformal-interval-benchmark-multi-seed)
- [4. Explainability](#4-explainability)
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
| FD001 | 9.07 ± 0.24 | **12.38 ± 0.28** | 0.904 | 237 ± 10 | 11.94 – 12.62 |
| FD002 | 10.13 ± 0.35 | **13.92 ± 0.43** | 0.895 | 899 ± 75 | 13.25 – 14.32 |
| FD003 | 8.19 ± 0.22 | **11.97 ± 0.26** | 0.907 | 272 ± 25 | 11.55 – 12.23 |
| FD004 | 11.08 ± 0.29 | **15.17 ± 0.17** | 0.875 | 1119 ± 57 | 14.94 – 15.39 |
| **FD001–FD004 mean** | **9.62** | **13.361 ± 0.136** | **0.895** | **632** | 13.21 – 13.51 |

The four-subset mean per seed is 13.213 / 13.233 / 13.373 / 13.480 / 13.507 for
seeds 42–46; 13.361 ± 0.136 is the mean ± sample std of those five aggregate
values. Bootstrap CIs that treat the *test engines* as the sampling unit are in
[docs/SOTA_BENCHMARK.md](docs/SOTA_BENCHMARK.md) § 3; they are ~±1 RMSE and
measure test-sampling noise, not seed noise. The two must not be conflated.

**What this replaces.** Earlier revisions of this file quoted
12.947 (FD001–FD004 mean) from a single seed-42 run, and 12.871 from a
three-member ensemble. Both are *one draw* from the distributions above. The
current trainer consumes its RNG stream differently (seeded DataLoader worker
and generator), so re-running "seed 42" today does not reproduce the historical
artifact bit-for-bit: the certified seed-42 run scores 13.213 on the aggregate,
0.27 above the superseded single-run number. The mean ± std table is the number
to cite; the single-run and ensemble values are kept in
[docs/SOTA_BENCHMARK.md](docs/SOTA_BENCHMARK.md) § 3 and labelled as such.

**Where PHM-XAI sits** (mean vs published single runs; a difference inside the
seed band is *not* a ranking):

- **FD001 12.38** — ahead of the classic CNN (12.61) and LSTM (16.14), behind
  every 2024 model in the table (`Dual-Mixer` 10.41, `BTSAM` 10.62,
  `LSTM-F` 10.98, `DRLSTM-DA` 11.55, `CNN-GRU` 11.66).
- **FD002 13.92** — ahead of `CNN-GRU` (14.93), `BTSAM` (14.18) and
  `LSTM-F` (14.60); behind `DRLSTM-DA` (13.74) and `Dual-Mixer` (13.90) by
  less than the seed std (0.43), i.e. not separable.
- **FD003 11.97** — behind `DRLSTM-DA` (11.25) and the 9.85 / 8.87 band.
- **FD004 15.17** — behind `DRLSTM-DA` (14.19), `BTSAM` (14.29),
  `CNN-GRU` (14.86) and `LSTM-F` (15.00); ahead of `DAST` (16.89) and
  `TTSNet` (18.26) on this subset.
- **Four-subset mean 13.361** — 0.53 behind the best published aggregate
  (`KESTF` 12.83), which is larger than the seed std (0.136), so this is a
  genuine deficit rather than seed noise. It is statistically
  indistinguishable from `TTSNet` (13.40): the 0.04 gap is inside the seed
  band. It is ahead of `Cau-BiMamba-LSTM` (13.62), `MLEAN` (13.71), `DAST`
  (14.09), `ATCN` (14.11), `DBA-DCRN` (14.28) and `CNN-LSTM-Att` (15.25).

This repository's contribution is **not** to beat those RMSE numbers. The
value-add is that every point prediction is wrapped in *calibrated,
engine-disjoint* uncertainty intervals and *auditable* maintenance decisions —
see Sections 3–4. A SOTA accuracy model can be dropped in behind the same
interface without changing the interval or decision layers.

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
| FD001 | split_conformal | 0.904 | 0.023 | 38.2 | 53.0 |
| FD001 | cqr | 0.908 | 0.019 | 34.4 | 49.1 |
| FD001 | adaptive | 0.908 | 0.024 | 34.1 | **47.1** |
| FD001 | nex_scp | 0.904 | 0.017 | **32.6** | 47.9 |
| FD001 | eara_conformal | 0.914 | 0.022 | 35.5 | 48.9 |
| FD002 | split_conformal | 0.910 | 0.007 | 43.8 | 56.1 |
| FD002 | cqr | 0.930 | 0.007 | 40.0 | **50.7** |
| FD002 | adaptive | 0.907 | 0.017 | 38.4 | 50.9 |
| FD002 | nex_scp | 0.897 | 0.003 | **37.3** | 51.4 |
| FD002 | eara_conformal | 0.917 | 0.010 | 40.2 | 54.7 |
| FD003 | split_conformal | 0.902 | 0.013 | 35.1 | 53.7 |
| FD003 | cqr | 0.894 | 0.018 | 31.2 | 50.5 |
| FD003 | adaptive | 0.884 | 0.029 | 29.2 | **46.8** |
| FD003 | nex_scp | 0.888 | 0.008 | 30.6 | 46.9 |
| FD003 | eara_conformal | 0.852 | 0.035 | **29.1** | 52.4 |
| FD004 | split_conformal | 0.896 | 0.032 | 44.1 | 63.7 |
| FD004 | cqr | 0.903 | 0.034 | 43.2 | 61.0 |
| FD004 | adaptive | 0.869 | 0.029 | **36.8** | 58.1 |
| FD004 | nex_scp | 0.894 | 0.006 | 41.1 | **57.9** |
| FD004 | eara_conformal | 0.902 | 0.023 | 40.3 | 59.9 |

**What the table shows**

- **Every method lands within ±5 points of the 90 % target.** That is the
  headline: the intervals are calibrated in practice, not only on paper, and the
  spread across seeds is 0.003–0.035.
- **`nex_scp` has the best coverage/width trade-off overall.** Its coverage is
  the most stable (std 0.003–0.017), it is the narrowest interval on FD001 and
  FD002, and it has the best (FD004) or second-best (FD001, FD003) Winkler
  score. Nothing in the set dominates it.
- **`split_conformal` is the widest on every subset** (35.1–44.1 cycles), which
  is expected: it spends the same radius on every window instead of scaling with
  the local difficulty.
- **`adaptive` is the narrowest on FD004 but under-covers there (0.869,
  −3.1 pts).** On FD004 its narrow interval is buying width with coverage; on
  FD003 the same trade shows (0.884).
- **`eara_conformal` is *not* the best-calibrated method.** On FD003 it lands at
  0.852 (−4.8 pts) while being the narrowest (29.1); on FD004 it is well
  calibrated at 0.902. Its scale is a k-NN average over training-engine
  profiles, which under-states the uncertainty of a last-window test query on
  some subsets. The earlier single-run claim that EARA "hits the 90 % target
  almost exactly" does not survive five seeds.
- **`cqr` over-covers on FD002 (0.930)** and is competitive elsewhere now that
  its quantile head is anchored to the point prediction. Before that fix its
  intervals were 3× too wide (104 vs 35 cycles at 90 % on FD003).

### 3.2 Coverage across all nominal levels (window unit)

Mean empirical coverage over seeds. A well-calibrated method lies on the
diagonal; the nominal values are 0.80 / 0.90 / 0.95.

| Subset | Method | 80% | 90% | 95% |
|---|---|---|---|---|
| FD001 | split_conformal | 0.798 | 0.904 | 0.946 |
| FD001 | cqr | 0.804 | 0.908 | 0.946 |
| FD001 | adaptive | 0.834 | 0.908 | 0.952 |
| FD001 | nex_scp | 0.812 | 0.904 | 0.938 |
| FD001 | eara_conformal | 0.832 | 0.914 | 0.952 |
| FD002 | split_conformal | 0.829 | 0.910 | 0.966 |
| FD002 | cqr | 0.842 | 0.930 | 0.961 |
| FD002 | adaptive | 0.818 | 0.907 | 0.952 |
| FD002 | nex_scp | 0.798 | 0.897 | 0.950 |
| FD002 | eara_conformal | 0.828 | 0.917 | 0.956 |
| FD003 | split_conformal | 0.802 | 0.902 | 0.936 |
| FD003 | cqr | 0.790 | 0.894 | 0.936 |
| FD003 | adaptive | 0.802 | 0.884 | 0.924 |
| FD003 | nex_scp | 0.790 | 0.888 | 0.942 |
| FD003 | eara_conformal | 0.766 | 0.852 | 0.922 |
| FD004 | split_conformal | 0.815 | 0.896 | 0.952 |
| FD004 | cqr | 0.818 | 0.903 | 0.947 |
| FD004 | adaptive | 0.767 | 0.869 | 0.932 |
| FD004 | nex_scp | 0.797 | 0.894 | 0.944 |
| FD004 | eara_conformal | 0.796 | 0.902 | 0.951 |

- **`split_conformal` and `nex_scp` are the two closest to nominal at every
  level on every subset** (worst deviation: split +2.9 on FD002@80, nex −1.2 on
  FD001@95). `nex_scp` buys that with a slightly narrower interval.
- **`adaptive` and `eara_conformal` systematically under-cover at 95 %**
  (FD003: 0.924 / 0.922; FD004: 0.932 / 0.951). Their local scale estimates are
  the weak link, not the conformal quantile.
- **`cqr` is the only method that is never more than 4.2 points off nominal.**

### 3.3 The engine unit (conservative reading)

Same checkpoints, same test windows, calibration reduced to one **maximum** score
per calibration engine.

| Subset | Method | Emp. coverage | Mean width | Winkler |
|---|---|---|---|---|
| FD001 | split_conformal | 1.000 | 66.7 | 66.7 |
| FD001 | cqr | 1.000 | 65.0 | 65.0 |
| FD001 | adaptive | 0.992 | 74.3 | 75.4 |
| FD001 | nex_scp | 0.980 | 61.8 | 65.3 |
| FD001 | eara_conformal | 0.990 | 84.4 | 86.2 |
| FD002 | split_conformal | 1.000 | 76.1 | 76.1 |
| FD002 | cqr | 0.998 | 72.4 | 72.4 |
| FD002 | adaptive | 0.998 | 89.8 | 90.0 |
| FD002 | nex_scp | 0.985 | 66.4 | 68.2 |
| FD002 | eara_conformal | 0.999 | 81.2 | 81.2 |
| FD003 | split_conformal | 0.998 | 70.3 | 70.3 |
| FD003 | cqr | 0.998 | 67.2 | 67.2 |
| FD003 | adaptive | 0.984 | 63.6 | 65.1 |
| FD003 | nex_scp | 0.982 | 60.2 | 62.2 |
| FD003 | eara_conformal | 0.994 | 86.1 | 86.5 |
| FD004 | split_conformal | 0.995 | 79.1 | 79.8 |
| FD004 | cqr | 0.995 | 78.7 | 79.9 |
| FD004 | adaptive | 0.991 | 87.2 | 89.3 |
| FD004 | nex_scp | 0.977 | 65.5 | 71.4 |
| FD004 | eara_conformal | 0.998 | 96.5 | 96.7 |

The engine unit sits **7.6–14.2 points above** the window unit at the same
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
scores 0.5. Source: `scripts/explain.py` outputs.

| Target | Mean ERI (across subsets and models) |
|---|---|
| HI | 0.880 |
| RUL | 0.872 |

These are agreement scores, not faithfulness scores — see
[Implementation Status](README.md#implementation-status) (`faithfulness
validation pending`).

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
python scripts/multiseed_pipeline.py --stages all --seeds 42 43 44 45 46

# 6. Explainability (the ERI numbers in Section 4)
python scripts/explain.py --model all --subset all --target all
```

Every multi-seed stage is resumable: re-running a command skips the
(subset, seed) combinations whose checkpoint and metrics already exist.
`--force` re-runs them.

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
- **Score-metric values in Section 1 are not directly comparable** across
  papers; see the warning above.

See [docs/ARCHITECTURE.md § 7](docs/ARCHITECTURE.md#7-known-gaps) for the full
list of known gaps.
