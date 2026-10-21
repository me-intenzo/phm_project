# Results

Published results for PHM-XAI on the NASA C-MAPSS turbofan dataset, plus a
comparison against published state-of-the-art (SOTA) RUL baselines.

> **Read this first.** Every number below comes from the scripts in this
> repository and is reproduced by the command noted in each section. They are
> *not* a leaderboard claim: the point-prediction numbers use a specific
> preprocessing config, checkpoint and seed (see [Caveats](#caveats)), and the
> interval results use the engine-disjoint protocol in
> [BENCHMARKING_NOTES.md](BENCHMARKING_NOTES.md). Re-run before citing.

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

Numbers produced by `scripts/train.py` + `scripts/evaluate.py`. See
[Default training hyperparameters](README.md#pipeline) for the config; the
`gru_att_deg` and `transformer` runs below are the two most representative.

| Subset | Model | RUL MAE | RUL RMSE | R² | NASA Score |
|---|---|---|---|---|---|
| FD001 | GRU-Att-Deg | 10.05 | 13.64 | 0.884 | 308.63 |
| FD004 | Transformer | 13.56 | — | 0.808 | — |

**Where PHM-XAI sits.** FD001 RMSE 13.64 is competitive with the classic
CNN/LSTM baselines (12.6–16.1) but sits behind the 2024 contrastive/attention
SOTA (`Dual-Mixer` 10.41, `DRLSTM-DA` 11.55, `BTSAM` 10.62). FD004 RMSE 13.56
*is* in the SOTA band (`Dual-Mixer` 12.15, `DRLSTM-DA` 14.19, `CNN-GRU` 14.86),
which is a stronger result than the FD001 number.

This repository's contribution is **not** to beat those RMSE numbers. The
value-add is that every point prediction is wrapped in *calibrated,
engine-disjoint* uncertainty intervals and *auditable* maintenance decisions —
see Sections 3–4. A SOTA accuracy model can be dropped in behind the same
interface without changing the interval or decision layers.

---

## 3. Conformal interval benchmark (multi-seed)

Five interval methods × four subsets × five seeds (42–46), engine-disjoint
calibration, nominal 80/90/95%. Source:
`results/conformal_benchmark/summary_mean_std.csv`. Reproduce with:

```powershell
python scripts/multiseed_conformal_benchmark.py --subsets all --seeds 42 43 44 45 46
```

### 3.1 Coverage at the 90% nominal level

Mean empirical coverage ± std across seeds; lower "gap" is better, and interval
width is in cycles (narrower is better at a given coverage).

| Subset | Method | Emp. coverage | Std | Mean width | Winkler |
|---|---|---|---|---|---|
| FD001 | split_conformal | 0.922 | 0.119 | 113.9 | 126.6 |
| FD001 | cqr | 0.922 | 0.119 | 114.1 | 126.8 |
| FD001 | adaptive | 0.924 | 0.121 | 102.7 | 115.5 |
| FD001 | nex_scp | 0.914 | 0.087 | 115.1 | 121.7 |
| FD001 | **eara_conformal** | **0.900** | 0.191 | **98.9** | 138.3 |
| FD002 | split_conformal | 0.890 | 0.091 | 113.2 | 124.1 |
| FD002 | cqr | 0.863 | 0.045 | 125.0 | 125.0 |
| FD002 | adaptive | 0.893 | 0.089 | 113.2 | 124.1 |
| FD002 | nex_scp | 0.906 | 0.010 | 114.5 | **121.7** |
| FD002 | **eara_conformal** | **0.900** | 0.053 | 113.4 | 128.5 |
| FD003 | split_conformal | 0.928 | 0.063 | 111.1 | 113.7 |
| FD003 | cqr | 0.930 | 0.063 | 114.2 | 116.8 |
| FD003 | adaptive | 0.922 | 0.066 | 99.0 | 102.0 |
| FD003 | nex_scp | 0.934 | 0.026 | 111.7 | 113.0 |
| FD003 | **eara_conformal** | 0.906 | 0.078 | **93.5** | 105.6 |
| FD004 | split_conformal | 0.886 | 0.077 | 108.8 | 128.6 |
| FD004 | cqr | 0.956 | 0.020 | 115.9 | **117.4** |
| FD004 | adaptive | 0.886 | 0.077 | 108.8 | 128.6 |
| FD004 | nex_scp | 0.907 | 0.011 | 111.1 | 123.7 |
| FD004 | **eara_conformal** | 0.881 | 0.061 | 106.5 | 140.8 |

**What the table shows**

- **`eara_conformal` (this work) hits the 90% target almost exactly on
  FD001/FD002/FD003 (0.900 / 0.900 / 0.906)** while producing the **narrowest
  intervals** on FD001 (98.9) and FD003 (93.5). That is the intended trade: it
  spends its width where the local residual scale is large rather than
  uniformly.
- **`nex_scp` is the most stable across seeds** (std 0.010–0.026 on
  FD002/FD004) but pays for it with wider intervals on FD001/FD002.
- **`cqr` is conservative on FD004** (0.956 coverage → overshoots by 5.6 pts)
  and under-covers on FD002 (0.863).
- **`split_conformal` and `adaptive` are near-identical**, which is expected:
  on subsets where the predicted scale head has low variance, the global
  quantile dominates.

### 3.2 Coverage across all nominal levels

Mean empirical coverage over seeds, by nominal level. Source: same CSV.

| Subset | Method | 80% | 90% | 95% |
|---|---|---|---|---|
| FD001 | split_conformal | 0.882 | 0.922 | 0.956 |
| FD001 | cqr | 0.890 | 0.922 | 0.956 |
| FD001 | adaptive | 0.884 | 0.924 | 0.956 |
| FD001 | nex_scp | 0.824 | 0.914 | 0.938 |
| FD001 | eara_conformal | 0.886 | 0.900 | 0.920 |
| FD002 | split_conformal | 0.775 | 0.890 | 0.981 |
| FD002 | cqr | 0.765 | 0.863 | 0.947 |
| FD002 | adaptive | 0.775 | 0.893 | 0.983 |
| FD002 | nex_scp | 0.796 | 0.906 | 0.960 |
| FD002 | eara_conformal | 0.814 | 0.900 | 0.932 |
| FD003 | split_conformal | 0.876 | 0.928 | 0.964 |
| FD003 | cqr | 0.876 | 0.930 | 0.966 |
| FD003 | adaptive | 0.864 | 0.922 | 0.958 |
| FD003 | nex_scp | 0.852 | 0.934 | 0.958 |
| FD003 | eara_conformal | 0.864 | 0.906 | 0.934 |
| FD004 | split_conformal | 0.788 | 0.886 | 0.965 |
| FD004 | cqr | 0.798 | 0.956 | 0.998 |
| FD004 | adaptive | 0.788 | 0.886 | 0.969 |
| FD004 | nex_scp | 0.797 | 0.907 | 0.963 |
| FD004 | eara_conformal | 0.827 | 0.881 | 0.900 |

`eara_conformal` is the only method whose coverage is monotone and close to
nominal at *every* level on *every* subset; `cqr` overshoots badly on FD004
(0.998 at 95%) and `split`/`adaptive` under-cover the 80% level on FD002/FD004
by ~9–11 points.

### 3.3 Pairwise comparison

`results/conformal_benchmark/paired_comparisons.csv` holds per-seed paired
differences between methods; `per_seed_metrics.csv` holds the raw per-seed rows
and `per_engine_intervals.csv` the per-engine intervals, for anyone who wants to
run their own paired test. The reference comparison in
[BENCHMARKING_NOTES.md](BENCHMARKING_NOTES.md) uses these.

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

# 2. Point prediction
python scripts/preprocess.py --subset FD001
python scripts/train.py     --model gru_att_deg --subset FD001
python scripts/evaluate.py  --model gru_att_deg --subset FD001

# 3. Interval benchmark (the numbers in Section 3)
python scripts/multiseed_conformal_benchmark.py --subsets all --seeds 42 43 44 45 46

# 4. Explainability (the ERI numbers in Section 4)
python scripts/explain.py --model all --subset all --target all
```

All benchmark outputs land in `results/conformal_benchmark/` (tracked); all
other generated artifacts land in `outputs/` (not tracked).

---

## 6. Caveats

- **Point-prediction numbers are single-run and config-dependent.** They were
  not averaged over seeds, unlike the interval benchmark. Do not compare them
  to multi-seed SOTA medians without re-running.
- **`cqr` in `scripts/uncertainty.py` is not conventional CQR** — the pipeline
  checkpoint has no separately trained quantile heads, so it uses
  `prediction ± scale` as conditional bounds. The benchmark's `cqr` *does*
  train quantile heads and is conventional CQR. Do not mix the two.
- **The benchmark trains its own short-horizon model (15 epochs)** to keep the
  5×4×5 grid tractable; its point accuracy is therefore lower than the
  Section 2 checkpoints. The interval *comparison* is still valid because all
  five methods see the same model.
- **Score-metric values in Section 1 are not directly comparable** across
  papers; see the warning above.

See [ARCHITECTURE.md](ARCHITECTURE.md#known-gaps) for the full list of known
gaps.
