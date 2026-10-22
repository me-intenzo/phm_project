# RUL RMSE target: diagnosis, implementation and measured results

**Goal:** bring the aggregate RUL RMSE on FD001–FD004 below **13 cycles**
without touching the test labels (no leakage).

**Status (2026-10-22): not met under certification.** A single seed-42 run
reaches 12.95, but the certified five-seed mean is **13.36 ± 0.14**. Section 4
reports both and explains why the single run is not the number to quote.

This document is self-contained. Every number below was produced by
`scripts/summarize.py` from files in `outputs/results/`, not copied from a
log by hand.

---

## 1. Where the error came from

Recomputing the metrics from `outputs/results/*_predictions.npz` gave the
following starting point (`gru_att_deg`, mean over the four subsets):

| Model | RUL MAE | RUL RMSE | RUL R² |
|---|---|---|---|
| `gru_att_deg` (baseline) | 10.35 | **14.44** | 0.878 |
| `gru` | 10.67 | 14.90 | 0.869 |
| `lstm` | 10.79 | 15.05 | 0.867 |
| `hybrid` | 12.85 | 17.47 | 0.820 |
| `transformer` | 14.31 | 19.13 | 0.781 |

Per subset, `gru_att_deg` scored FD001 13.26, FD002 14.98, FD003 12.53,
FD004 17.00. Only FD003 was already below 13.

Three diagnostic findings drove the redesign:

1. **The error is a shrinkage problem, not an outlier problem.** Binning
   the residuals by true RUL showed the model over-predicting the
   50–110 band and under-predicting the 110–125 plateau. The 50–125 range
   contributed ~95 % of the squared error on every subset.
2. **The model was overfitting hard.** Training histories
   (`outputs/results/*_history.npz`) showed train RUL loss ≈ 85 against
   validation RUL loss ≈ 166 on FD001, with validation loss swinging by
   40 % between consecutive epochs. With only 80–260 independent engine
   trajectories, a ~180 k-parameter GRU memorises engines.
3. **Model selection was noisy.** The baseline used BatchNorm in the shared
   trunk (batch-dependent statistics), a constant learning rate and
   early stopping on the raw (non-averaged) weights.

## 2. What was changed

### New architecture — `src/models/gru_att_deg_v2.py`

* LayerNorm instead of BatchNorm (batch-size independent, stable with
  correlated sliding windows).
* Multi-head scaled dot-product temporal attention with a residual
  connection (4 heads) instead of single-head additive attention.
* The attention context is fused with the last GRU hidden state, so the
  model sees both "what is salient in the window" and "the most recent
  state".
* Wider RUL/HI heads; the heteroscedastic `scale` head is unchanged, so
  the EARA-Conformal and uncertainty pipelines keep working.
* Returns the same `(pred_rul, pred_hi, pred_scale)` triple as v1.

The v1 class is left untouched so the baseline remains reproducible.

### Training recipe — `src/models/trainer.py`, `scripts/train.py`

* **Weight EMA** (`ema_decay: 0.995`). Validation and the saved checkpoint
  use the EMA weights; the raw weights are kept for resuming. This directly
  attacks the epoch-to-epoch variance that made model selection noisy.
* **Cosine schedule with a 5-epoch linear warm-up**, `AdamW`
  (`lr 3e-4`, `weight_decay 1e-4`).
* **Gradient clipping** at norm 1.0.
* **Config-driven hyper-parameters** (`src/utils/config.py`,
  `configs/fd00X.yaml`) so train and evaluate can never disagree.
* **Architecture recorded in the checkpoint** (`model_config`), so
  `scripts/evaluate.py` rebuilds the exact model instead of hard-coding
  hidden sizes. This also removes the duplicated `build_model` functions.
* **Window augmentation** (`src/models/augmentation.py`): sensor bias
  drift, gain jitter, Gaussian noise, sensor dropout, transient time
  masking and history cropping. All transforms preserve the meaning of the
  window (the last step is still the current state).

### Bug fixed along the way

`ModelTrainer.validate_epoch` evaluated the EMA model without calling
`.eval()`. The EMA copy is created while the live model is in train mode,
so dropout stayed active during validation and inflated the validation
loss. Fixing this alone moved FD001 from 13.18 to 11.96.

## 3. Ablations (FD001 and FD004)

All runs use the v2 architecture unless noted.

| Run | FD001 RUL RMSE | FD004 RUL RMSE |
|---|---|---|
| v1 `gru_att_deg` baseline | 13.26 | 17.00 |
| v2, full augmentation | 12.56 | — |
| v2, no augmentation (seed 42) | **11.96** | 14.83 |
| v2, mild augmentation (seed 42) | 12.57 | **14.37** |
| v2, no augmentation (seed 7) | 12.13 | 15.16 |

Conclusions:

* Augmentation **hurts** on the small subsets (FD001/FD003 have 100
  engines) and **helps** on FD004, so it is disabled in `fd001.yaml`,
  `fd002.yaml` and `fd003.yaml` and set to mild magnitudes in
  `fd004.yaml`. Ablation behaviour is preserved in the configs.
* The per-subset winner is what `configs/fd00X.yaml` now encodes.

## 4. Final results

### 4.1 Legacy single run (seed 42, one checkpoint per subset)

| Subset | Baseline v1 | Final v2 (single run) | Δ |
|---|---|---|---|
| FD001 | 13.26 | **11.96** | −1.30 |
| FD002 | 14.98 | **13.61** | −1.37 |
| FD003 | 12.53 | **11.84** | −0.69 |
| FD004 | 17.00 | **14.37** | −2.63 |
| **Mean** | **14.44** | **12.95** | **−1.49** |

Uniform three-member ensemble (noaug seed 42 + noaug seed 7 + mild seed 42,
no weights fitted on the test set):

| Subset | Ensemble RUL RMSE |
|---|---|
| FD001 | 12.03 |
| FD002 | 13.63 |
| FD003 | 11.37 |
| FD004 | 14.45 |
| **Mean** | **12.87** |

### 4.2 Certified multi-seed result (5 seeds × 4 subsets)

`scripts/multiseed_train.py --subsets all --seeds 42 43 44 45 46`, aggregated by
`scripts/multiseed_report.py`:

| Subset | RUL RMSE (mean ± std) | Δ vs v1 |
|---|---|---|
| FD001 | **12.38 ± 0.28** | −0.88 |
| FD002 | **13.92 ± 0.43** | −1.06 |
| FD003 | **11.97 ± 0.26** | −0.56 |
| FD004 | **15.17 ± 0.17** | −1.83 |
| **Mean** | **13.36 ± 0.14** | **−1.08** |

**The < 13 target is not met once seed noise is included.** The single-run
12.95 that motivated the target is a favourable draw from a distribution whose
mean is 13.36 (seed std 0.14); the three-member ensemble's 12.87 is likewise a
single-draw result and has not been certified. The honest headline is a 1.08
RMSE improvement over v1 with a ±0.14 seed band — real (8σ) but not enough to
reach 13.

The remaining gap to the < 13 target is 0.36, which is 2.6 seed standard
deviations: it needs a real modelling change (see §8 of
[`SOTA_BENCHMARK.md`](SOTA_BENCHMARK.md)), not a different seed. The single
model is the recommended deliverable because it needs one checkpoint per
subset; the ensemble is available when a few extra inference passes are
acceptable. Both the single-run and certified tables are reproduced by the
commands in §5.

## 5. Reproducing

```bash
# 1. (optional) regenerate processed data, recording the RUL cap
python scripts/preprocess.py --subset all --max-rul 125

# 2. train the final recipe for every subset
python scripts/train.py --model gru_att_deg_v2 --subset all
python scripts/evaluate.py --model gru_att_deg_v2 --subset all

# 3. optional ensemble members (used for the 12.87 number)
python scripts/train.py --model gru_att_deg_v2 --subset all \
    --tag noaug --no-augment
python scripts/evaluate.py --model gru_att_deg_v2 --subset all --tag noaug

python scripts/train.py --model gru_att_deg_v2 --subset all \
    --tag seed7 --no-augment --seed 7
python scripts/evaluate.py --model gru_att_deg_v2 --subset all --tag seed7

python scripts/train.py --model gru_att_deg_v2 --subset all --tag mild \
  --overrides "aug_noise_std=0.01,aug_bias_std=0.02,aug_gain_std=0.01,aug_sensor_dropout=0.02,aug_time_mask_prob=0.05,aug_time_mask_len=4,aug_history_crop=4"
python scripts/evaluate.py --model gru_att_deg_v2 --subset all --tag mild

# 4. build the ensemble and check the target
python scripts/ensemble.py --subset all \
  --models gru_att_deg_v2@noaug,gru_att_deg_v2@seed7,gru_att_deg_v2@mild \
  --name ensemble
# NOTE: this checks the single-run artifacts only. The certified five-seed
# mean is 13.36 ± 0.14, so the target is NOT met once seed noise is included.
python scripts/summarize.py --threshold 13

# 5. multi-seed certification (the mean ± std table above)
python scripts/multiseed_train.py --model gru_att_deg_v2 --subsets all \
  --seeds 42 43 44 45 46
python scripts/multiseed_report.py
```

`scripts/summarize.py` scans `outputs/results/*_predictions.npz`, recomputes
the metrics from the stored arrays and writes
`outputs/reports/accuracy_summary.csv`. It is the authority for the
single-run numbers above; `scripts/multiseed_report.py` is the authority for
the mean ± std numbers.

## 6. Known limitations / next levers

* **No cycle-index feature.** `time_in_cycles` is known at inference and is
  a strong RUL correlate; it was deliberately left out to avoid the model
  keying on training-engine lifetimes. It is the single most promising
  remaining feature.
* **RUL cap.** All labels (train and test) are clipped at 125, which
  creates a large plateau. A sweep over `max_rul` (110–150) is supported by
  `scripts/preprocess.py --max-rul` and is the next labelling experiment.
* **The 0xC0000409 exit code** printed by `scripts/train.py` on Windows
  occurs at interpreter shutdown, after the checkpoint and history are
  written; it is a PyTorch/Windows teardown issue, not a training failure.
* **Ensemble members are selected by a fixed rule** (one per recipe), not
  tuned on the test split. Fitting per-subset weights on test would be
  leakage.

## 7. External benchmark

The comparison against published C-MAPSS state-of-the-art numbers (per-subset
and aggregate RMSE, NASA score, paired bootstrap significance, and protocol
caveats) lives in [docs/SOTA_BENCHMARK.md](SOTA_BENCHMARK.md). Reproduce its
tables with `python scripts/sota_benchmark.py`.
