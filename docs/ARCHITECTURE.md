# Architecture

Internal design of PHM-XAI: the pipeline, the module boundaries, the artifact
contracts, the conformal calibration math, the decision policy, and the gaps
that a reader must know about before quoting any number in this repository.

**Audience.** Someone who has to *change* this codebase — add an architecture,
a dataset, an interval method, or a metric — and needs to know which file owns
which responsibility, what the boundaries are, and which parts are known to be
unsound.

**How to read it.** Sections 1–3 are the map (pipeline → modules → config).
Sections 4–6 are the algorithms. Section 7 is the honest list of what does not
work yet. Sections 8–12 are the contracts, the extension recipes, and the test
strategy. If you only want the sharp edges, read §7.

**Companion documents.**

| Document | Covers |
|---|---|
| [README.md](../README.md) | Install, run, and what the project claims |
| [RESULTS.md](../RESULTS.md) | Published numbers and SOTA comparison |
| [docs/SOTA_BENCHMARK.md](SOTA_BENCHMARK.md) | Authoritative RMSE benchmark, bootstrap CIs, significance tests |
| [docs/RMSE_TARGET_PLAN.md](RMSE_TARGET_PLAN.md) | Why each hyper-parameter is what it is |

---

## 1. Pipeline stages

### 1.1 Stage graph

```text
raw C-MAPSS text  (data/raw/train_FD00X.txt, test_FD00X.txt, RUL_FD00X.txt)
      │
      ▼  scripts/preprocess.py
data/processed/<subset>_{train,test}_{X,y_rul,y_hi,engine_ids}.npy
      +  <subset>_{train,test}_metadata.json
      │
      ▼  scripts/train.py
outputs/checkpoints/<arch>/<subset>/best_model.pt   (stores model_config)
      │
      ├──────────────► scripts/evaluate.py   → outputs/results/<subset>_<model>_predictions.npz
      │                                          outputs/results/<subset>_<model>_metrics.json
      │
      ├──────────────► scripts/uncertainty.py → outputs/results/<subset>_<model>_<variant>.npz
      │
      ├──────────────► scripts/explain.py     → outputs/xai/<subset>/<model>/xai_<target>.npz
      │                                          outputs/xai/<subset>/<model>/eri_<target>.json
      │                                          outputs/xai/<subset>/<model>/xai_report_<subset>_<target>.html
      │
      ▼  scripts/decision.py
outputs/results/decision_<stem>.json
outputs/reports/decision_<stem>.html
      │
      ▼  scripts/hitl.py
outputs/hitl/feedback.jsonl  (+ HTML report, review log)
```

Three scripts sit **beside** the chain rather than inside it. They read the
artifacts the chain produces and never feed back into it:

```text
outputs/results/*_predictions.npz
      ├──────────────► scripts/summarize.py       → outputs/reports/accuracy_summary.csv
      ├──────────────► scripts/sota_benchmark.py  → outputs/reports/sota_*.csv
      └──────────────► scripts/ensemble.py        → outputs/results/<subset>_<name>_predictions.npz
```

### 1.2 Entry-point reference

Every stage is a standalone CLI. `--subset all` and `--model all` iterate over
every combination; a stage that fails for one combination is reported and
skipped rather than aborting the whole run.

| Entry point | Purpose | Key flags | Reads | Writes |
|---|---|---|---|---|
| `main.py` | Exploratory data analysis (FD001 only) | — | `data/raw/` | `outputs/figures/`, `outputs/reports/` |
| `scripts/preprocess.py` | Raw text → windows, labels, scaler | `--subset`, `--config`, `--window-size`, `--max-rul` | `data/raw/` | `data/processed/`, `outputs/models/`, `outputs/figures/` |
| `scripts/train.py` | Train one (model, subset) pair | `--model`, `--subset`, `--config`, `--seed`, `--tag`, `--epochs`, `--lr`, `--overrides`, `--no-augment`, `--no-ema`, `--no-deterministic` | `data/processed/` | `outputs/checkpoints/<arch>[_tag]/<subset>/best_model.pt`, `outputs/results/<subset>_<arch>[_tag]_history.npz` |
| `scripts/evaluate.py` | Point-prediction metrics for a checkpoint | `--model`, `--subset`, `--tag` | checkpoint, `data/processed/` | `outputs/results/<subset>_<model>[_tag]_predictions.npz`, `..._metrics.json` |
| `scripts/uncertainty.py` | Conformal prediction intervals | `--model`, `--subset`, `--method`, `--tag` | checkpoint, `data/processed/` | `outputs/results/<subset>_<model>_<variant>.npz` (tag-aware — see §7.13) |
| `scripts/explain.py` | IG + SHAP + temporal relevance + ERI | `--subset`, `--model`, `--target`, `--samples`, `--ig-steps`, `--shap-nsamples`, `--engine-id` | checkpoint, `data/processed/` | `outputs/xai/<subset>/<model>/` |
| `scripts/decision.py` | Rules + constraints → recommendation | `--subset`, `--model`, `--uncertainty-variant`, `--xai-context`, constraint flags | predictions, intervals, `eri_rul.json`, `data/external/decision_input*.json` | `outputs/results/decision_<stem>.json`, `outputs/reports/decision_<stem>.html` |
| `scripts/hitl.py` | Review queue, resolution, policy refinement | `--subset`, `--model`, `--input`, `--review-id`, `--reviewer`, `--action`, `--override-action`, `--confidence`, `--reason` | decision JSON | `outputs/hitl/feedback.jsonl`, HTML report, review log |
| `scripts/demo.py` | Runs the whole chain in one command | `--subset`, `--model`, `--eda`, `--skip-<stage>`, constraint flags | — | everything above |
| `scripts/multiseed_train.py` | Trains the deliverable architecture across a seed grid and records an auditable manifest | `--model`, `--subsets`, `--seeds`, `--tag-prefix`, `--overrides`, `--force`, `--allow-single-seed` | `data/processed/` | `outputs/checkpoints/<arch>_ms<seed>/<subset>/`, `results/multiseed/training_manifest.json` |
| `scripts/ensemble.py` | Uniform mean over several checkpoints | `--subset`, `--models`, `--name`, `--device` | checkpoints | `outputs/results/<subset>_<name>_predictions.npz`, `..._metrics.json` |
| `scripts/summarize.py` | Scans prediction files → accuracy table | `--models`, `--threshold` | `outputs/results/*_predictions.npz` | `outputs/reports/accuracy_summary.csv` |
| `scripts/sota_benchmark.py` | RMSE benchmark, bootstrap CI, significance | `--models`, `--reference`, `--resamples`, `--seed` | `outputs/results/*_predictions.npz` | `outputs/reports/sota_*.csv` |
| `scripts/multiseed_conformal_benchmark.py` | 5 methods × 4 subsets × 5 seeds | `--subsets`, `--seeds`, `--epochs`, `--batch-size`, `--n-regimes`, `--aci-gamma` | `data/processed/` | `results/conformal_benchmark/` (tracked) |
| `scripts/decision_sensitivity.py` | Risk-score weight sensitivity + constraint-scenario comparison | `--subsets`, `--seeds`, `--model`, `--dirichlet-samples` | `outputs/results/decision_*.json` | `outputs/reports/decision_sensitivity.*`, `decision_weight_sweep.csv`, `decision_constraint_scenarios.csv` |
| `scripts/regime_sweep.py` | Validates the k-means regime count | — | `data/processed/` | `outputs/results/regime_sweep.json` |

`scripts/demo.py` only wires the four architectures that the decision engine
consumes (`lstm`, `gru`, `transformer`, `hybrid`). `gru_att_deg` and
`gru_att_deg_v2` — including the default architecture in `configs/*.yaml` —
are not part of the demo chain.

`scripts/decision.py` has **no** `--input` flag: it resolves
`data/external/decision_input_<subset>.json` and falls back to
`data/external/decision_input.json`. Only `scripts/hitl.py` takes an explicit
`--input` path.

### 1.3 The invariants

Four properties hold at every stage and are asserted before a run continues.

1. **Engine-disjoint splits.** Windows from one engine are strongly
   correlated, so train/validation and calibration/test splits are grouped by
   engine ID. No engine appears in two splits.
2. **Calibration never sees test targets.** Conformal quantiles are estimated
   on a calibration split carved out of the *training* set; the test set is
   used only to measure coverage.
3. **The checkpoint owns the architecture.** `scripts/evaluate.py`,
   `scripts/uncertainty.py` and `scripts/explain.py` rebuild the model from the
   `model_config` stored inside the checkpoint, never from the YAML. A model
   can therefore never be evaluated with a configuration it was not trained
   with.
4. **Constraints escalate, never de-escalate.** The constraint layer may make
   an action more conservative; it can never make it less so. The single
   documented exception is the maintenance-window rule, which *substitutes*
   an infeasible commitment (SCHEDULE_MAINTENANCE) with the nearest feasible
   one (INSPECT) and always flags the record for human review — see §5.4.

---

## 2. Module responsibilities

### 2.1 Layering and dependency rules

The package is layered. Dependencies point **downwards only**.

```text
scripts/            CLI entry points; may import anything in src/
  └── src/evaluation, src/hitl, src/decision_engine, src/explainability,
      src/uncertainty, src/models   (higher-level layers)
        └── src/preprocessing, src/models, src/utils   (lower-level layers)
```

Rules that the code actually follows:

| Rule | Why |
|---|---|
| `src/preprocessing`, `src/models`, `src/utils` do not import from the upper layers. | A model must be trainable without the decision engine existing. |
| `src/decision_engine` imports from `src/utils.config` only. | The policy is pure: state in, action out, no torch, no numpy arrays of features. |
| `src/uncertainty` and `src/explainability` depend on a *checkpoint*, not on `src/models.registry`. | They rebuild the architecture from the checkpoint's own `model_config`. |
| `src/evaluation` may import from `src/decision_engine`. | The ablation re-runs the decision engine; that is the whole point. |
| `src/hitl` imports from `src/decision_engine` only. | Feedback refines the policy, never the model. |
| Cross-layer communication happens through **files on disk**, not Python objects. | Each stage is independently runnable and independently testable. |

### 2.2 `src/preprocessing/`

| Module | Responsibility |
|---|---|
| `loader.py` | Reads the raw `train_FD00X.txt` / `test_FD00X.txt` / `RUL_FD00X.txt` files into tidy DataFrames. |
| `validator.py` | Schema and value-range checks (monotonic `time_in_cycles`, no gaps, sane sensor values). Writes a validation report. |
| `labeling.py` | Piecewise-linear RUL label and the *legacy* HI label (`HI = RUL / max_rul`, kept for comparison only — see §4.5). |
| `health_index.py` | The sensor-derived health index (SDHI) and the calibration of the decision policy's HI breakpoints. |
| `feature_selection.py` | Variance / correlation / mutual-information selection down to the 14 operating sensors (more on the multi-condition subsets). |
| `scaling.py` | Per-engine z-score normalisation; the fitted scaler is persisted to `outputs/models/`. |
| `windowing.py` | Sliding-window construction, engine-wise train/test split, `.npy` persistence, metadata sidecar. |
| `visualization.py` | EDA and per-subset diagnostic plots. |

### 2.2a The sensor-derived health index

The health index used to be `HI = RUL / max_rul` — a deterministic affine
function of the RUL label, so the two correlated at `1.000000` and the HI
head had nothing to learn that the RUL head had not already learned
(§7.2).  The pipeline now builds the index from the **sensor signal**
instead (`src/preprocessing/health_index.py`):

1. **Condition removal.**  Each sensor is regressed on the three
   operational settings (least squares, fitted on the *training* split
   only) and the residual is kept.  On FD002/FD004 this step is what makes
   the index usable at all: without it the dominant variance direction of
   the sensors is the operating condition rather than the degradation.
2. **Per-engine baseline.**  Each engine's first `hi_baseline_cycles`
   (30) cycles define its own healthy reference.  Subtracting it removes
   the engine-to-engine offset without ever looking at the engine's end of
   life.
3. **Degradation direction.**  A one-component **PLS** direction is fitted
   on the training windows against the training RUL label.  PLS is used
   instead of PCA because the wanted direction is the one that tracks
   degradation, not the one that explains the most variance.  On FD002 the
   two differ sharply: PCA-1 gives `corr(HI, RUL) = 0.04` (the condition
   direction), PLS-1 gives `0.69`.
4. **Projection and mapping.**  The projection of a window's centred
   feature vector is mapped onto `[0, 1]` with reference quantiles taken
   from the training windows:
   `HI = 1 - clip((d - lo) / (hi - lo), 0, 1)`.

**Window alignment.**  The index of a row is computed from the *same
trailing window the model sees* (left-padded with the engine's first cycle
when the engine is shorter than `window_size`, matching
`WindowGenerator.create_test_windows`).  The label attached to a training
window is therefore a function of exactly that window's features.

**Breakpoint calibration.**  The decision policy decides a health state
from *either* the RUL or the HI criterion.  With the legacy label the two
were consistent by construction; with the sensor-derived index they are
not, because the index has its own scale.  Preprocessing therefore
calibrates the three HI breakpoints on the training split — each is the
median index of the training windows in the corresponding RUL band — and
stores them in the dataset metadata as `hi_thresholds`.
`scripts/decision.py` merges them into the policy thresholds.

The legacy label is still reachable (`--hi-mode rul`) so the two can be
compared; it is not the default any more.

### 2.3 `src/models/`

| Module | Responsibility |
|---|---|
| `lstm.py`, `gru.py`, `transformer.py`, `hybrid.py`, `gru_att_deg.py`, `gru_att_deg_v2.py` | Architectures. Each returns the triple `(pred_rul, pred_hi, pred_scale)`. |
| `registry.py` | **Single source of truth** for building *and loading* a model: `ARCHS`, `ALL_MODELS`, `build_model()`, `model_config()`, `resolve_checkpoint_path()`, `load_model_from_checkpoint()`. The scripts no longer carry their own `load_model`. |
| `heads.py` | The shared multi-task heads (`rul_head` / `hi_head` / `scale_head`). The architectures import them, so the head layout is defined once; the attribute names are unchanged, so existing checkpoints still load. |
| `layers.py` | The shared sinusoidal `PositionalEncoding` (used by the Transformer and the Hybrid). |
| `outputs.py` | `unpack_model_output()` — the single place that knows how to read a model output. |
| `losses.py` | Multi-task loss (RUL + HI + heteroscedastic scale). |
| `trainer.py` | Training loop: warm-up, cosine/plateau LR, gradient clipping, weight EMA, early stopping. Stores run provenance in the checkpoint. The train and the validation pass share one `_run_epoch()` implementation. |
| `checkpoint.py` | Save / load, including `load_state_dict_compat()` for tolerant loading. |
| `uncertainty_heads.py` | `QuantileHead` and `ScaleHead` used by the CQR and adaptive-scale paths. |
| `augmentation.py` | Sensor-physical window augmentation (noise, bias, gain, sensor dropout, time mask, history crop). |

### 2.4 `src/uncertainty/`

| Module | Responsibility |
|---|---|
| `conformal.py` | Split-conformal baseline, regime detection, EARA-Conformal, the named interval variants **and the calibration entry points** (`calibrate_eara_levels`, `calibrate_multiple_levels`). It is the single source of truth for the finite-sample quantile (`conformal_quantile()` / `signed_conformal_quantile()`), the shared scale floor (`SCALE_FLOOR`, applied as a bound with `max`, never as an addition) and the interval builder (`clip_interval()`). An unattainable rank yields `+inf`, not a clamped value (see §7.1). |
| `pipeline.py` | The model-agnostic interval engine shared by the CLI and the certification benchmark: window/engine calibration units, scale and quantile heads, EARA engine profiles, prequential ACI, and the interval metrics (§7.14). |
| `coverage.py` | Empirical coverage, coverage error, MPIW, PINAW, Winkler score. Every public metric routes through the single `_validate_interval_inputs` validator. The evaluation layer calls these functions instead of re-implementing them. |

There is deliberately **no** `calibration.py` or `adaptive.py`: both were
pass-through modules whose logic now lives in `conformal.py`, which is the
only place that declares a scale floor, a conformal quantile or an interval
builder.

### 2.5 `src/explainability/`

| Module | Responsibility |
|---|---|
| `integrated_gradients.py` | Path-integrated gradients (Sundararajan et al.), the temporal gradient relevance, and `explain_attention`. All three share one eval-guarded gradient path (`_eval_guard`), so dropout and batch-norm are never active during an explanation. |
| `shap_explainer.py` | SHAP sensor importances. |
| `eri.py` | Explainability Reliability Index — cross-method agreement. |
| `degradation_viz.py`, `visualization.py` | Plots. |

There is deliberately **no** `attention.py`. The temporal-relevance path lives
next to the IG path so that both are guarded by the same context manager; a
separate module could not guarantee that.

### 2.6 `src/decision_engine/`

| Module | Responsibility |
|---|---|
| `rules.py` | Health state, uncertainty class, explanation reliability, risk score, urgency index, candidate action, and the severity ladder. |
| `constraints.py` | Operational constraint layer applied after the rules. |
| `scenarios.py` | Named operational-constraint scenarios (`unconstrained` … `degraded_operations`). |
| `recommender.py` | Public entry point `recommend_action(state, constraints, thresholds)`. |

### 2.7 `src/hitl/`

| Module | Responsibility |
|---|---|
| `workflow.py` | Turns decisions into review items, resolves reviews, records feedback. |
| `updater.py` | Conservative policy refinement from accumulated feedback. |
| `feedback.py` | `ExpertFeedback` record schema, creation, validation, and append-only storage. |
| `logger.py` | Loads the feedback log and derives summary statistics from it. |

### 2.8 `src/evaluation/`

| Module | Responsibility |
|---|---|
| `metrics.py` | Six evaluation dimensions (predictive accuracy, uncertainty reliability, explainability quality, decision effectiveness, robustness, user trust) plus `full_evaluation_report`, which assembles all six. |
| `benchmark.py` | Comparative benchmark helpers. |
| `ablation.py` | Ablation runner. Re-runs the decision engine on patched inputs — see §9.3. |
| `decision_sensitivity.py` | Risk-score weight sensitivity and the constraint-scenario comparison — see §9.4. |
| `robustness.py` | Noise and sensor-dropout sweeps, cross-subset consistency. |

### 2.9 `src/utils/`

| Module | Responsibility |
|---|---|
| `config.py` | `load_config()`, `config_path_for()`, `DEFAULT_CONFIG`, `ALL_SUBSETS`. |
| `paths.py` | The single place that knows how a **tag** is spelled into a path: `run_dir_name()`, `flat_stem()`, `checkpoint_path()`, `results_path()`, `structured_results_path()`. Without a tag the historical untagged paths are produced unchanged. |
| `logger.py` | Per-script loggers writing to `outputs/logs/`. |
| `io.py` | Small IO helpers. |
| `seed.py` | `set_seed()` — seeds `random`, `numpy`, `torch` and `torch.cuda`, and enables the deterministic torch/CUDA kernels (see §7.7 for the remaining caveat). |
| `provenance.py` | The run identity carried by every generated artifact: `sha256_file()`, `git_commit()`, `build_provenance()`, `pack_provenance()`, `unpack_provenance()`, `compare_provenance()`. The stamp is embedded in the `.npz` (see § 8.1a). |

---

## 3. Configuration

### 3.1 The merge model

Every knob lives in `configs/fd001.yaml` … `configs/fd004.yaml`. Each file
lists only its deviations from `DEFAULT_CONFIG` in `src/utils/config.py`;
`load_config()` deep-merges the file over the defaults, so a script can rely
on every key being present. `DEFAULT_CONFIG` is never mutated.

```python
from src.utils.config import load_config

config = load_config(subset="FD001")          # configs/fd001.yaml over defaults
config = load_config(path="my.yaml")          # explicit file
config = load_config(subset="FD001", overrides={"epochs": 200})
```

### 3.2 Key reference

`DEFAULT_CONFIG` is the fallback for every key; the shipped YAMLs override the
subset-specific ones.

| Group | Keys | Default | Notes |
|---|---|---|---|
| Data | `window_size`, `random_seed`, `device` | `40`, `42`, `cuda` | `device: cuda` falls back to CPU when no GPU is present. |
| Labelling | `max_rul`, `hi_mode`, `hi_baseline_cycles` | `125`, `sensor`, `30` | `max_rul` is the piecewise-linear RUL cap (§3.4); `hi_mode` selects the health-index construction (§2.2a); `hi_baseline_cycles` is the per-engine healthy reference length. |
| Model | `hidden_size`, `num_layers`, `dropout`, `num_heads`, `use_layer_norm` | `128`, `2`, `0.3`, `4`, `true` | `hidden_size` is 160 on FD002/FD004. The architecture itself is **not** a config key — it comes from `--model` and is stored in the checkpoint (see §3.3). |
| Optimisation | `batch_size`, `epochs`, `learning_rate`, `optimizer`, `weight_decay`, `scheduler`, `warmup_epochs`, `min_lr`, `patience`, `grad_clip`, `ema_decay` | `256`, `300`, `3e-4`, `AdamW`, `1e-4`, `cosine`, `5`, `1e-6`, `40`, `1.0`, `0.995` | The shipped YAMLs use `epochs: 150`, `patience: 30`. |
| Loss | `rul_loss_type`, `huber_delta`, `rul_loss_weight`, `hi_loss_weight`, `scale_loss_weight`, `hi_loss_scale` | `mse`, `10.0`, `1.0`, `0.4`, `0.1`, `rul_max` | `hi_loss_scale` puts the HI term on the RUL scale (§7.3); `rul_max` means "multiply the HI loss by `max_rul²`". |
| Augmentation | `augment`, `aug_noise_std`, `aug_bias_std`, `aug_gain_std`, `aug_sensor_dropout`, `aug_time_mask_prob`, `aug_time_mask_len`, `aug_history_crop` | `true`, `0.02`, `0.05`, `0.02`, `0.05`, `0.15`, `8`, `8` | All magnitudes are on the *standardised* feature scale. Augmentation is **off** in the FD001/FD002/FD003 YAMLs — see §7.12. |

`DEFAULT_CONFIG["epochs"]` is 300 while the shipped subset YAMLs say 150. The
YAML wins, so the effective value for all four subsets is 150.

### 3.3 Who reads what

| Consumer | Reads from config |
|---|---|
| `scripts/train.py` | Everything: architecture, optimiser, scheduler, loss weights, augmentation, seed, device. `--overrides key=value,...` patches individual keys. |
| `scripts/preprocess.py` | `window_size`, `max_rul`, `hi_mode`, `hi_baseline_cycles`. All four can be overridden on the CLI. |
| `scripts/evaluate.py` | The architecture recorded in the **checkpoint** (`model_config`), not the YAML. |
| `scripts/uncertainty.py` | The architecture recorded in the checkpoint. |
| `scripts/explain.py` | `window_size` / `max_rul` from the **processed-data metadata**; the architecture from the checkpoint. |

Reading the architecture from the checkpoint rather than from YAML is
deliberate: the checkpoint records the config the weights were actually
trained with, so evaluation can never rebuild a mismatched model.

### 3.4 The `max_rul` contract

`max_rul` is declared in the config and applied by `preprocess.py` when the
labels are generated. `preprocess.py` writes it into the dataset metadata and
`train.py` **refuses to train** when the stored metadata disagrees with the
config. The RUL domain constants in `src/uncertainty/conformal.py` and the
`rul_max` decision threshold are derived from `DEFAULT_CONFIG["max_rul"]`,
so all four stay in sync.

---

## 4. Conformal calibration math

### 4.1 Split conformal (baseline)

Nonconformity scores are absolute residuals on a held-out calibration split:

```text
s_i = |y_i - ŷ_i|,   i = 1..n
```

The finite-sample conformal quantile is

```text
k      = ceil((n + 1) * (1 - α))
q̂      = s_(k)          (the k-th smallest score)
C(x)   = [ŷ - q̂, ŷ + q̂]
```

When `k > n` the correct quantile is `+inf`, not `s_(n)` — see §7.1.

### 4.1a The calibration unit

The **unit** of calibration is what `n` counts, and the repository uses two
different ones. This matters because the finite-sample rank depends on `n`,
and on how small `n` is allowed to get.

| Producer | Unit | `n` |
|---|---|---|
| `scripts/uncertainty.py` | **Window** — every calibration window of the held-out engines contributes one score. | Hundreds (all calibration windows of the engine hold-out). |
| `scripts/multiseed_conformal_benchmark.py` | **Both.** `window` is the primary unit; `engine` is reported alongside it from the same inference pass. | `window`: hundreds; `engine`: one per calibration engine (20 FD001/FD003, 52 FD002, 50 FD004). |
| `src/uncertainty/pipeline.py` | Selected by `compute_all_intervals(..., calibration_units=...)`; returns `results[unit][method][coverage]`. | — |

All units are engine-disjoint from training and from test. The two units answer
different questions:

- `window` keeps every calibration-window score, so `n` is large and the rank is
  trivially attainable at 95 %. It is the unit the test set is scored on (the
  coverage statistic counts test windows), and it tracks the nominal level.
- `engine` keeps one *maximum* score per calibration engine. A lifetime maximum
  is much larger than any single test window, so this unit is conservative: on
  FD003 it reports 0.99 coverage at an 80 % target. It is kept because it is the
  strictest engine-level reading of the guarantee, not because it is the primary
  number.

Because both units share one inference pass (predictions, features, the
quantile head, the scale head and the EARA profiles), adding the second unit
costs only the score reduction.

### 4.2 EARA-Conformal (engine/regime-aware)

1. **Scale.** A `ScaleHead` learns a positive `s(x)` from the latent features
   of the training engines (post hoc, on training-engine residuals).
2. **Regimes.** K-means over the training-engine latent features produces `K`
   degradation regimes.
3. **Normalised scores.** `s_i = |y_i - ŷ_i| / max(scale(x_i), ε)` with a
   shared floor `ε = 1e-3` (`SCALE_FLOOR` in `src/uncertainty/conformal.py`)
   so calibration and prediction cannot disagree on near-zero scales. The
   floor is a *bound*, not an addition: an additive floor would change the
   scale the score was built from and desynchronise the quantile from the
   radius.
4. **Quantile.** `q̂` is computed on the calibration scores. The library
   `calibrate_eara_levels` computes one `q̂_k` per regime; the CLI
   `engine_joint` path and the multi-seed benchmark compute a single global
   `q̂` and let the regime enter only through the retrieved *scale*.
5. **Interval.** `C(x) = [ŷ - q̂ · max(s(x), ε), ŷ + q̂ · max(s(x), ε)]`, clipped
   to the RUL domain `[0, max_rul]` and ordered by the single interval builder
   (`clip_interval`).

Steps 1–3 and 5 are shared. Step 4 is where the producers differ, and the
difference is easy to miss:

| Producer | Step 4 — the quantile | `min_regime_size` |
|---|---|---|
| `src/uncertainty/conformal.py::calibrate_eara_levels` (the library) | One `q̂_k` per regime, on the calibration windows of that regime. | 30 windows |
| `scripts/uncertainty.py::run_engine_joint_conformal` (the CLI `engine_joint`) | A **single global** `q̂` on all scale-normalised calibration scores. | — |
| `scripts/multiseed_conformal_benchmark.py` (`eara_conformal`) | A **single global** `q̂` on the scale-normalised scores of the selected calibration unit. Identical to the CLI path, so the benchmark measures the method the repository actually ships. | — |

The library's per-regime path is the one whose small pools make the finite-
sample rank unattainable (§7.1); neither shipped path is affected because both
use a global quantile over a large pool.

### 4.3 The CLI methods and their artifacts

The four `--method` values are the same machinery with different radius
formulas:

| `--method` | Radius | Output file |
|---|---|---|
| `split` | `q̂` (global) | `<subset>_<model>_conformal.npz` |
| `cqr` | `s(x) + q̂` (quantile head trained with the pinball loss) | `<subset>_<model>_cqr.npz` |
| `adaptive` | `q̂ · s(x)` | `<subset>_<model>_adaptive_conformal.npz` |
| `engine_joint` | `q̂ · s(x)` (EARA-Conformal, global `q̂`) | `<subset>_<model>_engine_joint_conformal.npz` |

The offline benchmark in `scripts/multiseed_conformal_benchmark.py` adds a
fifth method, `nex_scp` — prequential adaptive conformal inference (Gibbs &
Candès), where the miscoverage level is updated from each test engine's
realised outcome in stored order:

```text
alpha_{t+1} = clip(alpha_t + gamma * (alpha - err_t), 0.01, 0.99)
```

`err_t` is 1 when the true label falls outside the interval produced for
engine `t`. The adaptation rate is `--aci-gamma` (default `0.02`). The level is
kept strictly inside `(0, 1)` because a 0 miscoverage level has no conformal
rank; the floor means the adaptor never targets more than 99 % coverage.

---

## 5. The decision policy

The engine is a pure function of a flat state dict. It has no dependency on
torch, on the model, or on any artifact — which is what makes the ablation
(§9.3), the sensitivity analysis (§9.4) and the HITL refinement (§5.5)
possible.

### 5.0 The severity ladder

Every action sits on one explicit ladder:

```text
CONTINUE_OPERATION  <  MONITOR  <  INSPECT  <  SCHEDULE_MAINTENANCE  <  URGENT_MAINTENANCE
   0                    1           2               3                        4
```

It is the *commitment* ordering: the further right, the more of the
maintenance action is committed to and the sooner the operator has to act.
The ladder is defined once, in `src/decision_engine/rules.py::SEVERITY_LADDER`,
and is used by the rule layer, by the constraint layer and by the
sensitivity analysis, so "more severe" means the same thing everywhere.

### 5.1 Health state

From RUL and HI, using `DEFAULT_THRESHOLDS`:

| State | Condition |
|---|---|
| `CRITICAL` | `rul <= 10` **or** `hi <= 0.20` |
| `AT_RISK` | `rul <= 25` **or** `hi <= 0.40` |
| `DEGRADING` | `rul <= 50` **or** `hi <= 0.70` |
| `HEALTHY` | otherwise |

The three HI breakpoints above are the defaults of the *legacy*
`HI = RUL / max_rul` label.  With the sensor-derived index they are
recalibrated on the training split so that the HI criterion fires at the
same degradation levels as the RUL criterion (§2.2a); the calibrated values
are stored in the dataset metadata and merged into the thresholds by
`scripts/decision.py`.

The engine evaluates this classification twice: once on the **point
estimate** (reported as `health_state`) and once on the **conservative end
of the calibrated interval** (reported as `planning_state`, §5.3).

### 5.2 Risk score

```text
R = w_rul · r_rul + w_hi · r_hi + w_u · r_u + w_eri · r_eri

r_rul  = 1 - clip(rul / rul_max, 0, 1)
r_hi   = 1 - hi
r_u    = clip((rul_upper - rul_lower) / rul_max, 0, 1)
r_eri  = 1 - eri

w_rul = 0.40,  w_hi = 0.30,  w_u = 0.15,  w_eri = 0.15
U     = min(R · (1 + uncertainty_amplifier · r_u), 1.0)
```

The sensitivity of the decision mix to these four weights is measured in
§9.4 rather than assumed.

### 5.3 Candidate action

The action is **not** read off a hand-written table.  It is computed from
the ladder:

```text
severity = min(base(planning_state) + eri_step, 4)
```

where the **planning state** is the health state the engine would be in if
the prediction were at the *conservative end* of its calibrated interval —
i.e. `determine_health_state(rul_lower, hi)`.

| Planning state | base severity | base action |
|---|---|---|
| `HEALTHY` | 0 | `CONTINUE_OPERATION` |
| `DEGRADING` | 1 | `MONITOR` |
| `AT_RISK` | 3 | `SCHEDULE_MAINTENANCE` |
| `CRITICAL` | 4 | `URGENT_MAINTENANCE` |

| Confidence signal | escalation step |
|---|---|
| explanation reliability `HIGH` / `MODERATE` | 0 |
| explanation reliability `LOW` | +1 |

The point estimate's state is reported as `health_state`; the state the
action is planned for is reported as `planning_state`.  The two differ
whenever the interval is wide enough to cross a threshold, and the
decision record then says so in its `reasons`.

**Why the planning state.**  A maintenance work order has to be scheduled
so that it is carried out *before the worst credible case*, not before the
most likely one.  The calibrated interval is exactly the statement of what
is credible, so the decision is planned on its lower bound.  The
consequences are:

* the action is monotone in the uncertainty **by construction** — widening
  the interval downwards can only make the planning state more severe, so
  it can only make the action more severe;
* the uncertainty is *not* a second escalation step on top of that, which
  would count the same evidence twice (and, because the ladder has only
  one step above `SCHEDULE_MAINTENANCE`, would push every at-risk engine
  straight to `URGENT_MAINTENANCE`);
* the *width* of the interval is handled where it belongs: the
  human-review flag below, and the operational constraints of §5.4, which
  turn "how much time is left" into a timing escalation
  (`maintenance_lead_time`).

**Why it is written this way.**  The policy is *monotone by construction*:
a less reliable explanation, or a wider interval, can never produce a less
severe action than the same evidence with better support.  The previous
hand-written table violated that: for `AT_RISK` it produced `INSPECT` for
`HIGH` uncertainty and `SCHEDULE_MAINTENANCE` for `MODERATE` uncertainty,
so more uncertainty produced a *less* committed action (§7.4).  Under the
severity ladder that inconsistency is structurally impossible — `INSPECT`
sits *below* `SCHEDULE_MAINTENANCE` and is only reachable from the
`DEGRADING` band or from a low-ERI escalation.

**Human review.** A degraded confidence signal always requires a human to
look at the record, even when the action itself did not change:

| Condition | Human review |
|---|---|
| uncertainty `HIGH` (relative interval width ≥ 0.50) | yes |
| explanation reliability `LOW` | yes |
| risk score > 0.85 (hard override to `URGENT_MAINTENANCE`) | yes |
| otherwise | no |

### 5.4 Constraints

Applied to the candidate action, in this order. Each rule only escalates
severity, except the maintenance-window rule.

| # | Constraint | Effect |
|---|---|---|
| 1 | `maintenance_window_available = False` | `SCHEDULE_MAINTENANCE` → `INSPECT`, human review. **Substitution**, not escalation: the plan is infeasible, so the nearest feasible action is used. Recorded separately in `constraint_substitutions`. |
| 2 | `minimum_safe_rul` | Passive action with `rul <= minimum_safe_rul` → `URGENT_MAINTENANCE`. |
| 3 | `maintenance_lead_time` | `SCHEDULE_MAINTENANCE` with `rul <= maintenance_lead_time` → `URGENT_MAINTENANCE`. |
| 4 | `max_allowable_delay` | Passive action with `rul <= max_allowable_delay` → `INSPECT`. |
| 5 | `resource_availability = False` | Active action → human review (the action itself is not changed). |

The order matters and is not commutative. Rule 1 rewrites
`SCHEDULE_MAINTENANCE` to `INSPECT`, so rule 3 can no longer see a scheduled
maintenance and will not fire on the same record. Rule 2 escalates a passive
action to `URGENT_MAINTENANCE`, which is no longer passive, so rule 4 will not
fire either.

The constraint layer records `constraint_status`
(`SATISFIED` / `VIOLATION`), the list of `constraint_violations`, the
list of `constraint_adjustments`, the list of `constraint_substitutions`,
and the severity of the final action together with its delta against the
unconstrained action, so every escalation is auditable.

**Named scenarios.** `src/decision_engine/scenarios.py` defines the
operational contexts the decision engine is evaluated under:
`unconstrained`, `no_maintenance_window`, `tight_safety_margin`,
`lead_time_pressure`, `resource_shortage` and `degraded_operations`.
`scripts/decision.py --constraint-scenario <name>` applies one, and
`scripts/multiseed_pipeline.py` records the scenario that produced each
decision artifact.  The certified runs use `degraded_operations`, so the
constraint layer is exercised by the published results instead of only by
unit tests (§9.4).

### 5.5 HITL

`src/hitl/updater.py` refines the **policy**, never the model. An
alternative action is adopted only with repeated, high-confidence,
high-consensus support (≥ 3 observations, weighted support ≥ 2.0, consensus
≥ 0.70), and `URGENT_MAINTENANCE` is never automatically weakened.
Constraints are re-checked after every adaptation. A review submitted during
a run is recorded but excluded from that run's adaptation pass, so it cannot
leak into the decision it just reviewed.

---

## 6. Explainability Reliability Index (ERI)

```text
ERI = 0.4 · (ρ_spearman(IG, Attention) + 1) / 2
    + 0.4 · (ρ_spearman(IG, SHAP) + 1) / 2
    + 0.2 · consensus_top_k
```

- `ρ_spearman` is the Spearman rank correlation between two sensor-importance
  vectors. The `(ρ + 1) / 2` normalisation maps `ρ ∈ [-1, 1]` onto
  `[0, 1]`, so an **uncorrelated pair (`ρ = 0`) contributes 0.5** to its term
  rather than 0.
- `consensus_top_k` is the fraction of the top-K sensors that all three
  methods agree on.

ERI is bounded in `[0, 1]`. The decision engine treats `eri < 0.60` as `LOW`
reliability and `eri >= 0.80` as `HIGH`.

> **Band boundaries.** The two band thresholds live in
> `src/utils/config.py` (`ERI_LOW_THRESHOLD`, `ERI_HIGH_THRESHOLD`) and are
> used by **both** `src/explainability/eri.py` and
> `src/decision_engine/rules.py`, with the same inclusive comparison. The
> two modules used to disagree on the boundary values 0.60 and 0.80 — the
> report used `>` and the policy `>=` — so an ERI of exactly 0.80 was
> `HIGH` for the policy and `MODERATE` for the report. The report and the
> policy now classify every value identically.

---

## 7. Known gaps

These are real and are listed so that no number in `RESULTS.md` is read as
stronger than it is.

<a id="71-conformal-rank-clamp--the-coverage-guarantee-is-void"></a>
### 7.1 The conformal rank clamp survives in the benchmark *(fixed)*

The finite-sample conformal rank is `k = ceil((n + 1)(1 - α))`. When `k > n`
— i.e. whenever `α < 1/(n + 1)` — the mathematically correct quantile is
**infinity**: the calibration set is too small to certify `1 - α` coverage.
Clamping the rank to `n` instead returns the largest calibration score, which
produces a *finite* interval whose nominal coverage is **not** justified.

**The library is fixed.** `src/uncertainty/conformal.py::conformal_quantile`
no longer clamps. The rank is computed by the extracted
`conformal_rank(n, alpha)` helper, and when the rank is unattainable the
function returns `+inf` (`allow_infinite=True`, the default) or raises a
`ValueError` (`allow_infinite=False`):

```python
rank = conformal_rank(n, alpha)
if rank > n:
    if not allow_infinite:
        raise ValueError(
            f"The finite-sample conformal rank k={rank} exceeds the "
            f"number of calibration scores n={n}; the requested "
            f"coverage {1.0 - alpha:.2%} is not attainable at this "
            f"calibration size."
        )
    return float("inf")
return float(sorted_scores[rank - 1])
```

`regime_conformal_quantile` inherits the same behaviour and returns `+inf`
for a regime with no calibration data at all.

**The script-local copies are gone.** There used to be three independent
implementations of the quantile:

| Copy | Status |
|---|---|
| `src/uncertainty/conformal.py::conformal_quantile` | fixed (returns `+inf`) |
| `scripts/multiseed_conformal_benchmark.py::finite_sample_quantile` | **deleted** — the benchmark now calls the shared pipeline |
| `scripts/uncertainty.py::cqr_quantile` | **delegates** to the shared implementation |

`scripts/uncertainty.py::cqr_quantile` now calls
`src.uncertainty.conformal::signed_conformal_quantile`. CQR uses the
**signed** nonconformity score `max(q_lo(x) − y, y − q_hi(x))`, which is
*negative* whenever the truth falls inside the predicted quantile band. The
non-negativity check of `conformal_quantile` must not be applied to it, so a
dedicated signed variant exists — with the same unclamped rank semantics:

```python
def signed_conformal_quantile(scores, alpha, allow_infinite=True):
    ...
    rank = conformal_rank(n, alpha)
    if rank > n:
        if not allow_infinite:
            raise ValueError(...)
        return float("inf")
    return float(sorted_scores[rank - 1])
```

**The benchmark is unclamped and reports both calibration units.**
`scripts/multiseed_conformal_benchmark.py` calls
`src/uncertainty/pipeline.py::compute_all_intervals` with
`calibration_units=("window", "engine")`. The primary unit is `window` (one
score per calibration window, thousands of scores); `engine` (one max-residual
score per calibration engine: 20 for FD001/FD003, 52 for FD002, 50 for FD004)
is reported alongside it as the conservative reading. Both go through
`finite_sample_quantile()` → `conformal_quantile()`, so an unattainable rank
becomes `+inf` and then a full-domain interval, never a clamped one. There is
no per-regime pool in either shipped path, so the clamped-regime problem does
not arise.

**The published numbers have been regenerated.** `RESULTS.md` § 3 reports the
coverage of intervals that carry a valid conformal guarantee, computed on the
real deliverable checkpoints rather than on a proxy model, for both units.

### 7.2 The HI label is a deterministic function of RUL *(fixed)*

`src/preprocessing/labeling.py` used to set `HI = RUL / max_rul`. On FD001–FD004
the correlation between the two labels was `1.000000` to machine precision.
The HI head therefore had almost nothing to learn beyond the RUL head, and
the strong HI numbers in the result tables were a by-product of the shared
trunk, not evidence of a well-supervised HI task.

**The label is now derived from the sensor signal.**  See §2.2a for the
construction.  The index is a genuine health indicator: it is strongly, but
not perfectly, related to the degradation state, and it is *not* a
deterministic function of the RUL label any more.  The measured
`corr(HI, RUL)` on the training split is 0.890 (FD001), 0.674 (FD002),
0.797 (FD003) and 0.705 (FD004) — the spread between subsets is itself the
evidence that the index is a *measurement* and not a re-scaling: the
multi-condition subsets have a weaker sensor-to-health relation, exactly as
one would expect.

The legacy label is still reachable through `--hi-mode rul`, so the two
constructions can be compared on the same pipeline.

### 7.3 The HI loss term is numerically negligible *(fixed)*

In `src/models/losses.py` the RUL MSE is on the order of `10³` while the
bounded HI MSE is on the order of `10⁻¹`. `hi_loss_weight = 0.4` therefore
changed the total loss by roughly `1e-5` in relative terms, and there was
no gradient balancing (no `GradNorm`, no uncertainty weighting). Changing
`hi_loss_weight` in the config had almost no effect.

**The HI term is now put on the RUL scale.**  `MultiTaskLoss` takes a
`hi_scale` factor; the pipeline passes `max_rul ** 2`, which expresses the
bounded HI loss in cycle-equivalent units.  The two terms are then of the
same order of magnitude and `hi_loss_weight` is a real dial: with the
default `0.4` the HI term contributes a few percent of the RUL term's
magnitude instead of `1e-5`.  `hi_loss_scale: none` restores the old
behaviour, and `tests/test_losses.py` pins both.

The RUL term itself is deliberately *not* rescaled: dividing it by
`max_rul²` would shrink the gradients by four orders of magnitude and
push them towards Adam's epsilon, which changes the optimisation far more
than the HI weighting does.

### 7.4 The AT_RISK action table is not monotonic in uncertainty *(fixed)*

For `AT_RISK`, `HIGH` uncertainty yielded `INSPECT` while `MODERATE`
uncertainty yielded `SCHEDULE_MAINTENANCE` — i.e. *more* uncertainty produced
a *less* severe action. Every other health state escalated with
uncertainty. This was a policy inconsistency, not a code error; it needed a
policy decision, not a patch.

**The policy decision was made and the table was replaced.**  The
repository now defines an explicit severity ladder (§5.0) and computes the
action from it (§5.3) instead of reading it off a hand-written table.  The
policy is monotone *by construction*: a wider interval can only make the
planning state more severe, and a less reliable explanation can only
escalate the action.  `tests/test_decision_policy.py` verifies both
properties — the severity never falls as the interval widens, and an
at-risk engine is never given a less committed action than
`SCHEDULE_MAINTENANCE`.

The behavioural consequences are:

* the action is planned on the **conservative end of the calibrated
  interval** rather than on the point estimate.  An at-risk engine whose
  interval still reaches into the at-risk band is `SCHEDULE_MAINTENANCE`;
  one whose interval reaches into the critical band is
  `URGENT_MAINTENANCE`.  Both `health_state` (the point estimate) and
  `planning_state` (the conservative end) are reported.
* `INSPECT` is no longer reachable from the `AT_RISK` band, so the
  inconsistency is structurally impossible.
* the `HEALTHY` and `DEGRADING` bands keep their old base actions.

The published action mix moved accordingly; see `RESULTS.md`.

### 7.5 The ERI's IG-vs-Attention term is high by construction

`src/explainability/eri.py::_sensor_importance_from_temporal_relevance`
derives the attention-side sensor importance by **re-weighting the IG
attributions** with the mean temporal relevance:

```python
weighted = np.abs(ig_attributions) * mean_temporal_relevance
```

So `rho(IG, Attention)` measures how much the temporal relevance *re-ranks*
the IG importances, not whether two independent methods agree. With a
uniform temporal relevance — what a trained model usually produces — the two
vectors are identical and `rho = 1.0` exactly. A measured `rho(IG, Attn)`
of `0.996` therefore says nothing about explanation quality.

Only the `rho(IG, SHAP)` term compares two genuinely independent methods,
and it carries `0.4` of the ERI weight. The `0.4` on the attention term
inflates every ERI reported in `RESULTS.md`.

### 7.6 Ablation results are degenerate *(fixed)*

`src/evaluation/ablation.py` used to patch metric keys that the pipeline does
not actually produce, so the `full` / `no_uncertainty` / `no_xai` /
`no_constraints` variants reported identical numbers and a `delta` of exactly
`0.0` for every metric.

The runner now **re-runs the decision engine** (`recommend_action`) on the
patched inputs of every record, so each variant goes through the same code
path as the full pipeline. `no_constraints` bypasses the constraint layer
instead of clearing violations after the fact, and the report carries the
review-rate, constraint-violation, risk-score and action-mix deltas.

Two caveats remain: the ablation inherits the *policy* of
`src/decision_engine/rules.py` (so it measures the policy's sensitivity to a
missing input, not the causal contribution of the input itself), and a
variant whose component was never exercised in the recorded run — e.g.
`no_constraints` for a decision produced without operational constraints —
legitimately reports a zero delta.

### 7.7 Seed control is complete for training and calibration, partial for XAI

`src/utils/seed.py::set_seed` is the single source of truth. It seeds
`random`, `numpy`, `torch` and `torch.cuda`, optionally enables
`torch.use_deterministic_algorithms`, pins cuDNN to deterministic mode,
disables TF32, sets `CUBLAS_WORKSPACE_CONFIG`, and returns the determinism
block that gets persisted in the checkpoint. `scripts/train.py` calls it
before it touches any RNG and stores the resulting block under
`run_metadata["determinism"]`, together with the seed itself
(`run_metadata["seed"]` and a flat `checkpoint["seed"]`). A training run is
therefore identified by its seed alone, and the multi-seed harness can prove
that it did not accidentally vary anything else.

Determinism is **on by default** (`config["deterministic"]`; opt out with
`--no-deterministic`). It costs roughly 1.7× runtime on the reference machine.

The calibration stage is now seed-complete too:

- `scripts/train.py` splits engines with
  `GroupShuffleSplit(random_state=seed)`, and `scripts/uncertainty.py` reads
  the seed back out of the checkpoint (`checkpoint_seed()`) before it
  reproduces that split. Before this, a seed-43 checkpoint was calibrated
  against the seed-42 hold-out, which can contain engines the seed-43 model
  was trained on — a silent leakage bug. `scripts/verify_seed_alignment.py`
  asserts that the three seeds produce three different 20-engine calibration
  sets and that each is reported from its own checkpoint.
- The auxiliary heads (`fit_quantile_head`, `fit_scale_head`) in
  `src/uncertainty/pipeline.py` create an explicit `torch.Generator` seeded
  from the checkpoint seed instead of relying on global RNG state.
- `scripts/multiseed_conformal_benchmark.py` never seeds anything itself: it
  derives the split from the checkpoint seed and passes that seed through to
  the pipeline, so two benchmark invocations produce identical rows.

What is **not** covered:

- `scripts/explain.py` seeds only `numpy` and `torch` locally, so a SHAP
  background sample is reproducible within a backend but not bit-identical
  across CPU/CUDA or across torch versions.
- `scripts/sota_benchmark.py` seeds its bootstrap resamples, but that
  bootstrap is **engine-level only**: it measures test-sampling noise, not
  seed-to-seed model noise. The multi-seed report keeps the two sources
  distinct and never adds them together.
- `warn_only=True` means a device without a deterministic kernel degrades to a
  warning instead of aborting the run.

### 7.8 Checkpoints are loaded tolerantly

`src/models/checkpoint.py::load_state_dict_compat` loads with
`strict=False` and re-initialises a missing scale head to a neutral scale.
`src/models/trainer.py` discards the missing/unexpected key lists rather
than logging them, so an architecture mismatch is silent. The scripts now
rebuild the architecture from the checkpoint's own `model_config`, which
removes the most common cause of this mismatch, but the tolerant loader
itself is still a silent-failure path.

### 7.9 Attention weights are not used by the model

`src/models/gru_att_deg.py` computes attention weights that are correct but
discarded before the prediction head. The attention importances used by ERI
come from `src/explainability/integrated_gradients.py::temporal_gradient_relevance`,
which re-derives them from the forward pass through the same eval-guarded
gradient path as Integrated Gradients.

### 7.10 Test coverage is uneven

`tests/test_decision.py`, `tests/test_uncertainty.py`, `tests/test_models.py`,
`tests/test_xai.py`, `tests/test_evaluation.py` and `tests/test_sota_benchmark.py`
are real and run without the dataset. Three areas are still uncovered:

- **The preprocessing pipeline** (`tests/test_preprocessing.py`) is an
  integration test that needs `data/raw/` and skips automatically when the
  dataset is absent, so CI never exercises it.
- **The training loop** (`src/models/trainer.py`) has no unit test. The
  EMA weights, the warm-up schedule, the early-stopping patience and the
  gradient clipping are only validated by the end-to-end runs.
- **The decision CLI and the HITL CLI** are covered at the library level
  (`src/decision_engine`, `src/hitl`), not through `scripts/decision.py` and
  `scripts/hitl.py` end-to-end.

### 7.11 The ERI Spearman correlation handles tied ranks

`src/explainability/eri.py::_spearman` now uses `scipy.stats.spearmanr`, the
same implementation as `src/evaluation/metrics.py`, so the two modules can no
longer disagree on the same pair of vectors. The previous rank-difference
shortcut `1 − 6·Σd² / (n(n²−1))` is only exact for tie-free data: tied
importances received arbitrary but distinct ranks from the double `argsort`,
which biased the reported `ρ` towards 1. The top-K consensus score remains
tie-sensitive by construction, because it is a set intersection.

### 7.12 Augmentation is disabled in three of the four subset configs

`DEFAULT_CONFIG["augment"]` is `true`, but `configs/fd001.yaml`,
`configs/fd002.yaml` and `configs/fd003.yaml` all set `augment: false` after
an ablation showed augmentation *raised* RUL RMSE (FD001 11.96 → 12.56,
FD002 13.61 → 14.72). Only FD004 keeps it on. Two consequences:

- The `augment` key in the config is **not** a reliable description of what a
  default run does — the YAML wins.
- The default recipe therefore does **not** exercise
  `src/models/augmentation.py` on three of four subsets, so augmentation bugs
  can hide until someone runs an explicitly augmented job.

### 7.13 The uncertainty artifact layout is legacy and not tag-aware *(fixed)*

`scripts/decision.py` reads intervals from the **flat**
`outputs/results/<subset>_<model>[_<tag>]_<variant>.npz` and falls back to
the legacy structured
`outputs/results/uncertainty/<variant>/<subset>/<model>[_<tag>].npz`. No
current script *writes* the structured layout — `scripts/uncertainty.py`
writes only the flat files — so the structured tree is a read-path
fallback. The four variants the CLI produces (`conformal`, `cqr`,
`adaptive_conformal`, `engine_joint_conformal`) are all flat.

The flat file must be preferred, and it used not to be: the resolver took
the structured path whenever it existed. The five FD001/`gru` files left in
that tree were four weeks old and disagreed with the flat artifacts by up
to 16 RUL cycles in their point predictions, so a maintenance decision
could have been produced from a different model's intervals than the ones on
disk. The resolver now takes the flat file first, keeps the structured tree
as a fallback for when it is absent, and logs which layout it used. The
stale files were deleted.

Every downstream stage now accepts a `--tag` and spells it through
`src/utils/paths.py::run_dir_name`:

| Script | `--tag` flag | Reads a tagged artifact | Writes a tagged artifact |
|---|---|---|---|
| `scripts/train.py` | yes | — | yes (`outputs/checkpoints/<model>[_<tag>]/<subset>/best_model.pt`) |
| `scripts/evaluate.py` | yes | yes (checkpoint) | yes (`<subset>_<model>_<tag>_predictions.npz`, `..._metrics.json`) |
| `scripts/uncertainty.py` | yes | yes (checkpoint) | yes (`<subset>_<model>_<tag>_<variant>.npz`) |
| `scripts/explain.py` | yes | yes (checkpoint) | yes (`outputs/xai/<subset>/<model>_<tag>/`) |
| `scripts/decision.py` | yes | yes (predictions, intervals, XAI) | yes (`decision_<subset>_<model>_<tag>.json/.html`) |
| `scripts/hitl.py` | yes | yes (the tagged decision JSON) | yes (`hitl_<subset>_<model>_<tag>.json/.html`) |

The tag travels *inside* the decision JSON (`"tag": "ms42"`), so HITL derives
its output stem from the decision artifact it was given rather than from a
second, independently-supplied flag. There is no code path left in which two
seeds of the same model write to the same file.

### 7.14 `src/uncertainty/pipeline.py` is wired in *(fixed)*

`src/uncertainty/pipeline.py` is a self-contained, model-agnostic interval
engine: window- and engine-level calibration, the scale and quantile heads, the
EARA engine profiles, prequential ACI, and the interval metrics. It is written
against the *fixed* conformal quantile and exposes
`conformal_rank_attainable()` so a caller can tell whether a requested
coverage is certifiable at the calibration size it has.

It now has two consumers:

* `scripts/multiseed_conformal_benchmark.py` calls
  `compute_all_intervals()` directly on each real checkpoint — the proxy
  `IntervalRegressor` is gone. All five methods (split, CQR, adaptive,
  `nex_scp` prequential ACI, EARA) are computed by one shared code path, so
  the comparison between them is paired by construction. It requests both
  calibration units in one call; `window` is the primary unit and `engine` is
  the conservative reading.
* `scripts/uncertainty.py` delegates the shared primitives — `_robust_scale`,
  `_build_training_engine_profiles`, `_nearest_engine_regime_scale`, the
  quantile-head and scale-head fitting helpers — to thin wrappers over this
  module. The production CLI keeps its **window-level** calibration unit
  (which is what its historic artifacts were computed with), which is also the
  benchmark's primary unit. Both share every other primitive.

`compute_all_intervals()` also attaches `y_rul_pred` and `y_rul_true` to each
method result, so a downstream consumer can pair an interval with the point
prediction that belongs to it. `scripts/multiseed_pipeline.py` verifies this:
it checks that the prediction stored by the uncertainty stage agrees with the
prediction stored by the evaluation stage to within 0.1 RUL cycles (the two
stages run the same weights with different batch sizes, so the GPU forward
pass is not bit-identical) and that the two correlate above 0.9999.

---

<a id="715-artifacts-do-not-record-which-checkpoint-produced-them-fixed"></a>
### 7.15 Artifacts do not record which checkpoint produced them *(fixed)*

`scripts/evaluate.py` and `scripts/uncertainty.py` write their artifacts
under names derived from the (subset, model, tag) triple, and nothing inside
the artifact said which checkpoint had produced it. `comparative_benchmark`
pairs the two by *filename* and compares the predictions against the
intervals, so regenerating one side and not the other silently produced a
comparison between **two different models** that still looked entirely
valid.

The multi-seed harness never had this problem — `results/multiseed/training_manifest.json`
records the seed, the git commit and the checkpoint SHA-256 for every run —
but the single-run chain in `outputs/results/` did.

Every `.npz` now carries a provenance stamp (see § 8.1a) and
`comparative_benchmark` compares the two stamps of each pair, reporting
`ok` / `mismatch` / `unknown` and warning when the two sides disagree.

The same gap existed in `comparative_benchmark`'s **summary** block, which
looked the model up under the untagged key while the report was written
under the tagged one — so every tagged run reported
`best_model_by_rul_mae: None` and `best_rul_mae: inf`. Fixed.

---

## 8. Artifact contracts

### 8.0 Naming rules

`src/utils/paths.py` is the single place that knows how a run **tag** is
spelled into a path. The rules are:

```text
run directory   <model>[_<tag>]
checkpoint      outputs/checkpoints/<model>[_<tag>]/<subset>/best_model.pt
flat artifact   outputs/results/<subset>_<model>[_<tag>]_<suffix>.<ext>
structured      outputs/results/uncertainty/<variant>/<subset>/<model>[_<tag>].<ext>
```

The `[<tag>]` part is optional. Without a tag the historical untagged paths
are produced unchanged, so every existing artifact and every existing
consumer keeps working. `scripts/train.py` derives the tag from `--tag`
(e.g. `--tag seed7` → `gru_att_deg_v2_seed7`), and the multi-seed harness
uses `--tag-prefix` to build the `ms<seed>` family.

### 8.1 Files

| Artifact | Contract |
|---|---|
| `data/processed/<subset>_<split>_X.npy` | `float32 (n_windows, window_size, n_features)` |
| `data/processed/<subset>_<split>_y_rul.npy` | `float32 (n_windows,)` |
| `data/processed/<subset>_<split>_y_hi.npy` | `float32 (n_windows,)` in `[0, 1]` |
| `data/processed/<subset>_<split>_engine_ids.npy` | `int (n_windows,)` — the source engine of each window |
| `data/processed/<subset>_<split>_metadata.json` | `window_size`, `stride`, `samples`, `features`, `engines`, `has_engine_ids`, `max_rul`, `subset`, `hi_mode`, `hi_baseline_cycles`, `health_index`, `hi_thresholds` |
| `outputs/checkpoints/<arch>[_tag]/<subset>/best_model.pt` | `model_state_dict` (EMA weights), `raw_model_state_dict`, `model_config` (incl. `arch`), `input_size`, `window_size`, `subset`, `history`, `metrics`, `seed`, `run_metadata`, `ema_decay` |
| `outputs/results/<subset>_<model>_predictions.npz` | `y_rul_true`, `y_rul_pred`, `y_hi_true`, `y_hi_pred`, `meta_json` (provenance stamp — see § 8.1a) |
| `outputs/results/<subset>_<model>_metrics.json` | `{"metrics": {"rul": {"MAE","RMSE","R2","NASA_Score"}, "hi": {"MAE","RMSE","R2"}}}`. The keys are lower-case; an earlier revision wrote `{"RUL","HI"}`. Also carries `provenance`. |
| `outputs/results/<subset>_<model>_<variant>.npz` | `y_rul_test`, `y_hi_test`, `y_rul_pred`, plus per-coverage-level `lower_<pct>`, `upper_<pct>`, `q_hat_<pct>`, the method-specific arrays (`test_scales`, `test_regimes`, `calibration_scales`, `calibration_regimes`, `normalized_calibration_scores`) and `meta_json` (provenance stamp). `<variant>` is `conformal`, `cqr`, `adaptive_conformal` or `engine_joint_conformal`. |
| `outputs/xai/<subset>/<model>/xai_<target>.npz` | `ig_attributions`, `shap_sensor_values`, `attention_temporal_relevance`, `eri`, `rho_ig_attn`, `rho_ig_shap`, `consensus_k` |
| `outputs/xai/<subset>/<model>/eri_<target>.json` | `eri`, `rho_ig_attn`, `rho_ig_shap`, `consensus_k`, `top_k`, `sensor_importance` (`ig` / `shap` / `temporal_relevance`), `sensor_ranks`, `interpretation` |
| `outputs/results/decision_<stem>.json` | Top level: `objective`, `description`, `subset`, `model`, `tag`, `constraint_scenario`, `thresholds`, `inputs`, `summary`, `decisions` (list), `xai_context`, `operational_constraints` (dict). Each decision record: `index`, `engine_id`, `inputs` (`rul`, `hi`, `rul_lower`, `rul_upper`, `eri`), `health_state`, `uncertainty_level`, `explanation_reliability`, `risk_score`, `urgency_index`, `risk_components`, `recommended_action`, `action_severity`, `human_review`, `reasons`, `constraint_status`, `constraint_violations`, `constraint_adjustments`, `constraint_substitutions`, `severity_delta`, `top_k_sensors` |
| `outputs/models/health_index_<subset>.pkl` | The fitted health-index model: the per-sensor condition-regression coefficients, the PLS direction and the two reference quantiles. Written by `scripts/preprocess.py`. |
| `outputs/hitl/feedback.jsonl` | Append-only. One JSON object per line. |
| `outputs/reports/accuracy_summary.csv` | `subset,model,n,rul_mae,rul_rmse,rul_r2,nasa,hi_mae,hi_rmse,hi_r2` — written by `scripts/summarize.py` |
| `outputs/reports/decision_sensitivity.{json,csv}` | The risk-score weight sensitivity report — see § 9.4. |
| `outputs/reports/decision_weight_sweep.csv` | One row per swept weight vector — see § 9.4. |
| `outputs/reports/decision_constraint_scenarios.csv` | One row per (run, scenario) — see § 9.4. |
| `results/conformal_benchmark/summary_mean_std.csv` | `subset,method,nominal_coverage,n_seeds,empirical_coverage_mean,…` — tracked in git, written by the multi-seed benchmark |
| `results/multiseed/training_manifest.json` | One entry per (model, subset, seed) run: the seed, the resolved config, the git commit, the SHA-256 of the checkpoint, and the determinism configuration that was active. Written by `scripts/multiseed_train.py`. |

### 8.1a Artifact provenance

Every artifact this repository writes is derived from a **checkpoint**, and
two artifacts are routinely compared against each other — the predictions
from `scripts/evaluate.py` and the intervals from `scripts/uncertainty.py`.
Such a comparison is only meaningful if both sides came from the same
weights.

`src/utils/provenance.py` writes a stamp into every `.npz` under the
`meta_json` key:

| Field | Meaning |
|---|---|
| `checkpoint_sha256` | SHA-256 of the checkpoint file. **This is the artifact's identity.** |
| `checkpoint_path` | Where that checkpoint was read from. |
| `seed`, `epoch`, `arch` | The run's coordinates, read out of the checkpoint. |
| `subset`, `model`, `tag` | The run's coordinates, as spelled on the command line. |
| `git_commit` | The commit the artifact was produced at. |
| `written_utc` | When the artifact was written. |

Two implementation details matter:

* **The stamp lives inside the `.npz`, not in a sidecar.** A sidecar can be
  deleted, or survive the artifact it describes; an embedded stamp is written
  and moved together with the data.
* **It is a 0-d numpy *unicode* array, not an object array.** `np.load` can
  therefore read it with the default `allow_pickle=False` — a pickled stamp
  would make every consumer of the artifact opt into pickle.

`src/evaluation/benchmark.py::comparative_benchmark` compares the two stamps
of every (subset, model) pair and reports:

| `provenance.status` | Meaning |
|---|---|
| `ok` | Both artifacts declare the same checkpoint. |
| `mismatch` | Both declare a checkpoint, but different ones. The comparison is not trustworthy; the report carries a `warning`. |
| `unknown` | At least one artifact predates the stamp (or the checkpoint was missing). Nothing is asserted. |

`strict_provenance=True` turns a `mismatch` into a `ValueError` instead of a
warning, for pipelines that must never compare mismatched artifacts.

### 8.2 Decision input format

`data/external/decision_input_<subset>.json` (falling back to
`data/external/decision_input.json`) is the input `scripts/decision.py`
resolves automatically. The file holds a **flat** record per engine:

```json
{
  "rul": 42.0,
  "hi": 0.35,
  "rul_lower": 30.0,
  "rul_upper": 55.0,
  "eri": 0.78,
  "top_k_sensors": [3, 7, 11]
}
```

`rul`, `hi`, `rul_lower`, `rul_upper` and `eri` are required. `top_k_sensors`
and engine identifiers are optional. `rul_lower > rul_upper` is rejected by
the engine with a `ValueError`.

### 8.3 HITL review identifiers

Review items are created with `review_id = f"decision-{identifier}-{index}"`,
where `identifier` is the engine id when the decision carries one and the
record index otherwise. List the actual ids from the review queue rather
than assuming a fixed string.

---

## 9. The evaluation framework

### 9.1 Six dimensions

`src/evaluation/metrics.py` organises every reported number into six
dimensions, assembled by `full_evaluation_report`:

| # | Dimension | Entry point | Needs |
|---|---|---|---|
| 1 | Predictive accuracy | `predictive_accuracy` | prediction arrays |
| 2 | Uncertainty reliability | `uncertainty_reliability` | predictions + 90 % interval + engine ids |
| 3 | Explainability quality | `explainability_quality` | an `eri_<target>.json` |
| 4 | Decision effectiveness | `decision_effectiveness` | a decision JSON |
| 5 | Robustness | `robustness_under_noise`, `robustness_under_dropout` | a live model + `X_test` |
| 6 | User trust | `user_trust_proxies` | a HITL report + the feedback log |

Dimension 5 is only computed when a model and `X_test` are supplied — it
needs a live PyTorch model.

### 9.2 The metric that lives in one place

`predictive_accuracy` in `src/evaluation/metrics.py` is the **only**
implementation of MAE / RMSE / R² / NASA score in the repository. It is used
by `scripts/evaluate.py`, `scripts/ensemble.py` and `scripts/summarize.py`
alike, so the same predictions always produce the same numbers regardless of
which script wrote them. `src/models/evaluator.py`, which carried a second,
slightly different copy, was removed.

### 9.3 Ablation semantics

`src/evaluation/ablation.py` does not "subtract a metric". It patches the
**inputs** of every decision record and re-runs `recommend_action`, so each
variant goes through exactly the same code path as the full pipeline.

| Variant | What it does |
|---|---|
| `full` | Re-runs the decision engine with the recorded inputs and the recorded operational constraints. The result is compared against the stored decisions. |
| `no_uncertainty` | Replaces the conformal interval with a **fixed width of 100 cycles**, so the uncertainty class is no longer derived from calibration. |
| `no_xai` | Sets ERI to **0.5** — below the `eri_low` threshold, so the explanation reliability is forced to `LOW`. |
| `no_constraints` | Runs the engine with **no** operational constraints, i.e. the constraint layer is not applied at all. |
| `point_only` | Not a decision variant: it reports the predictive accuracy of the saved predictions and nothing else. It has no uncertainty, XAI, or constraint component. |

The report carries `delta_human_review_rate`,
`delta_constraint_violation_rate`, `delta_mean_risk_score` and
`delta_action_counts`, plus a `matches_stored_decisions` flag that asserts the
`full` variant reproduces the decisions that were recorded in the JSON.

The ablation re-runs the engine with the **thresholds recorded in the
decision JSON**, not with the library defaults, so the re-run is the same
policy — including the calibrated HI breakpoints of the sensor-derived
index.

### 9.4 Decision-policy sensitivity and operational scenarios

`src/evaluation/decision_sensitivity.py` asks a question the published
results did not answer before: **how much of the decision mix is an
artefact of the hand-set risk-score weights?**

It re-runs the decision engine over a weight space and measures how far the
decisions move.  Two sweeps are used, because they answer different
questions:

| Sweep | What it varies | What it answers |
|---|---|---|
| one-at-a-time | each weight in turn over a grid, the remaining mass redistributed proportionally | "how sensitive is the policy to *this* weight?" |
| Dirichlet | random draws over the simplex | "how much does the decision mix move under a plausible re-weighting?" |

For every perturbation the module reports:

| Metric | Meaning |
|---|---|
| `action_tv` | total-variation distance between the action distribution and the baseline one (0 = identical mix) |
| `action_switch_rate` | fraction of records whose recommended action changed |
| `mean_severity_delta` | mean change of the action's position on the severity ladder (§5.0) |
| `review_rate_delta` | change of the human-review rate |
| `risk_rank_spearman` | Spearman correlation of the baseline and perturbed risk *ranking* of the records |
| `mean_abs_risk_delta` | mean absolute change of the risk score |

The headline of the report is `influence_by_weight`: the largest
`action_tv` each weight can reach on its own.

**Operational scenarios.**  The same script compares the decision mix under
every named scenario of `src/decision_engine/scenarios.py`.  This is also
what makes the constraint layer produce non-zero violation counts in a
recorded run: the certified multi-seed decisions are produced under the
`degraded_operations` scenario (§5.4), so `constraint_violation_count` in
`results/multiseed/downstream_summary.csv` is a real measurement.

Artifacts:

| File | Contents |
|---|---|
| `outputs/reports/decision_sensitivity.json` | the full report: one block per analysed run plus the aggregate |
| `outputs/reports/decision_sensitivity.csv` | one summary row per analysed run |
| `outputs/reports/decision_weight_sweep.csv` | one row per swept weight vector |
| `outputs/reports/decision_constraint_scenarios.csv` | one row per (run, scenario) |

---

## 10. Extension guide

### 10.1 Add a model architecture

1. Create `src/models/<name>.py`. The module must expose a class whose
   `forward` returns the triple `(pred_rul, pred_hi, pred_scale)`.
2. Register it in `src/models/registry.py`:
   - add the class to the imports,
   - add an entry to `ARCHS`,
   - extend `build_model` if the architecture needs extra constructor
     arguments,
   - extend `model_config` so the checkpoint records everything needed to
     rebuild it.
3. Add the name to `ALL_MODELS` (it derives from `ARCHS`, so this is
   automatic) and check that `scripts/decision.py` accepts it — it validates
   against `ALL_MODELS`.
4. Add a regression test in `tests/test_models.py`.

`scripts/train.py`, `scripts/evaluate.py`, `scripts/uncertainty.py` and
`scripts/explain.py` need no changes: they all go through the registry.

### 10.2 Add an interval method

1. Implement the radius rule in `src/uncertainty/conformal.py` and expose an
   interval function.
2. Add the method to the `--method` choices in `scripts/uncertainty.py` and
   to the `run_*` dispatch at the bottom of that script.
3. Write the output to `outputs/results/<subset>_<model>_<variant>.npz` with
   the `lower_<pct>` / `upper_<pct>` / `q_hat_<pct>` keys — `scripts/decision.py`
   reads exactly those.
4. If the method should be selectable by `scripts/decision.py`, add its
   variant name to `UNCERTAINTY_VARIANTS`.

### 10.3 Add an evaluation metric

1. Put the implementation in `src/evaluation/metrics.py` and return a plain
   dict of scalars.
2. Wire it into the relevant dimension in `full_evaluation_report`.
3. Export it from `src/evaluation/__init__.py`.
4. Add a test in `tests/test_evaluation.py`.

Never add a second implementation of an existing metric. The duplicate
NASA-score implementation in the removed `src/models/evaluator.py` was the
reason `scripts/evaluate.py` and `scripts/summarize.py` once reported scores
that differed by a factor of four.

### 10.4 Add a dataset or a subset

1. Drop the raw files into `data/raw/` using the C-MAPSS naming convention.
2. Add `configs/<subset>.yaml` listing only the deviations from
   `DEFAULT_CONFIG`.
3. Add the subset to `ALL_SUBSETS` in `src/utils/config.py`.
4. Run `scripts/preprocess.py --subset <subset>`, then the rest of the chain.

---

## 11. Testing and CI

```powershell
python -m pytest -q          # 177 tests
python -m ruff check src scripts tests   # lint (advisory)
```

`.github/workflows/ci.yml` runs the suite on Linux and Windows across Python
3.9 and 3.12, using a CPU-only PyTorch build. Ruff runs as a separate,
non-blocking step (`continue-on-error: true`), so a lint finding never fails
CI — the tree currently carries style findings (mostly `E402`, from the
deliberate `sys.path` bootstrap before the project imports, and `I001`).
The integration test in `tests/test_preprocessing.py` skips automatically when
`data/raw/` is absent, so CI is green without the dataset.

| Test file | Scope | Needs data |
|---|---|---|
| `tests/test_decision.py` | Decision rules, constraints, risk scoring, HITL review and adaptation | No |
| `tests/test_uncertainty.py` | Conformal quantiles, interval clipping, coverage semantics | No |
| `tests/test_models.py` | Model registry, architecture contracts, checkpoint round-trip, config loading | No |
| `tests/test_xai.py` | Integrated Gradients (no model mutation, determinism, mode preservation) and ERI | No |
| `tests/test_evaluation.py` | The six evaluation dimensions, the ablation re-run, the coverage validator, per-window dropout | No |
| `tests/test_sota_benchmark.py` | Benchmark aggregation, bootstrap CI, significance | No |
| `tests/test_preprocessing.py` | Short test engines are padded, not dropped | **Yes — `data/raw/`** |

---

## 12. Glossary

| Term | Meaning in this repository |
|---|---|
| **RUL** | Remaining useful life, in cycles, capped at `max_rul` (125). |
| **HI** | Health index in `[0, 1]`; 1 = new, 0 = failed. Here `HI = RUL / max_rul`. |
| **Engine** | One turbofan in the C-MAPSS dataset, with its own operating condition and its own degradation trajectory. The unit of splitting. |
| **Window** | A contiguous slice of 40 cycles of sensor history. The unit of prediction. |
| **Nonconformity score** | The quantity a conformal quantile is computed from; here `|y − ŷ|`, optionally normalised by a learned scale. |
| **Calibration split** | An engine-disjoint slice of the *training* set used only to estimate the conformal quantile. |
| **ERI** | Explainability Reliability Index — cross-method agreement between IG, SHAP and temporal relevance (§6). |
| **Temporal relevance** | Per-timestep gradient relevance derived from the same eval-guarded path as Integrated Gradients. |
| **EARA-Conformal** | Engine-Aware Regime-Adaptive conformal prediction — the `engine_joint` method (§4.2). |
| **nex_scp** | Prequential adaptive conformal inference (Gibbs & Candès) — the fifth benchmark method (§4.3). |
| **Risk score** | The continuous `[0, 1]` fusion of RUL, HI, interval width and ERI (§5.2). |
| **Urgency index** | The risk score amplified by the uncertainty term, capped at 1 (§5.2). |
| **HITL** | Human-in-the-loop review of flagged decisions. Refines the policy, never the model. |
| **Ablation** | Re-running the decision engine with a signal removed, and reporting the delta (§9.3). |
