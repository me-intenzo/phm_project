# PHM-XAI — CLI Report

**Project:** Engine-Aware Trustworthy Prognostics (NASA C-MAPSS RUL/HI)  
**Repository:** `D:\phm-xai` · **Version:** 0.1.0 · **Python:** ≥ 3.9  
**Scope:** Every command-line entry point in the project (`main.py`, `scripts/*.py`, and the Streamlit dashboard launcher).

> All commands are intended to be run from the repository root. Every `scripts/*.py` entry point inserts the repository root into `sys.path` before importing `src.*`, so they behave identically regardless of the current working directory. Outputs are written under `outputs/` and `results/` (paths are anchored to the project root).

---

## 0. Conventions shared by almost every CLI

| Convention | Details |
|---|---|
| `--subset` | One of `FD001 FD002 FD003 FD004`, or `all` (iterates every subset; a failing subset is reported and skipped, not aborting the run). `main.py` is hard-coded to `FD001`. |
| `--model` | One of `lstm gru transformer hybrid gru_att_deg gru_att_deg_v2`, or `all`. |
| `--tag` | Selects one specific run of a model (e.g. `ms42`, `seed7`, `noaug`). The tag selects the checkpoint **and** every artifact downstream of it, so decisions are never assembled from a mixture of two different sets of weights. |
| `--config` | Explicit YAML config path (default: `configs/<subset>.yaml`). |
| `--overrides` | Comma-separated `key=value` pairs patched onto the config (e.g. `epochs=200,hidden_size=160`). Coerced to bool/int/float/str. |
| `--seed` | Override the random seed from the config. |
| `--constraint-scenario` | Named operational context: `unconstrained`, `no_maintenance_window`, `tight_safety_margin`, `lead_time_pressure`, `resource_shortage`, `degraded_operations` (certified runs use the last). |
| Operational flags | `--maintenance-window` / `--no-maintenance-window`, `--minimum-safe-rul`, `--maintenance-lead-time`, `--max-allowable-delay`, `--resource-availability` (true/false). Explicit flags override a scenario. |
| Uncertainty variants | `eara_conformal` (default in `decision.py`), `conformal`, `adaptive_conformal`, `cqr`, `engine_joint_conformal`. |
| Explain targets | `rul`, `hi`, `all`. |

Valid subsets: `FD001, FD002, FD003, FD004`. Valid models: `lstm, gru, transformer, hybrid, gru_att_deg, gru_att_deg_v2`.

---

## 1. Exploratory Data Analysis — `main.py`

**Purpose:** EDA entry point for NASA C-MAPSS FD001. Runs dataset validation, summary statistics and sensor visualisations. All outputs go to `outputs/reports/EDA/` and `outputs/figures/EDA/`.

**Invocation:**
```powershell
python main.py
```

**Arguments:** none (subset is hard-coded to `FD001`).

**Outputs:** `outputs/reports/EDA/`, `outputs/figures/EDA/`.

---

## 2. Preprocessing — `scripts/preprocess.py`

**Purpose:** Full preprocessing pipeline: load → validate → summary → visualise → RUL/HI labels → feature selection → scaling → sliding windows → save. Writes `data/processed/<subset>_*.{npy,json,npz}` plus a scaler.

**Invocation:**
```powershell
python scripts/preprocess.py --subset FD001
python scripts/preprocess.py --subset all
python scripts/preprocess.py --subset FD001 --hi-mode rul
```

**Arguments:**

| Flag | Default | Description |
|---|---|---|
| `--subset` | `FD001` | Subset to preprocess, or `all`. |
| `--config` | None | Explicit YAML config path (default `configs/<subset>.yaml`). Supplies `window_size` and `max_rul`. |
| `--window-size` | None | Sliding window length in cycles (defaults to config). |
| `--max-rul` | None | RUL cap for the piecewise-linear RUL label (defaults to config). |
| `--hi-mode` | None | `sensor` (default) sensor-derived HI, or `rul` (legacy `HI = RUL / max_rul`). Defaults to config. |
| `--hi-baseline-cycles` | None | Leading cycles per engine used as the healthy reference for the sensor-derived HI (defaults to config, usually 30). |

**Outputs:** `data/processed/<subset>_{train,test}_{X,y_rul,y_hi,engine_ids}.npy`, `<subset>_metadata.json`, `<subset>_test_frame.npz`; `outputs/models/scaler_<subset>.pkl`; `outputs/figures/<subset>/`, `outputs/reports/<subset>/`.

---

## 3. Training — `scripts/train.py`

**Purpose:** Config-driven training. Rebuilds the model from the YAML config, performs engine-wise train/validation split, augmentation, multi-task loss, warm-up/cosine LR, gradient clipping, weight EMA, early stopping; saves the best checkpoint (EMA weights) + full history.

**Invocation:**
```powershell
python scripts/train.py --model gru_att_deg_v2 --subset FD001
python scripts/train.py --model gru_att_deg_v2 --subset all
python scripts/train.py --model gru_att_deg_v2 --subset FD001 --seed 7 --tag seed7
python scripts/train.py --subset FD001 --overrides epochs=200,hidden_size=160
python scripts/train.py --model gru_att_deg_v2 --subset FD001 --no-augment
```

**Arguments:**

| Flag | Default | Description |
|---|---|---|
| `--model` | `gru_att_deg_v2` | Architecture to train, or `all`. |
| `--subset` | `FD001` | Subset to train, or `all`. |
| `--config` | None | Explicit YAML config path. |
| `--seed` | None | Override the random seed from the config. |
| `--tag` | `""` | Suffix for the checkpoint directory (e.g. `seed7`). |
| `--epochs` | None | Override the maximum number of epochs. |
| `--lr` | None | Override the learning rate. |
| `--no-augment` | false | Disable window augmentation. |
| `--no-ema` | false | Disable the exponential moving average of the weights. |
| `--no-deterministic` | false | Disable deterministic torch/CUDA kernels (run stays seeded). |
| `--overrides` | None | Additional config overrides as comma-separated `key=value` pairs. |

**Outputs:** `outputs/checkpoints/<model>[_tag]/<subset>/best_model.pt`; `outputs/results/<subset>_<model>[_tag]_history.npz` (via evaluate).

---

## 4. Evaluation — `scripts/evaluate.py`

**Purpose:** Rebuilds the model from the checkpoint's stored `model_config`, runs inference on the test set, computes predictive accuracy (RUL MAE/RMSE/R²/NASA score, HI MAE/RMSE/R²) and writes prediction NPZs.

**Invocation:**
```powershell
python scripts/evaluate.py --model gru_att_deg_v2 --subset FD001
python scripts/evaluate.py --model gru_att_deg_v2 --subset all
python scripts/evaluate.py --model gru_att_deg_v2 --subset FD001 --tag seed7
```

**Arguments:**

| Flag | Default | Description |
|---|---|---|
| `--model` | `gru_att_deg_v2` | Architecture to evaluate, or `all`. |
| `--subset` | `FD001` | Subset to evaluate, or `all`. |
| `--tag` | `""` | Suffix of the checkpoint directory. |

**Outputs:** `outputs/results/<subset>_<model>[_tag]_predictions.npz`, `<subset>_<model>[_tag]_metrics.json`.

---

## 5. Uncertainty / conformal intervals — `scripts/uncertainty.py`

**Purpose:** Conformal uncertainty estimation. Reproduces the engine-wise train/validation split from the checkpoint seed, calibrates conformal quantiles on the validation split, predicts test RUL, generates prediction intervals, evaluates coverage/width. Test targets are never used during calibration.

**Invocation:**
```powershell
python scripts/uncertainty.py --model gru --subset FD001 --method split
python scripts/uncertainty.py --model gru --subset FD001 --method cqr
python scripts/uncertainty.py --model gru --subset FD001 --method adaptive
python scripts/uncertainty.py --model gru --subset FD001 --method engine_joint
python scripts/uncertainty.py --model gru_att_deg_v2 --subset FD001 --tag ms42 --method engine_joint
```

**Arguments:**

| Flag | Default | Description |
|---|---|---|
| `--model` | `gru` | Model architecture whose checkpoint is evaluated (choices: registered models). |
| `--subset` | `FD001` | `FD001..FD004`. |
| `--method` | `split` | `split`, `cqr`, `adaptive`, `engine_joint`. |
| `--tag` | `""` | Checkpoint tag (selects a specific run). |

**Outputs:** `outputs/results/<subset>_<model>_<method_suffix>[_tag].npz` where suffixes are `conformal` (split), `cqr`, `adaptive_conformal`, `engine_joint_conformal`; plus logs under `outputs/logs/uncertainty/`.

---

## 6. Explainability — `scripts/explain.py`

**Purpose:** Comprehensive XAI pipeline: Integrated Gradients, Kernel SHAP, temporal relevance, ERI, degradation trajectory plots, HTML report.

**Invocation:**
```powershell
python scripts/explain.py --model gru_att_deg --subset FD001 --target rul
python scripts/explain.py --model all --subset all --target all
python scripts/explain.py --model gru_att_deg_v2 --subset FD001 --target all --tag ms42
```

**Arguments:**

| Flag | Default | Description |
|---|---|---|
| `--subset` | `FD001` | Subset, or `all`. |
| `--model` | `gru_att_deg` | Model name, or `all`. |
| `--target` | `rul` | `rul`, `hi`, `all`. |
| `--samples` | 32 | Number of explained test samples. |
| `--ig-steps` | 50 | Integrated Gradients steps. |
| `--engine-id` | 1 | Engine index (0-based) for degradation trajectory plots. |
| `--shap-nsamples` | 256 | SHAP kernel samples. |
| `--tag` | `""` | Checkpoint tag. |

**Outputs:** `outputs/xai/<subset>/<model>[_tag]/` — `xai_<target>.npz`, `eri_<target>.json`, PNGs, `degradation_engine{id}_{model}.png`; plus `outputs/xai/<subset>/xai_report_<subset>_{rul,hi}.html`, `eri_radar_<target>.png`.

---

## 7. Decision engine — `scripts/decision.py`

**Purpose:** O4 explainable decision intelligence. Fuses RUL, HI, the conformal interval and ERI into a risk score, classifies health/uncertainty, selects an action on the severity ladder, then applies operational constraints. Consumes upstream artifacts directly and names every missing artifact.

**Invocation:**
```powershell
python scripts/decision.py --subset all
python scripts/decision.py --subset FD001 --model hybrid --uncertainty-variant engine_joint_conformal
python scripts/decision.py --subset FD001 --model gru_att_deg_v2 --tag ms42
python scripts/decision.py --subset FD001 --constraint-scenario degraded_operations
python scripts/decision.py --subset FD001 --minimum-safe-rul 10 --maintenance-lead-time 10 --no-maintenance-window
```

**Arguments:**

| Flag | Default | Description |
|---|---|---|
| `--subset` | `all` | Subset, or `all` (also accepts lowercase). |
| `--model` | `gru` | Model, or `all`. |
| `--uncertainty-variant` | `eara_conformal` | Calibrated uncertainty artifact to use. |
| `--tag` | `""` | Checkpoint tag. |
| `--xai-context` | None | Optional JSON file with `top_k_sensors` per record from O3. |
| `--maintenance-window` | false | Scheduled maintenance can be performed now. |
| `--no-maintenance-window` | false | No maintenance window available. |
| `--minimum-safe-rul` | None | Minimum RUL (cycles) to defer maintenance. |
| `--maintenance-lead-time` | None | Cycles required before maintenance can begin. |
| `--max-allowable-delay` | None | Longest wait the operator will accept (cycles). |
| `--resource-availability` | None | `true`/`false` — whether crew and parts are available. |
| `--constraint-scenario` | `""` | Named scenario (see §0). |
| `--thresholds` | None | Optional JSON file with policy threshold overrides (e.g. `{"w_rul":0.5,"rul_critical":8.0}`). |

**Outputs:** `outputs/results/decision_<subset>_<model>[_tag].json` (+ scenario variants), `outputs/reports/*.html`.

---

## 8. HITL review — `scripts/hitl.py`

**Purpose:** O5 human-in-the-loop. Consumes a `decision.py` JSON, creates review items for decisions flagged `human_review`, writes JSON results + HTML report + log, then applies conservative policy refinement from accumulated expert feedback.

**Invocation:**
```powershell
python scripts/hitl.py --subset FD001 --model gru
python scripts/hitl.py --subset all --model all
python scripts/hitl.py --subset FD001 --model gru_att_deg_v2 --tag ms42
python scripts/hitl.py --input outputs/results/decision_FD001_gru.json --review-id decision-... --reviewer operator-1 --action APPROVE --confidence 0.9 --reason "Reviewed by maintenance control"
```

**Arguments:**

| Flag | Default | Description |
|---|---|---|
| `--subset` | `FD001` | `FD001..FD004` or `all`. |
| `--model` | `gru` | Model name, or `all`. |
| `--tag` | `""` | Checkpoint tag. |
| `--input` | None | O4 decision JSON output. |
| `--review-id` | None | Review item to resolve. |
| `--reviewer` | None | Human reviewer identifier. |
| `--action` | None | `APPROVE` or `OVERRIDE`. |
| `--override-action` | None | Alternative maintenance action for `OVERRIDE` (must differ from the AI action). |
| `--confidence` | None | Expert confidence in `[0,1]`. |
| `--reason` | `""` | Reason/comment. |

Resolving a review requires `--review-id`, `--reviewer`, `--action` and `--confidence` together. `OVERRIDE` additionally requires `--override-action`.

**Outputs:** `outputs/results/hitl_<subset>_<model>[_tag].json`, `outputs/reports/*.html`, `outputs/hitl/feedback.jsonl` (append-only).

---

## 9. Ensemble — `scripts/ensemble.py`

**Purpose:** Uniform ensemble over several checkpoints (no weights fitted on the test set). Averages predictions across members.

**Invocation:**
```powershell
python scripts/ensemble.py --subset FD001 --models gru_att_deg_v2@noaug,gru_att_deg_v2@seed7
python scripts/ensemble.py --subset all --models gru_att_deg_v2,gru_att_deg,gru
```

**Arguments:**

| Flag | Default | Description |
|---|---|---|
| `--subset` | `FD001` | Subset, or `all`. |
| `--models` | `gru_att_deg_v2,gru_att_deg,gru` | Comma-separated members; use `name` or `name@tag`. |
| `--name` | `ensemble` | Stem used for the saved prediction/metric files. |
| `--device` | None | Force `cpu` or `cuda` (default: auto). |

**Outputs:** `outputs/results/<subset>_<name>_predictions.npz` (and metrics).

---

## 10. Summarise accuracy — `scripts/summarize.py`

**Purpose:** Aggregate every saved `*_predictions.npz` into one accuracy table, recomputing metrics from stored arrays. Answers "is the RUL RMSE target met?".

**Invocation:**
```powershell
python scripts/summarize.py
python scripts/summarize.py --threshold 13
python scripts/summarize.py --models gru_att_deg_v2,ensemble
```

**Arguments:**

| Flag | Default | Description |
|---|---|---|
| `--threshold` | 13.0 | Target mean RUL RMSE. |
| `--models` | None | Comma-separated model stems to include (default: every prediction file found). |

**Outputs:** `outputs/reports/accuracy_summary.csv` (printed to stdout too).

---

## 11. SOTA benchmark — `scripts/sota_benchmark.py`

**Purpose:** Reproducible SOTA-benchmark evidence: engine-level paired bootstrap CIs for the four-subset mean RUL RMSE, paired bootstrap tests vs. a reference model, error profiles/bands.

**Invocation:**
```powershell
python scripts/sota_benchmark.py
python scripts/sota_benchmark.py --reference gru_att_deg_v2 --resamples 20000
python scripts/sota_benchmark.py --models gru_att_deg_v2,ensemble,gru_att_deg
```

**Arguments:**

| Flag | Default | Description |
|---|---|---|
| `--models` | `gru_att_deg_v2,ensemble,gru_att_deg,gru,lstm,hybrid,transformer` | Comma-separated model stems to compare. |
| `--reference` | `gru_att_deg_v2` | Reference model for paired tests. |
| `--resamples` | 10000 | Number of bootstrap resamples. |
| `--seed` | 42 | Bootstrap RNG seed. |

**Outputs:** `outputs/reports/sota_benchmark.csv`, `sota_bootstrap_ci.csv`, `sota_significance.csv`, `sota_error_profile.csv`, `sota_error_bands.csv`.

---

## 12. Multi-seed training — `scripts/multiseed_train.py`

**Purpose:** Certification harness. Trains the deliverable architecture across a grid of seeds and subsets, records an auditable manifest (seed, resolved config, git commit, SHA-256 of every checkpoint, determinism config). No proxy model.

**Invocation:**
```powershell
python scripts/multiseed_train.py --model gru_att_deg_v2 --subsets all --seeds 42 43 44 45 46
python scripts/multiseed_train.py --subsets all --force
```

**Arguments:**

| Flag | Default | Description |
|---|---|---|
| `--model` | `gru_att_deg_v2` | Architecture to train across seeds. |
| `--subsets` | `["all"]` | Subsets to cover, or `all`. |
| `--seeds` | `42 43 44 45 46` | Seeds to train. |
| `--tag-prefix` | `ms` | Prefix of the per-run tag (default `ms` → tag `ms42`). |
| `--overrides` | None | Config overrides applied to every run, comma-separated `key=value`. |
| `--force` | false | Retrain even when checkpoint + metrics already exist. |
| `--allow-single-seed` | false | Allow a one-seed run (smoke tests only; recorded as such in the manifest). |

**Outputs:** `outputs/checkpoints/<model>_ms<seed>/<subset>/best_model.pt`; `outputs/results/<subset>_<model>_ms<seed>_metrics.json`; `results/multiseed/training_manifest.json`.

---

## 13. Multi-seed pipeline — `scripts/multiseed_pipeline.py`

**Purpose:** Runs the downstream stages (uncertainty, XAI, decision, HITL) on every (subset, seed) of the multi-seed family, with a consistency check between the uncertainty-stage and evaluation-stage point predictions (tolerance 0.1 RUL cycles, r > 0.9999).

**Invocation:**
```powershell
python scripts/multiseed_pipeline.py --subsets all --seeds 42 43 44 45 46 --stages all
```

**Arguments:**

| Flag | Default | Description |
|---|---|---|
| `--model` | `gru_att_deg_v2` | Model architecture. |
| `--subsets` | `["all"]` | Subsets, or `all`. |
| `--seeds` | `42 43 44 45 46` | Seeds. |
| `--tag-prefix` | `ms` | Tag prefix. |
| `--stages` | `["all"]` | Stages to run: `uncertainty`, `xai`, `decision`, `hitl`, or `all`. |
| `--xai-samples` | None | Number of explained samples (default: script default). |
| `--constraint-scenario` | `degraded_operations` | Operational-constraint scenario applied by the decision stage. |

**Outputs:** `results/multiseed/pipeline_manifest.json` plus all per-run artifacts.

---

## 14. Multi-seed downstream summary — `scripts/multiseed_downstream.py`

**Purpose:** Summarises the downstream artifacts of the multi-seed grid: ERI of both XAI targets, decision action mix, review rate, constraint violations, interval width consumed by the decision engine, HITL counts.

**Invocation:**
```powershell
python scripts/multiseed_downstream.py
python scripts/multiseed_downstream.py --model gru_att_deg_v2 --subsets FD001 FD002 --seeds 42 43
```

**Arguments:**

| Flag | Default | Description |
|---|---|---|
| `--model` | `gru_att_deg_v2` | Architecture whose downstream artifacts are summarised. |
| `--subsets` | None | Subsets (default: every subset in the pipeline manifest). |
| `--seeds` | None | Seeds (default: every seed in the pipeline manifest). |

**Outputs:** `results/multiseed/downstream_per_run.csv`, `downstream_summary.csv`.

---

## 15. Multi-seed conformal benchmark — `scripts/multiseed_conformal_benchmark.py`

**Purpose:** Engine-disjoint comparison of five interval methods (`split_conformal`, `cqr`, `adaptive`, `nex_scp`, `eara_conformal`) across FD001–FD004 and seeds 42–46, on the real checkpoints. Reproduces each checkpoint's own engine hold-out from the stored seed.

**Invocation:**
```powershell
python scripts/multiseed_conformal_benchmark.py --subsets all --seeds 42 43 44 45 46
python scripts/multiseed_conformal_benchmark.py --calibration-units window
```

**Arguments:**

| Flag | Default | Description |
|---|---|---|
| `--model` | `gru_att_deg_v2` | Model architecture to benchmark. |
| `--subsets` | `["all"]` | `FD001..FD004` or `all`. |
| `--seeds` | `42 43 44 45 46` | Seeds. |
| `--tag-prefix` | `ms` | Tag prefix used by `multiseed_train.py`. |
| `--batch-size` | 64 | Batch size. |
| `--n-regimes` | 3 | Number of k-means regimes. |
| `--aci-gamma` | 0.02 | ACI adaptation rate. |
| `--calibration-units` | `window engine` | `window` and/or `engine`. |

**Outputs:** `results/conformal_benchmark/` — `per_engine_intervals.csv`, `per_seed_metrics.csv`, `summary_mean_std.csv`, `paired_comparisons.csv`, `split_manifest.json`, `run_config.json`, coverage/width plots.

---

## 16. Multi-seed report — `scripts/multiseed_report.py`

**Purpose:** Aggregate the multi-seed grid into mean ± std, paired tests, literature comparison and a human-readable report.

**Invocation:**
```powershell
python scripts/multiseed_report.py
python scripts/multiseed_report.py --model gru_att_deg_v2 --seeds 42 43 44 45 46 --subsets all
```

**Arguments:**

| Flag | Default | Description |
|---|---|---|
| `--model` | `gru_att_deg_v2` | Model whose multi-seed runs are aggregated. |
| `--tag-prefix` | `ms` | Tag prefix used by `multiseed_train.py`. |
| `--seeds` | None | Seeds to aggregate (default: read from the training manifest). |
| `--subsets` | None | Subsets to aggregate (default: all four). |
| `--reference` | `results/multiseed/literature_reference.json` | Published reference values (JSON). |

**Outputs:** `results/multiseed/per_seed_metrics.csv`, `summary_mean_std.csv`, `aggregate_by_seed.csv`, `literature_comparison.csv`, `certification.json`; `outputs/reports/multiseed_report.md`.

---

## 17. Regime sweep — `scripts/regime_sweep.py`

**Purpose:** Offline sweep of the k-means regime count `k ∈ {2,3,4,5,6,7,8,10}` for EARA-Conformal across all four subsets, using the GRU model. Composite score = mean coverage error + mean PINAW (lower is better).

**Invocation:**
```powershell
python scripts/regime_sweep.py
```

**Arguments:** none (constants are hard-coded: GRU, seed 42, all four subsets).

**Outputs:** `outputs/results/regime_sweep.json`.

---

## 18. Decision sensitivity — `scripts/decision_sensitivity.py`

**Purpose:** Measures how much of the published decision mix the hand-set risk-score weights (`w_rul, w_hi, w_u, w_eri`) actually determine. Re-runs the decision engine over a one-at-a-time sweep and a Dirichlet sweep of the simplex, plus a comparison across named constraint scenarios.

**Invocation:**
```powershell
python scripts/decision_sensitivity.py --subsets all --seeds 42 43 44 45 46
python scripts/decision_sensitivity.py --decision outputs/results/decision_FD001_gru_att_deg_v2_ms42.json
```

**Arguments:**

| Flag | Default | Description |
|---|---|---|
| `--subsets` | `FD001 FD002 FD003 FD004` | Subsets to analyse. |
| `--seeds` | `42 43 44 45 46` | Seeds to analyse. |
| `--model` | `gru_att_deg_v2` | Architecture whose decisions are analysed. |
| `--tag-prefix` | `ms` | Tag prefix of the multi-seed family. |
| `--decision` | None | Analyse a single decision artifact (path used verbatim). |
| `--dirichlet-samples` | 200 | Number of Dirichlet weight draws per run. |
| `--dirichlet-concentration` | 5.0 | Dirichlet concentration (higher = closer to uniform). |
| `--seed` | 42 | Seed of the Dirichlet sampler. |

**Outputs:** `outputs/reports/decision_sensitivity.json`, `decision_sensitivity.csv`, `decision_weight_sweep.csv`, `decision_constraint_scenarios.csv`.

---

## 19. Seed-alignment verification — `scripts/verify_seed_alignment.py`

**Purpose:** Demonstrates that the conformal calibration split follows the checkpoint seed (not a hard-coded 42) and that calibration sets are disjoint from training sets across seeds.

**Invocation:**
```powershell
python scripts/verify_seed_alignment.py
```

**Arguments:** none.

**Outputs:** stdout only (asserts pass/fail).

---

## 20. End-to-end demo — `scripts/demo.py`

**Purpose:** Runs the entire pipeline in one shot — preprocess → train → evaluate → EARA-Conformal uncertainty → explain → decision → HITL — by invoking the same CLI entry points, then verifies every key artifact. Supports per-stage skip flags and operational-constraint passthroughs.

**Invocation:**
```powershell
python scripts/demo.py
python scripts/demo.py --subset all
python scripts/demo.py --eda
python scripts/demo.py --subset FD001 --model gru --skip-explain --skip-hitl
```

**Arguments:**

| Flag | Default | Description |
|---|---|---|
| `--subset` | `FD001` | Subset, or `all`. |
| `--model` | `gru` | Architecture (choices: `lstm, gru, transformer, hybrid` — `gru_att_deg`/`gru_att_deg_v2` are not wired into the demo). |
| `--eda` | false | Run the EDA entry point (`main.py`) first (FD001 only). |
| `--skip-preprocess` / `--skip-train` / `--skip-evaluate` / `--skip-uncertainty` / `--skip-explain` / `--skip-decision` / `--skip-hitl` | false | Skip a stage. |
| `--maintenance-window` / `--no-maintenance-window` | false | Forwarded to `decision.py`. |
| `--minimum-safe-rul` | None | Forwarded to `decision.py`. |
| `--maintenance-lead-time` | None | Forwarded to `decision.py`. |
| `--max-allowable-delay` | None | Forwarded to `decision.py`. |
| `--resource-availability` | None | `true`/`false`, forwarded to `decision.py`. |
| `--constraint-scenario` | `""` | Named scenario forwarded to `decision.py`. |

**Outputs:** all artifacts produced by the stages it runs (see §2–§8).

---

## 21. Streamlit dashboard — `app.py`

**Purpose:** Operator-facing dashboard (5 tabs: Model, Uncertainty, XAI, Decision, HITL). Never recomputes the pipeline; reads CLI artifacts only. The only write is expert feedback via `src.hitl`.

**Invocation:**
```powershell
pip install -e ".[dashboard]"
streamlit run app.py
```

**Arguments:** none (run discovery and selection happen in the sidebar; subset/model/run-tag/uncertainty-variant/coverage/XAI target are chosen in the UI).

**Outputs:** none beyond reading artifacts; appends to `outputs/hitl/feedback.jsonl` when a review is submitted.

---

## Quick reference — common one-liners

```powershell
python main.py
python scripts/preprocess.py --subset all
python scripts/train.py --model gru_att_deg_v2 --subset FD001
python scripts/evaluate.py --model gru_att_deg_v2 --subset FD001
python scripts/uncertainty.py --model gru --subset FD001 --method engine_joint
python scripts/explain.py --model gru_att_deg --subset FD001 --target rul
python scripts/decision.py --model gru --subset FD001
python scripts/hitl.py --subset FD001 --model gru
python scripts/demo.py
streamlit run app.py
```

---

*Generated from the repository at `D:\phm-xai`. Argument lists mirror the `parse_args()` of each entry point; defaults and choices are taken verbatim from the source.*
