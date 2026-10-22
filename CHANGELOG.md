# Changelog

All notable changes to PHM-XAI are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Sensor-derived health index** — `src/preprocessing/health_index.py` builds
  the HI label from the *sensor signal* instead of from the RUL label. The
  construction removes the operating conditions (per-sensor least squares on
  the three operational settings, fitted on the training split only),
  subtracts each engine's own healthy reference (its first 30 cycles), fits a
  one-component PLS direction against the training RUL label, and maps the
  projection onto `[0, 1]` with reference quantiles from the training windows.
  The index of a row is computed from the *same trailing window the model
  sees*, so a window's label is a function of exactly that window's features.
  `corr(HI, RUL)` on the training split is 0.890 / 0.674 / 0.797 / 0.705 on
  FD001–FD004 — strongly related to the degradation state, but no longer a
  deterministic function of the RUL label. The legacy `HI = RUL / max_rul`
  label is still reachable with `scripts/preprocess.py --hi-mode rul`.
- **Calibrated HI breakpoints** — `calibrate_hi_thresholds()` derives the
  decision policy's three HI breakpoints from the training split (each is the
  median index of the windows in the corresponding RUL band). They are stored
  in the dataset metadata as `hi_thresholds` and merged into the policy
  thresholds by `scripts/decision.py`, so the HI criterion fires at the same
  degradation levels as the RUL criterion.
- **Severity ladder** — `src/decision_engine/rules.py` defines
  `SEVERITY_LADDER` (`CONTINUE_OPERATION < MONITOR < INSPECT <
  SCHEDULE_MAINTENANCE < URGENT_MAINTENANCE`) and computes the candidate
  action from it instead of from a hand-written table:
  `severity = min(base(planning_state) + eri_step, top)`. The policy is now
  monotone **by construction**: a wider prediction interval can only make
  the planning state more severe, and a less reliable explanation can only
  escalate the action. This fixes the `AT_RISK` row that used to produce
  `INSPECT` for `HIGH` uncertainty and `SCHEDULE_MAINTENANCE` for
  `MODERATE` uncertainty. Every decision record now carries
  `action_severity` and `planning_state`.
- **The action is planned on the conservative end of the calibrated
  interval.** The *planning state* is the health state the engine would be
  in if the prediction were at the lower bound of its 90 % interval, and
  the action is the base action of that state. The point estimate's state
  is still reported as `health_state`. A maintenance work order has to be
  scheduled so that it is carried out before the worst credible case, not
  before the most likely one — and the interval's *width* is handled where
  it belongs: the human-review flag and the operational constraints
  (`maintenance_lead_time`).
- **Named operational-constraint scenarios** —
  `src/decision_engine/scenarios.py` defines `unconstrained`,
  `no_maintenance_window`, `tight_safety_margin`, `lead_time_pressure`,
  `resource_shortage` and `degraded_operations`.
  `scripts/decision.py --constraint-scenario <name>` applies one (explicit
  constraint flags still override individual values) and
  `scripts/multiseed_pipeline.py --constraint-scenario` records the scenario
  that produced each decision artifact. The certified runs use
  `degraded_operations`, so the constraint layer is exercised by the
  published results instead of only by unit tests.
- **Decision-policy sensitivity analysis** —
  `src/evaluation/decision_sensitivity.py` and
  `scripts/decision_sensitivity.py` re-run the decision engine over a weight
  space (one-at-a-time plus a Dirichlet sweep over the simplex) and report how
  far the action mix, the review rate and the risk *ranking* move. The same
  script compares the decision mix under every constraint scenario. Outputs:
  `outputs/reports/decision_sensitivity.{json,csv}`,
  `decision_weight_sweep.csv` and `decision_constraint_scenarios.csv`.
- **Tests** — `tests/test_health_index.py` (construction, generalisation to
  unseen engines, row-order preservation, save/load round trip, breakpoint
  calibration), `tests/test_decision_policy.py` (severity ladder, exhaustive
  monotonicity over 4 states × 3 uncertainty levels × 3 ERI levels, the
  constraint scenarios), `tests/test_decision_sensitivity.py` (weight sweeps,
  the report structure) and `tests/test_losses.py` (the HI scale factor).

### Changed

- **Tier 2 — duplicated infrastructure removed.** The conformal layer, the
  model layer, the evaluation layer and the preprocessing layer each had
  several copies of the same logic, with divergent values. What changed:

  - **One conformal quantile.** `src/uncertainty/conformal.py` holds the only
    implementation of the finite-sample quantile; `signed_conformal_quantile`
    (CQR) is a wrapper over the same code, and the CLI/benchmark wrappers
    delegate to it. `src/uncertainty/calibration.py` and
    `src/uncertainty/adaptive.py` — pass-through modules whose logic now
    lives in `conformal.py` — are **removed**; `calibrate_eara_levels` and
    `calibrate_multiple_levels` moved into `conformal.py`.
  - **One scale floor, applied as a bound.** `SCALE_FLOOR` (1e-3) is declared
    once, in `src/uncertainty/conformal.py`, and is applied with `max(...)`
    everywhere. The previous mix — an *additive* 1e-6 floor in the library
    and a `max` 1e-3 floor in the pipeline — let the calibration score and
    the interval radius use different scales.
  - **One interval builder.** Every interval (split, adaptive, CQR, EARA,
    engine-joint, prequential ACI) is clipped to the RUL domain and ordered
    by `clip_interval()`.
  - **One interval-metric implementation.** `src/evaluation/metrics.py` now
    calls `src/uncertainty/coverage.py` instead of re-implementing coverage,
    MPIW, PINAW, Winkler and engine-level coverage with a hard-coded 125.0
    bound and its own key names. MAE/RMSE/R²/NASA live in
    `src/evaluation/metrics.py`; `scripts/sota_benchmark.py` imports them
    instead of carrying its own copies.
  - **One model-output unpacking.** `src/models/outputs.py::unpack_model_output`
    replaces the five ad-hoc unpacking patterns in the XAI modules, the
    uncertainty pipeline, the trainer and the evaluation layer.
  - **One model loader.** `src/models/registry.py::load_model_from_checkpoint`
    replaces the divergent `load_model()` copies in `scripts/evaluate.py`,
    `scripts/explain.py` and `scripts/uncertainty.py`.
  - **Shared model building blocks.** `src/models/heads.py` holds the
    multi-task heads and `src/models/layers.py` the sinusoidal positional
    encoding (previously copy-pasted into six architectures and two files
    respectively). The module and attribute names are unchanged, so existing
    checkpoints still load.
  - **One training loop.** `ModelTrainer.train_epoch` and
    `ModelTrainer.validate_epoch` share a single `_run_epoch()`.
  - **One HITL aggregation.** `src/hitl/updater.py::aggregate_expert_actions`
    computes the action support, the weighted support and the consensus ratio
    for both the policy profile and the recommendation refiner.
  - **One feature-column definition.** `src/preprocessing/loader.py::get_feature_columns`
    replaces the copies in the window generator, the scaler and the feature
    selector.
  - **One `to_numpy` and one importance normalisation.**
    `src/utils/math_utils.py` holds both; the zero-total case is now defined
    once (an all-zero vector) instead of returning the raw magnitudes in the
    ERI module and zeros in the visualisation module.
  - **One set of enums and utilities.** `ALL_MODELS` comes from
    `src/models/registry.py` and `ALL_SUBSETS` from `src/utils/config.py`
    (the demo used to reject the headline model `gru_att_deg_v2`);
    `get_device()` lives in `src/utils/config.py`, `set_seed()` in
    `src/utils/seed.py`, `PROJECT_ROOT` in `src/utils/paths.py` and
    `VALID_ACTIONS` in `src/utils/config.py`.
  - **The XAI stage reads the processed artifacts.** `scripts/preprocess.py`
    writes the processed test frame to `data/processed/<subset>_test_frame.npz`
    and `scripts/explain.py` builds its degradation windows from that
    artifact, instead of re-running the whole preprocessing pipeline (which
    produced a *different* health index than the one the model was trained
    on).
  - **The recommender passes the full payload.** `recommend_action()` no
    longer projects the decision onto a hand-written list of twenty keys, so
    a field added upstream is no longer silently dropped.
- **The HI loss term is put on the RUL scale.** `MultiTaskLoss` takes a
  `hi_scale` factor and the pipeline passes `max_rul ** 2`
  (`hi_loss_scale: rul_max`), so the bounded HI loss enters the total in
  cycle-equivalent units. Previously the HI term moved the total loss by
  ~1e-5 in relative terms and `hi_loss_weight` was effectively inert.
- **The decision engine reports its severity** and the constraint layer
  records `constraint_substitutions` and `severity_delta` alongside the
  existing violation and adjustment lists. The maintenance-window rule is now
  documented as the single *substitution* (an infeasible commitment replaced
  by the nearest feasible one) rather than an escalation.
- **The decision JSON carries the effective thresholds** (`thresholds`) and
  the `constraint_scenario` that produced it, so the ablation and the
  sensitivity analysis re-run the *same* policy instead of the library
  defaults.
- `scripts/preprocess.py` writes the health-index model to
  `outputs/models/health_index_<subset>.pkl` and records `hi_mode`,
  `hi_baseline_cycles`, `health_index` and `hi_thresholds` in the dataset
  metadata. `scripts/train.py` refuses to train when the stored `hi_mode`
  disagrees with the config, exactly as it does for `max_rul`.

### Fixed

- **The `AT_RISK` policy row is monotone in the uncertainty** (see the
  severity ladder above). The action is now planned on the conservative end
  of the calibrated interval, so `INSPECT` is structurally unreachable from
  the `AT_RISK` band; an at-risk engine is planned as
  `SCHEDULE_MAINTENANCE`, or as `URGENT_MAINTENANCE` when its interval
  reaches into the critical band. The `DEGRADING` and `CRITICAL` base
  actions are unchanged.
- **The HI label is no longer a deterministic function of RUL**, and the
  reported HI metrics measure a real estimation task instead of echoing the
  RUL label.
- **README staleness** — the "Implementation Status" section no longer claims
  that the conformal benchmark keeps a private clamped copy of the quantile
  (it calls the shared, unclamped implementation), that the downstream
  artifacts are not tag-aware (they are), or that the multi-seed benchmark
  does not seed deterministically (it does).

### Added

- **`src/utils/provenance.py`** — a provenance stamp for every generated
  artifact. The stamp records the SHA-256 digest of the checkpoint the
  artifact was produced from, plus the seed, epoch, architecture, git commit
  and a UTC timestamp, and is embedded **inside** the `.npz` (under the
  `meta_json` key, as a 0-d unicode array so `np.load` reads it without
  `allow_pickle=True`). Embedding it in the artifact rather than writing a
  sidecar means the stamp can never drift away from the data it describes.
  `scripts/evaluate.py` and all four `scripts/uncertainty.py` methods now
  write it. `sha256_file()` and `git_commit()` moved here from
  `scripts/multiseed_train.py` so a file digest has exactly one
  implementation in the repository.
- **Artifact-pairing checks.** `src/evaluation/benchmark.py::comparative_benchmark`
  compares the checkpoint declared by the predictions artifact with the one
  declared by the intervals artifact and reports a `provenance` block per
  (subset, model): `ok`, `mismatch` or `unknown`. A mismatch carries a
  warning explaining that the two artifacts came from different weights; with
  `strict_provenance=True` it raises `ValueError` instead.
- **Tests** — `tests/test_provenance.py` (stamp round trip without pickle,
  corrupt-stamp tolerance, the three comparison statuses, the tagged summary
  lookup, strict mode, and the decision engine's artifact preference).

- **Multi-seed certification harness** — `scripts/multiseed_train.py` trains the
  deliverable architecture across seeds × subsets and records an auditable
  manifest (seed, resolved config, git commit, checkpoint SHA-256, determinism
  block); `scripts/multiseed_report.py` aggregates it into per-seed,
  mean ± std, four-subset aggregate and literature-comparison CSVs plus
  `results/multiseed/certification.json` and
  `outputs/reports/multiseed_report.md`; `scripts/multiseed_pipeline.py`
  drives uncertainty → XAI → decision → HITL for every seed and verifies that
  the artifacts describe the same weights. Runs are tagged `ms<seed>`.
- **`src/utils/paths.py`** — the single place that spells a run tag into a
  checkpoint, flat-artifact or structured-artifact path. Untagged paths are
  byte-identical to the historical ones, so every existing artifact and
  consumer keeps working.
- **`--tag` on every downstream stage** — `scripts/uncertainty.py`,
  `scripts/explain.py`, `scripts/decision.py` and `scripts/hitl.py` now all
  read *and write* tag-scoped artifacts, so two seeds of the same model never
  overwrite each other. The tag travels inside the decision JSON, so HITL
  derives its output stem from the artifact it was given.
- **`signed_conformal_quantile`** in `src/uncertainty/conformal.py` — the CQR
  nonconformity score is signed (negative when the truth falls inside the
  predicted quantile band), so it needs the unclamped rank without the
  non-negativity check.
- **`scripts/verify_seed_alignment.py`** — asserts that each checkpoint's
  calibration split is reproduced from the seed stored *inside* that
  checkpoint, and that different seeds yield different hold-outs.
- **Tests** — `tests/test_multiseed.py` (conformal rank, signed scores,
  engine-level reduction, tag-aware paths) and
  `tests/test_multiseed_report.py` (aggregation and the seed-noise verdict).
- **Two calibration units in the conformal benchmark.** `compute_all_intervals`
  now takes `calibration_units` and returns `results[unit][method][coverage]`.
  The benchmark computes the classic **window** unit (one score per calibration
  window; matches the scoring unit of the test set) and the conservative
  **engine** unit (one maximum score per engine) in a single inference pass, and
  writes a `calibration_unit` column everywhere. The engine unit was previously
  the only one reported, and it over-covered massively (0.99 at an 80 % target
  on FD003) because a lifetime maximum is much larger than any single test
  window. `--calibration-units` selects the set; `window` is the primary unit.
- **The downstream chain is certified for every seed, not only for one.**
  `scripts/multiseed_pipeline.py` now runs the four stages
  (uncertainty → XAI → decision → HITL) for all 20 (subset, seed) pairs and
  records the result of each run — stage status, timings, artifact paths, the
  cross-stage prediction check and any problem — in
  `results/multiseed/pipeline_manifest.json`. The XAI stage explains **both**
  targets (`--target all`), so `eri_hi.json` exists for every seed as well, and
  the artifact check covers it.
- **`scripts/multiseed_downstream.py`** — summarises the downstream artifacts
  of the certified grid: the ERI of both targets, the decision engine's action
  mix, health-state and uncertainty mix, review rate, constraint violations,
  risk and urgency, the width of the interval the decision engine actually
  consumed, and the HITL review/feedback counts. Writes
  `results/multiseed/downstream_per_run.csv` (one row per subset × seed) and
  `results/multiseed/downstream_summary.csv` (mean ± std across seeds, per
  subset and for the four-subset aggregate). A missing artifact is reported as
  missing, never replaced by a default.
- **The certification report covers the downstream chain.** `multiseed_report.py`
  renders a "Downstream chain on the same checkpoints" section (runs executed,
  runs clean, wall-clock, and the per-subset ERI / decision / interval-width
  table) and `certification.json` gains a `downstream_chain` block listing any
  run that produced an error or an artifact-consistency problem.

### Fixed

- **The comparative benchmark's summary was empty for every tagged run.**
  `comparative_benchmark` keys its report by `{subset}_{model}_{tag}` but the
  summary block looked up `{subset}_{model}`, so a tagged run reported
  `best_model_by_rul_mae: None` and `best_rul_mae: inf` — a silently broken
  headline number rather than an error. The lookup now uses the same run key
  the report is written with.
- **A stale artifact could shadow a fresh one.** `scripts/decision.py`
  preferred the legacy structured layout
  (`outputs/results/uncertainty/<variant>/<subset>/<model>.npz`) over the flat
  file the current pipeline writes. The five FD001/`gru` files still present
  in that tree were four weeks old and disagreed with the flat artifacts by up
  to 16 RUL cycles in their point predictions, so a maintenance decision could
  have been produced from a different model's intervals. The flat artifact now
  wins, the structured tree remains a fallback for when it is absent, and the
  resolution is logged. The stale files themselves were removed.
- **Artifact pairs could be compared without knowing whether they matched.**
  Nothing recorded which checkpoint produced a predictions file and which
  produced the intervals file, so regenerating one side and not the other
  produced a comparison between two different models that still looked
  valid. Both artifacts now carry a provenance stamp and the benchmark
  compares them (see *Added* above).

- **Silent calibration leakage across seeds.** `scripts/uncertainty.py` used a
  hard-coded `random_state=42` for the engine hold-out, so a seed-43 or
  seed-44 checkpoint was calibrated against the seed-42 hold-out — which can
  contain engines that checkpoint was trained on. The seed is now read from
  the checkpoint (`checkpoint_seed()` → `run_metadata["seed"]` →
  `checkpoint["seed"]` → `model_config["random_seed"]`) and the split is
  reproduced exactly.
- **The certification benchmark no longer uses a proxy model.**
  `scripts/multiseed_conformal_benchmark.py` previously trained a short-horizon
  surrogate (`IntervalRegressor`) and reported its intervals. It now loads the
  real checkpoints through `scripts.uncertainty.load_model`, reproduces each
  checkpoint's engine split, and computes all five methods with the shared
  `src/uncertainty/pipeline.py` engine. The comparison between methods is
  paired by construction.
- **The finite-sample conformal rank is no longer clamped anywhere.**
  `conformal_quantile` and the new `signed_conformal_quantile` return `+inf`
  when `ceil((n + 1)(1 − α)) > n`; `prediction_interval` clips an infinite
  radius to the RUL domain. The benchmark's private clamped copy is deleted and
  the CLI's CQR path delegates to the shared implementation.

- **The `adaptive` conformal baseline was an exact duplicate of split
  conformal.** `scripts/uncertainty.load_model` deleted every `scale_head.*`
  weight from the checkpoint and the tolerant loader then re-initialised the
  head to a constant 1.0, so `residual / scale` was `residual`. Adaptive and
  split conformal therefore reported identical coverage and width for every
  subset (they can be seen matching to the last digit in the old tables). The
  benchmark now fits the scale head on training-engine residuals — exactly as
  `run_adaptive_conformal` does — and `load_model` no longer discards the
  checkpoint's trained head (missing keys are still filled by
  `load_state_dict_compat`). On FD003 at 90 % the adaptive interval is now
  29.4 cycles wide versus 34.6 for split conformal.
- **The CQR quantile head underfitted and collapsed to marginal quantiles.**
  `QuantileHead` had to re-learn RUL from frozen features through a single
  linear layer, so its pinball fit settled on a very wide, near-constant band:
  CQR intervals were 3× wider than split conformal (104 vs 35 cycles at 90 %
  on FD003). The head is now anchored to the checkpoint's own point prediction
  (`centre = point_prediction + delta(features)`), which is the standard
  conformalized-quantile-regression construction. CQR is now 30.9 cycles wide
  on the same cell and tracks nominal coverage.
- **Adaptive conformal could crash at the miscoverage boundary.**
  `prequential_aci_intervals` clipped `alpha` to `[0, 1]`; a run of misses could
  drive it to exactly 0, and a 0 level has no conformal rank (`ValueError:
  alpha must be in (0, 1)`). The full five-seed run hit this on FD002. `alpha`
  is now floored at 0.01 and the update is documented as "never target more
  than 99 % coverage"; the engine-level unit correctly degenerates to the full
  RUL domain when that level is unattainable rather than aborting the run.

- **`docs/ARCHITECTURE.md`** — the internal design: the pipeline stages and a
  per-script entry-point reference, the module responsibilities and the
  layering rules between them, the artifact contracts and the path-naming
  rules, the conformal calibration math (including the two different
  calibration units the repository uses), the full decision policy, the
  Explainability Reliability Index, the six evaluation dimensions and the
  ablation semantics, an extension guide, the test strategy and CI, a
  glossary, and **fifteen known gaps with their code locations**. The
  `README.md` linked to an `ARCHITECTURE.md` at the repository root that has
  never existed.
- **`docs/RMSE_TARGET_PLAN.md`** — why each value in `configs/fd00X.yaml` is
  what it is, and what the recipe does *not* fix. Every config file already
  pointed at this document.
- **`src/models/registry.py`** — the single source of truth for building a
  model (`ARCHS`, `ALL_MODELS`, `build_model()`, `model_config()`), plus
  `src/models/augmentation.py` and `src/models/gru_att_deg_v2.py`.
- **Regression tests** — `tests/test_models.py` and `tests/test_xai.py` were
  single-`assert True` placeholders and are now real suites covering the model
  registry contract, the checkpoint round-trip, config loading, Integrated
  Gradients and ERI. The suite went from 113 to 152 tests.
- **`LICENSE`** — Apache-2.0. A repository with no license is legally
  unusable; this grants explicit use, modification and redistribution rights
  with a patent grant, and is now declared in `pyproject.toml` and
  `CITATION.cff`.
- **`.github/workflows/ci.yml`** — CI that actually runs the test suite. The
  project ships 113 tests but nothing ran them on commit; the workflow now runs
  `pytest` on `ubuntu-latest` and `windows-latest` across Python 3.9 and 3.12,
  on CPU-only PyTorch, so a reviewer without a GPU gets the same result. Lint
  (`ruff`) runs too.
- **`pyproject.toml`** — PEP 621 project metadata: name, version, authors,
  license, classifiers, dependencies, and tool configuration for `pytest` and
  `ruff`. PyTorch is deliberately *not* pinned (see below).
- **The repository is now `ruff`-clean** (`ruff check src scripts tests`). The
  only suppression is a targeted `per-file-ignores` for `E402` in `scripts/*.py`,
  because CLI entry points insert the repository root into `sys.path` before
  importing `src.*`; the imports are correct, the rule is what is wrong. Import
  order (`I001`), trailing whitespace (`W291`/`W293`) and unused imports were
  fixed rather than suppressed.
- **`CHANGELOG.md`** — this file.
- **`CITATION.cff`** — machine-readable citation metadata, matching the BibTeX
  block in `README.md`, so GitHub/Zenodo and reference managers can pick it up.
- **`RESULTS.md`** — a published results table with a comparison against
  published C-MAPSS SOTA RUL baselines (Li 2018, Zheng 2017, Fu 2024
  Dual-Mixer, Wu 2024 DRLSTM-DA, Bektas 2023, Liu 2024 DSFormer), the full
  multi-seed conformal benchmark tables, and reproduction commands.
- `requires_data` pytest marker, plus an automatic skip guard on the
  preprocessing integration test, so the suite stays green in CI where the
  (uncommitted) C-MAPSS dataset is absent.
- **`tests/test_evaluation.py`** — regression tests for the evaluation-layer
  fixes below: the ablation has to produce non-degenerate deltas, the ERI key
  has to be the one that is written, the duplicate NASA-score module has to
  stay deleted, the interval metrics have to share the validator, the dropout
  mask has to be per window and the attention attribution has to leave the
  model untouched.

### Changed

- **The published headline numbers are now the certified five-seed means.**
  `RESULTS.md`, `docs/SOTA_BENCHMARK.md`, `docs/RMSE_TARGET_PLAN.md` and the
  README summary report `13.361 ± 0.136` (and per-subset
  12.38 / 13.92 / 11.97 / 15.17 with their seed stds) instead of the single-run
  12.947 and the 3-member ensemble's 12.871. Both legacy values are retained and
  labelled as single draws. Consequences stated in the docs rather than hidden:
  the repository's own `< 13` target is **not met**, it sits **2nd of 9** on the
  four-subset mean behind KESTF (12.83), and the 0.04 gap to TTSNet (13.40) is
  inside the seed band, so that comparison is *not established*. The earlier
  claim that `eara_conformal` "hits the 90 % target almost exactly" is replaced
  by the measured five-seed behaviour (0.852–0.917), and the docs now say that
  `nex_scp` is the best-calibrated of the five methods while noting that five
  seeds cannot separate them statistically.
- **Configuration is now wired up.** `configs/fd001.yaml` … `configs/fd004.yaml`
  drive the pipeline instead of being dead files:
  - `scripts/train.py` reads the architecture, optimiser, scheduler, loss
    weights, augmentation and seed from the config (`--overrides key=value,...`
    patches individual keys).
  - `scripts/preprocess.py` gained `--config`, `--window-size` and `--max-rul`;
    `window_size` and `max_rul` now come from the config by default, and all
    output paths are anchored to the project root instead of the working
    directory.
  - `scripts/evaluate.py`, `scripts/uncertainty.py` and `scripts/explain.py`
    rebuild the architecture from the `model_config` stored **inside the
    checkpoint** rather than from a hard-coded registry, so a model trained
    with a per-subset override (e.g. `hidden_size: 160` on FD002/FD004) is no
    longer evaluated with the wrong shape.
  - `scripts/explain.py` reads `window_size` / `max_rul` from the processed-data
    metadata instead of hard-coding `40` / `125`.
  - The RUL domain constants in `src/uncertainty/conformal.py` and the `rul_max`
    decision threshold are derived from `DEFAULT_CONFIG["max_rul"]`, and
    `scripts/train.py` refuses to train when the processed data was labelled
    with a different cap.
- **Integrated Gradients no longer mutates the model it explains.**
  `src/explainability/integrated_gradients.py` called `model.train()`, which
  (a) enabled Dropout, so two identical calls returned different attributions,
  and (b) updated BatchNorm running statistics, permanently corrupting the
  model for every downstream stage that reused it. It now runs in `eval()`
  mode, restores the caller's original mode, and disables cuDNN for the
  backward pass — cuDNN's fused RNN kernels only implement backward in
  training mode, so with the model in `eval()` the gradient computation
  otherwise raised *"cudnn RNN backward can only be called in training mode"*
  on CUDA.
- **`scripts/uncertainty.py`** — the quantile and scale heads now take their
  width from the latent features instead of a hard-coded `hidden_size = 128`,
  which would have failed for the FD002/FD004 checkpoints. Its `--model`
  choices now come from the shared registry, so the attention architectures
  can be evaluated too.
- **`README.md`** — corrected against the current code: the hyper-parameter
  table now matches `configs/*.yaml` (window 40, batch 256, 150 epochs, lr
  3e-4, patience 30, HI weight 0.4), `gru_att_deg_v2` was added to the model
  list, the "config files are not wired up" note is gone, `src/evaluation/` is
  no longer described as placeholder stubs, and the results summary now points
  at the committed multi-seed benchmark instead of a second, irreconcilable set
  of numbers.
- **`docs/` is no longer git-ignored.** It held no files, and both the
  configuration comments and the README linked into it.
- **`requirements.txt`** — removed the hard `torch==2.11.0+cu128` /
  `torchvision==0.26.0+cu128` pins. The previous pin was a CUDA-12.8-only
  build, so anyone on CPU or a different CUDA version could not install or run
  the project. PyTorch installation is now documented (CPU / cu121 / cu124) and
  left to the user's hardware. `numpy` bound retained as `>=1.23,<2.5`.
- **`tests/test_preprocessing.py`** — integration test now skipped when
  `data/raw/<subset>` is missing instead of erroring.
- **`RESULTS.md`** — the `cqr` caveat was stale: `scripts/uncertainty.py` now
  trains quantile heads with the pinball loss and *is* conventional CQR. The
  conformal rank clamp was documented instead.

### Fixed

- **The ablation runner measured nothing.** `src/evaluation/ablation.py`
  patched top-level keys of the decision records (`rul_lower`, `rul_upper`,
  `eri`) that no downstream module reads — the recommender consumes the
  prognostic inputs nested under `inputs` — and the reported metric was
  frozen before the patch was applied. Every variant therefore reported
  identical numbers and a `delta` of exactly `0.0`. The runner now re-runs
  `recommend_action` on the patched inputs of every record, so each variant
  goes through the same code path as the full pipeline; `no_constraints`
  bypasses the constraint layer instead of clearing violations after the
  fact, and the report carries the review-rate, constraint-violation,
  risk-score and action-mix deltas.
- **Two of the three ERI rank correlations were never computed.**
  `src/evaluation/metrics.py` read the temporal relevance under the key
  `temporal`, while `src/explainability/eri.py` writes it under
  `temporal_relevance`. `spearman_ig_temporal` and `spearman_temporal_shap`
  were silently missing from every explainability-quality report.
- **Two NASA-score implementations disagreed in the last digits.**
  `src/models/evaluator.py` (a 35-line Python loop that also averaged the
  error across the `(N, 1)` axis) and `src/evaluation/metrics.py` (vectorised)
  published the same score with different values — `1436.2990129040745` in
  `*_metrics.json` versus `1436.2990129040754` in `accuracy_summary.csv`.
  The duplicate module is deleted; `scripts/evaluate.py` and
  `scripts/ensemble.py` now use `src/evaluation/metrics.py`.
- **Interval metrics bypassed the shared validator.**
  `src/uncertainty/coverage.py::_validate_interval_inputs` was called by
  `empirical_coverage` but re-implemented inline by
  `mean_prediction_interval_width` and `engine_level_coverage`, so a change to
  the invariants would not have reached those callers. All three now route
  through the validator.
- **The dropout robustness sweep zeroed the same sensors in every window.**
  `src/evaluation/metrics.py::robustness_under_dropout` masked a fixed set of
  sensor channels for the whole test set, i.e. it evaluated a permanent
  sensor loss instead of the transient per-window faults the training
  augmentation models. The mask is now drawn per window
  (`per_window_dropout_mask`).
- **The temporal-relevance attribution ran without the model-mode guard.**
  `src/explainability/attention.py` computed input gradients without
  switching the model to `eval()` first, so the attribution was
  non-deterministic under Dropout and could corrupt BatchNorm running
  statistics. The module is deleted; `explain_attention` is now a thin
  wrapper over the eval-guarded gradient path in
  `src/explainability/integrated_gradients.py`.
- **The published point-prediction numbers did not match any artifact.**
  `RESULTS.md § 2` reported "FD001 / GRU-Att-Deg / MAE 10.05 / RMSE 13.64 /
  R² 0.884 / NASA 308.63", a row that mixed values from different models
  (RMSE 13.64 and R² 0.884 are the `lstm` run; NASA 308.63 matches no
  prediction file at all) and was left over from an earlier run. The table is
  now the four-subset `gru_att_deg_v2` grid plus the three-run `ensemble`,
  recomputed from `outputs/results/*_predictions.npz` by `scripts/summarize.py`
  and `scripts/sota_benchmark.py`, and the surrounding prose quotes the
  published baselines against those numbers. The `README.md` summary and the
  reproduction commands point at the same table.
- **The README hyper-parameter table claimed augmentation was on everywhere.**
  `configs/fd001.yaml`, `fd002.yaml` and `fd003.yaml` all set
  `augment: false` after an ablation showed it raised RUL RMSE; only FD004
  keeps it on. The table now says `off / off / off / on`, with a caveat that
  `DEFAULT_CONFIG["augment"]` is still `true` and the per-subset YAML wins.
- **`docs/ARCHITECTURE.md` documented a module that does not exist.** It listed
  `src/hitl/schema.py`; the real modules are `src/hitl/feedback.py` and
  `src/hitl/logger.py`. The interval-artifact contract also named the output
  file `<subset>_<model>_<method>.npz`, while the four CLI methods write
  `conformal`, `cqr`, `adaptive_conformal` and `engine_joint_conformal`.

### Removed

- **Two dead config keys.** `dataset` and `model` were declared in
  `DEFAULT_CONFIG` and in every `configs/fd00X.yaml`, but nothing ever read
  them:
  - `dataset` was overwritten by `load_config()` with the subset resolved
    from the file name, so the YAML value was inert and could silently
    disagree with the file it lived in.
  - `model` was never read at all — the architecture comes from
    `scripts/train.py --model` and is recorded in the checkpoint's
    `model_config`, from which `scripts/evaluate.py` rebuilds the model. A
    `model:` key in the YAML could therefore only ever disagree with the
    architecture that was actually trained.

  Both keys are gone from `DEFAULT_CONFIG`, from the four subset YAMLs and
  from the `load_config()` result; `load_config()` no longer injects
  `config["dataset"]`. The subset is derived from the file name and the
  architecture from the command line, which is where both already came
  from. `tests/test_models.py::test_subset_yaml_only_uses_known_keys` now
  fails the build if a subset YAML declares a key that `DEFAULT_CONFIG` does
  not know about — the failure mode this cleanup removed.

### Notes

- `CONTRIBUTING.md` still instructs `pytest tests/ -v`; both that and
  `pytest -q` are configured in `pyproject.toml`.
- The library-level conformal rank clamp **is** fixed:
  `src/uncertainty/conformal.py::conformal_quantile` now computes the rank
  through the extracted `conformal_rank()` helper and returns `+inf` when the
  rank is not attainable (`allow_infinite=True`, the default) or raises a
  `ValueError` with `allow_infinite=False`. `regime_conformal_quantile`
  inherits the behaviour. **The benchmark was not re-run**, and
  `scripts/multiseed_conformal_benchmark.py` keeps its own clamped copy of the
  quantile, so the published `eara_conformal` coverage numbers still reflect
  a clamped estimator — the per-regime calibration pools are small enough that
  the 95 % finite-sample rank is unattainable in every one of the 20 runs.
  See `docs/ARCHITECTURE.md § 7.1` for the verified breakdown.
- **`src/uncertainty/pipeline.py` is not wired into anything.** It is a
  complete, model-agnostic interval engine with engine-level calibration and
  a correctly-implemented finite-sample quantile, but no script imports it —
  `scripts/uncertainty.py` and `scripts/multiseed_conformal_benchmark.py`
  each keep their own copy of the interval logic, which is how the rank clamp
  of § 7.1 survived in the benchmark. Documented in
  `docs/ARCHITECTURE.md § 7.14`.
- **Tag support is only half-wired.** `scripts/train.py` and
  `scripts/evaluate.py` honour `--tag`, but `scripts/uncertainty.py` writes
  its interval files to the untagged path and `scripts/explain.py` /
  `scripts/decision.py` resolve untagged inputs, so two tagged runs of the
  same (model, subset) overwrite each other's downstream artifacts.
  `src/utils/paths.py` already knows the correct names; the writers have not
  been switched over. Documented in `docs/ARCHITECTURE.md § 7.13`.
- **Seed control is now complete for training.** `src/utils/seed.py::set_seed`
  seeds `random`, `numpy`, `torch` and `torch.cuda`, enables
  `torch.use_deterministic_algorithms(warn_only=True)`, pins cuDNN to
  deterministic mode and disables TF32; `scripts/train.py` calls it before any
  RNG is touched and persists the determinism block in the checkpoint. The
  multi-seed benchmark and the uncertainty/explainability scripts still seed
  their RNGs locally, so they do not enable the deterministic kernels — the
  remaining gap is documented in `docs/ARCHITECTURE.md § 7.7`.


## [0.1.0] - 2026-01-01

Initial public research release. (No git tag has been cut yet; this entry
records the state of the repository prior to the Unreleased changes.)

### Added

- Preprocessing and windowing pipeline for NASA C-MAPSS (`FD001`–`FD004`),
  with engine-disjoint splits and zero engine overlap asserted at every stage.
- Joint RUL + HI prediction across five model families: LSTM, GRU, Transformer,
  Hybrid, GRU-Att-Deg.
- Conformal uncertainty — four CLI methods (`split`, `cqr`, `adaptive`,
  `engine_joint` / EARA-Conformal).
- Multi-seed conformal benchmark — five methods × four subsets × five seeds.
- Explainability — Integrated Gradients, SHAP, temporal relevance, and the
  Explainability Reliability Index (ERI).
- Decision engine — rule-based actions with operational constraints and HTML
  reports.
- HITL review workflow with append-only feedback log and conservative policy
  refinement (model retraining intentionally out of scope).
- 113-test pytest suite.
