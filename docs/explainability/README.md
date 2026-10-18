# ERI agreement and faithfulness benchmark

The ERI is an **explanation-agreement** score; it is not itself evidence of
faithfulness. This benchmark checks whether per-sample ERI covaries with
model-output sensitivity under feature ablation, and compares that association
with Integrated Gradients (IG), Kernel SHAP, and a LIME-style local masking
surrogate.

## Run

Train each desired model seed using the seed-specific output option, then run:

```powershell
python scripts/benchmark_explainability.py `
  --subset FD001 `
  --model gru `
  --target rul `
  --seeds 42 43 44 45 46 `
  --samples 100
```

Train one checkpoint for every requested model seed first:

```powershell
foreach ($seed in 42, 43, 44, 45, 46) {
  python scripts/train.py --subset FD001 --model gru --seed $seed
}
```

The runner expects checkpoints at
`outputs/checkpoints/<model>/<subset>/seed_<seed>/best_model.pt`, matching
`scripts/train.py --seed <seed>`. For an existing non-seed-specific checkpoint,
run one seed with `--use-default-checkpoint`; do not list the same checkpoint as
multiple independent seeds.
The training seed controls both model initialization and the engine-wise
train/validation split, so differences across checkpoints reflect both sources
of variation rather than initialization alone.

The runner samples windows round-robin across test engines (repeating an engine
only when the requested sample count exceeds the number of engines). It writes
sample-level measurements and correlation summaries to
`outputs/xai/<subset>/<model>/faithfulness_benchmark/<target>/` and a Markdown
result report to this directory. Overall and per-seed correlation scatter plots
are archived under `docs/explainability/figures/`. The target (`rul`/`hi`) is a
subdirectory so target results do not overwrite one another.
SHAP and LIME perturbation counts can be reduced for a smoke test; use the same
settings for all seeds in a reported experiment.

## Protocol and interpretation

- **Rank agreement:** fraction of the top-k absolute-attribution sensors shared
  by an explainer pair.
- **Sign agreement:** fraction with matching signed contribution among shared
  top-k sensors.
- **Rank correlation:** Spearman correlation of full sensor-importance ranks.
  These complement the top-k and sign comparisons used in OpenXAI/EXAGREE-style
  explanation-agreement evaluations. ERI uses tie-aware average ranks for its
  Spearman terms.
- **Faithfulness:** mask complete sensor trajectories with zero (the
  standardized feature baseline). Deletion-gap AUC integrates normalized
  absolute prediction change as ranked sensors are removed (larger means
  earlier/larger sensitivity, opposite to the conventional residual-output
  deletion AUC); insertion AUC integrates
  fidelity to the full prediction as ranked sensors are restored; prediction
  gap reports the absolute output change after removing the top half.
- **Correlation:** Pearson and Spearman coefficients, two-sided p-values, and
  Benjamini-Hochberg q-values are computed across sample-seed rows, after
  averaging within engine across all seeds, and after averaging within engine
  separately for each model seed. Per-seed engine-level q-values are adjusted
  within each seed and report outcome consistency across checkpoints. Prefer
  engine-level analyses over pooled sample-seed rows when engines have repeated
  windows.
  A positive coefficient and `p < 0.05` are empirical outcomes, not guaranteed
  properties of ERI.

The temporal-relevance component of the current ERI is derived by reweighting
IG, so ERI contains an IG-dependent agreement component. Correlation with
faithfulness is therefore a validation result, not proof of causal truth.
Samples from the same test set are not independent evidence for generalization.
Confirmatory claims still require evaluation on held-out engines; do not tune
on the test set to manufacture significance.
