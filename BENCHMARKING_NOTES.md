# Conformal RUL Benchmark

## Protocol

`scripts/multiseed_conformal_benchmark.py` compares split conformal, neural CQR,
global scale-adaptive conformal, prequential nex-SCP, and EARA-Conformal on
FD001 through FD004. It defaults to five fixed seeds (42-46) and evaluates
nominal coverage levels of 80%, 90%, and 95%. Every method for a subset/seed
shares one engine-disjoint training and calibration split and one trained GRU
predictor. The official test arrays are used for evaluation; their stored order
defines the sequential nex-SCP stream.

The processed training arrays contain multiple windows per engine, while the
official test arrays contain one censored endpoint per engine. Calibration
samples one reproducible window per held-out engine as an endpoint proxy. A
target-distribution check showed that these sampled training windows have RUL
quartiles close to the official test labels across all four subsets; engine
failure-end windows instead concentrate near RUL zero. The engine IDs for each
training/calibration partition are saved once in `split_manifest.json`;
per-engine interval rows carry the stable split fingerprint.

## Methods

- **Split conformal:** symmetric interval around the shared point estimate,
  calibrated from engine-maximum absolute residuals.
- **CQR:** the shared model learns quantile bounds at the tails needed for each
  requested coverage. Engine-maximum CQR violation scores calibrate interval
  expansion.
- **Adaptive:** global normalized residual calibration using the shared
  model's learned scale, without regime-specific calibration or sequential
  feedback. This preserves the distinction from nex-SCP and EARA-Conformal.
- **nex-SCP:** sequential adaptive miscoverage (ACI update). It uses a
  calibration residual quantile for each test engine, then updates the target
  miscoverage level from the revealed outcome before predicting the next
  engine. This is an online protocol, not a strictly offline batch baseline.
- **EARA-Conformal:** k-means regimes are fitted on training engines; normalized
  residuals use the model's learned scale; engine-level scores are calibrated
  within regimes. Regimes with fewer than five calibration engines fall back to
  the global normalized score quantile.

Bounds are clipped to RUL domain [0, 125]. For each seed, the runner records
empirical coverage, signed coverage gap, mean interval width, and Winkler
interval score. Summary tables report mean and sample standard deviation over
seeds. Paired comparisons report EARA minus each baseline, a paired t interval,
and an exact sign-flip p-value with Holm adjustment within each subset,
coverage, and metric family.

## Running

```powershell
python scripts/multiseed_conformal_benchmark.py --subsets all --seeds 42 43 44 45 46
```

Use `--epochs` and `--batch-size` to set training cost. Outputs are written to
`results/conformal_benchmark/`: per-engine intervals, per-seed metrics,
mean/std summaries, paired comparisons, run configuration, and one coverage /
width plot per subset.

## Initial Run (90% Target)

The completed run used five seeds and 15 training epochs. The table shows
EARA-Conformal against the lowest-Winkler baseline for each subset; values are
mean ± seed standard deviation for coverage, followed by mean width and mean
Winkler score.

| Subset | EARA coverage | EARA width | EARA Winkler | Lowest-Winkler baseline | Baseline coverage | Baseline width | Baseline Winkler |
|---|---:|---:|---:|---|---:|---:|---:|
| FD001 | 0.900 ± 0.191 | 98.90 | 138.29 | adaptive | 0.924 ± 0.121 | 102.71 | 115.54 |
| FD002 | 0.900 ± 0.053 | 113.39 | 128.53 | nex-SCP | 0.906 ± 0.010 | 114.49 | 121.70 |
| FD003 | 0.906 ± 0.078 | 93.50 | 105.58 | adaptive | 0.922 ± 0.066 | 99.01 | 101.97 |
| FD004 | 0.881 ± 0.061 | 106.52 | 140.85 | CQR | 0.956 ± 0.020 | 115.90 | 117.38 |

This is mixed evidence, not a consistent EARA improvement. EARA gives narrower
intervals in all four 90% comparisons shown, but has worse Winkler scores in
FD001, FD002, and FD004; FD004 also under-covers the 90% target on average.
The complete comparison at all target levels is in
`results/conformal_benchmark/summary_mean_std.csv`; paired seed-level analyses
are in `paired_comparisons.csv`.

## Interpretation Limits

Five seeds provide only coarse inferential resolution. The exact two-sided
sign-flip test has a minimum attainable p-value of 0.0625 at five pairs, and
the official test engines are reused across seeds. Treat seed-level inference
as a stability comparison, not a substitute for independent test fleets. The
shared GRU is a controlled benchmark predictor; method rankings may differ for
other architectures. Fix hyperparameters and epochs before examining test
results.
