# Changelog

All notable changes to PHM-XAI are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

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

### Changed

- **`requirements.txt`** — removed the hard `torch==2.11.0+cu128` /
  `torchvision==0.26.0+cu128` pins. The previous pin was a CUDA-12.8-only
  build, so anyone on CPU or a different CUDA version could not install or run
  the project. PyTorch installation is now documented (CPU / cu121 / cu124) and
  left to the user's hardware. `numpy` bound retained as `>=1.23,<2.5`.
- **`tests/test_preprocessing.py`** — integration test now skipped when
  `data/raw/<subset>` is missing instead of erroring.

### Notes

- `CONTRIBUTING.md` still instructs `pytest tests/ -v`; both that and
  `pytest -q` are configured in `pyproject.toml`.

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
