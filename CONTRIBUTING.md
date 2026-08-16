# Contributing to PHM-XAI

Thank you for your interest in contributing to **PHM-XAI** — a prognostics and health monitoring framework for aircraft engine degradation modeling using the NASA C-MAPSS dataset.

This guide covers development setup, project conventions, and the contribution workflow for the **currently implemented pipeline** (data preprocessing, multi-task model training, and evaluation).

---

## Table of Contents

- [Project Overview](#project-overview)
- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Architecture](#architecture)
- [Development Workflow](#development-workflow)
- [Adding a New Model](#adding-a-new-model)
- [Adding a Preprocessing Component](#adding-a-preprocessing-component)
- [Configuration](#configuration)
- [Testing](#testing)
- [Code Standards](#code-standards)
- [Pull Request Guidelines](#pull-request-guidelines)
- [Commit Messages](#commit-messages)
- [What Not to Commit](#what-not-to-commit)
- [Future Contribution Areas](#future-contribution-areas)
- [Getting Help](#getting-help)

---

## Project Overview

PHM-XAI currently implements a complete **Objective 1** pipeline:

- Load and validate NASA C-MAPSS sensor data (FD001–FD004)
- Generate RUL and Health Index labels
- Preprocess features (selection, scaling, windowing)
- Train multi-task prognostics models (LSTM, GRU, Transformer, Hybrid)
- Evaluate predictions with standard and NASA-specific metrics

The codebase is modular so future components (explainability, uncertainty, decision support) can be added without modifying the core pipeline.

---

## Development Setup

### Prerequisites

- Python 3.9+
- Git
- pip (or conda)
- CUDA-compatible GPU (optional, recommended for training)

### Clone and Install

```bash
git clone <repository-url>
cd phm_project

python -m venv .venv
```

**Windows:**

```bash
.venv\Scripts\activate
```

**Linux / macOS:**

```bash
source .venv/bin/activate
```

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Verify Installation

```bash
python -c "import torch; print(torch.__version__)"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
pytest tests/ -v
```

### Dataset

Place NASA C-MAPSS files in `data/raw/` before running preprocessing:

```
data/raw/train_FD001.txt
data/raw/test_FD001.txt
data/raw/RUL_FD001.txt
```

---

## Project Structure

```
phm_project/
├── configs/                  # YAML experiment configurations
├── data/
│   ├── raw/                  # Original C-MAPSS data (not committed)
│   └── processed/            # Preprocessed .npy arrays (not committed)
├── scripts/
│   ├── preprocess.py         # Preprocessing pipeline
│   ├── train.py              # Model training
│   └── evaluate.py           # Model evaluation
├── src/
│   ├── preprocessing/        # Loader, validator, labeling, windowing
│   ├── models/               # LSTM, GRU, Transformer, Hybrid, trainer
│   ├── evaluation/           # Metrics utilities
│   └── utils/                # Config, logging, seed, I/O
├── tests/                    # pytest test suite
├── outputs/                  # Generated artifacts (not committed)
├── main.py                   # EDA entry point
├── requirements.txt
└── README.md
```

---

## Architecture

### Implemented Pipeline

```
Raw Sensor Data (C-MAPSS)
        │
        ▼
Preprocessing
  ├── Validation
  ├── RUL / HI Label Generation
  ├── Feature Selection
  ├── StandardScaler
  └── Sliding-Window Generation
        │
        ▼
Multi-Task Prognostics Model
  ├── LSTM
  ├── GRU
  ├── Transformer
  └── Hybrid
        │
        ▼
RUL + Health Index Predictions
        │
        ▼
Evaluation (MAE, RMSE, R², NASA Score)
```

### Model Interface Convention

All prognostics models must implement:

```python
def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    x: (batch, window_size, num_features)
    Returns: (pred_rul, pred_hi) each of shape (batch,)
    """
```

Keep new models consistent with this interface so they work with `ModelTrainer`, `MultiTaskLoss`, and `scripts/evaluate.py`.

---

## Development Workflow

1. Pull the latest changes:

```bash
git pull
```

2. Create a feature branch:

```bash
git checkout -b feature/your-feature-name
```

Example branch names:

```
feature/new-model
feature/improved-windowing
fix/preprocessing-validation
test/add-transformer-coverage
docs/update-readme
```

3. Make changes and run tests:

```bash
pytest tests/ -v
```

4. Commit and push:

```bash
git add .
git commit -m "feat: add improved feature selection"
git push origin feature/your-feature-name
```

5. Open a Pull Request with a clear description of what changed and why.

---

## Adding a New Model

Place new models in `src/models/`:

```
src/models/
├── lstm.py
├── gru.py
├── transformer.py
├── hybrid.py
└── your_model.py
```

### Checklist

1. Implement the model in `src/models/your_model.py`
2. Follow the shared encoder + dual-head pattern (RUL + HI)
3. Match the `forward()` return signature used by existing models
4. Register the model in `scripts/train.py` and `scripts/evaluate.py`
5. Add configuration parameters to `configs/` if needed
6. Add tests in `tests/test_models.py`
7. Train and evaluate on at least one C-MAPSS subset (e.g. FD001)
8. Document the architecture and hyperparameters

### Example

```bash
python scripts/train.py --model your_model --subset FD001
python scripts/evaluate.py --model your_model --subset FD001
```

---

## Adding a Preprocessing Component

Preprocessing modules live in `src/preprocessing/`. The current pipeline order is:

```
Loader → Validator → Label Generator → Feature Selector → Scaler → Window Generator
```

When adding a new step:

- Assign a single, clear responsibility
- Fit on training data only; transform test data separately
- Do not break engine-level grouping used for train/validation splits
- Add tests in `tests/test_preprocessing.py`
- Update `scripts/preprocess.py` if the step belongs in the main pipeline

---

## Configuration

Experiment parameters belong in `configs/`:

```yaml
dataset: FD001
window_size: 30
batch_size: 64
epochs: 50
learning_rate: 0.001
hidden_size: 128
num_layers: 2
dropout: 0.3
rul_loss_weight: 1.0
hi_loss_weight: 0.5
device: cuda
random_seed: 42
```

Avoid hard-coding experiment-specific values in source files when they can be configured externally.

---

## Testing

Tests are in `tests/`:

```bash
# Full suite
pytest tests/ -v

# Specific modules
pytest tests/test_preprocessing.py -v
pytest tests/test_models.py -v
```

When adding features, include or update tests. Do not open a PR with known failing tests unless the failure is documented and intentional.

---

## Code Standards

- Follow **PEP 8** for Python style
- Use **type hints** where practical
- Keep modules focused on a single responsibility
- Prefer readable code over clever abstractions
- Use Python `logging` instead of `print()` in pipeline code
- Document non-obvious business logic only
- Match naming and structure of surrounding code

### Logging Example

```python
import logging

logger = logging.getLogger(__name__)
logger.info("Loading processed training data...")
logger.warning("Engine %s has insufficient cycles", engine_id)
```

---

## Pull Request Guidelines

Before submitting:

```bash
pytest tests/ -v
```

Verify:

- [ ] Code runs without errors on at least one C-MAPSS subset
- [ ] Relevant tests pass
- [ ] New functionality includes tests where appropriate
- [ ] Documentation is updated (README or this file if needed)
- [ ] No datasets, checkpoints, or large generated files are included
- [ ] Configuration is not unnecessarily hard-coded

### PR Description Template

```markdown
## What changed?
Brief description of the implementation.

## Why?
Problem or research requirement being addressed.

## Testing
Commands run and results (e.g. pytest, train/eval on FD001).

## Results
Metrics or screenshots if applicable.

## Notes
Limitations, assumptions, or follow-up work.
```

---

## Commit Messages

Use conventional commit format:

```
<type>: <description>
```

| Type | Use |
|------|-----|
| `feat` | New functionality |
| `fix` | Bug fix |
| `refactor` | Code restructuring |
| `test` | Tests |
| `docs` | Documentation |
| `perf` | Performance improvement |

Examples:

```
feat: add TCN prognostics model
fix: correct test RUL label assignment for FD002
test: add hybrid model forward pass tests
docs: update quick start in README
```

---

## What Not to Commit

Do not commit:

- Raw NASA C-MAPSS datasets (`data/raw/`)
- Processed `.npy` files (`data/processed/`)
- Model checkpoints (`outputs/checkpoints/`)
- Log files and generated figures
- Virtual environments (`.venv/`)
- Local IDE or OS files

These paths are excluded via `.gitignore`.

---

## Future Contribution Areas

The following modules are reserved for upcoming research phases and are not yet part of the active pipeline. Contributions in these areas should follow the same modular conventions described above.

| Phase | Module | Planned Focus |
|-------|--------|---------------|
| O2 | `src/uncertainty/` | Conformal prediction, calibration |
| O3 | `src/explainability/` | SHAP, Integrated Gradients, attention |
| O4 | `src/decision_engine/` | Maintenance rules and recommendations |
| O5 | `src/hitl/` | Expert feedback and model refinement |
| O6 | `src/evaluation/` | Benchmarking, ablation, robustness |

Refer to the **Future Development** section in `README.md` for the full roadmap.

---

## Getting Help

Before opening an issue:

1. Read `README.md` and this guide
2. Check existing issues
3. Verify dataset files are in `data/raw/`
4. Review logs under `outputs/logs/`

When reporting a bug, include:

- Operating system and Python version
- PyTorch version and CUDA availability
- C-MAPSS subset and model used
- Full error message and steps to reproduce

---

## Questions?

Open a GitHub issue for discussion, bug reports, or feature proposals.
