# Contributing to PHM-XAI

Thank you for contributing to **PHM-XAI**, an Explainable AI framework for Prognostics and Health Monitoring of aircraft engines.

This guide explains how to set up the project, understand its structure, develop new components, run tests, and submit changes.

---

## 1. Project Overview

PHM-XAI processes multivariate sensor time-series data from the NASA C-MAPSS dataset and supports:

- Remaining Useful Life (RUL) prediction
- Health Index (HI) prediction
- LSTM, GRU, Transformer, and Hybrid models
- Explainable AI
- Uncertainty quantification
- Maintenance decision support
- Human-in-the-loop learning

The project follows a modular architecture so new models, XAI methods, uncertainty techniques, and decision policies can be added independently.

---

## 2. Development Setup

### Prerequisites

- Python 3.9+
- Git
- pip or conda
- CUDA-compatible GPU (optional)

### Clone the Repository

```bash
git clone <repository-url>
cd phm_project
```

### Create a Virtual Environment

#### Windows

```bash
python -m venv .venv
.venv\Scriptsctivate
```

#### Linux/macOS

```bash
python -m venv .venv
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Verify PyTorch

```bash
python -c "import torch; print(torch.__version__)"
python -c "import torch; print(torch.cuda.is_available())"
```

---

## 3. Project Structure

```text
phm_project/
│
├── data/
│   ├── raw/                  # Original C-MAPSS data
│   └── processed/            # Preprocessed datasets
│
├── configs/                  # YAML experiment configurations
│
├── src/
│   ├── preprocessing/        # Data preparation pipeline
│   ├── models/               # Prognostics models
│   ├── explainability/       # XAI methods
│   ├── uncertainty/          # Uncertainty estimation
│   ├── decision_engine/      # Maintenance decisions
│   ├── hitl/                 # Human-in-the-loop components
│   └── utils/                # Shared utilities
│
├── scripts/                  # Main execution scripts
├── tests/                    # Automated tests
├── outputs/                  # Reports, figures, checkpoints, logs
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 4. Architecture

The main development flow is:

```text
Raw Sensor Data
      │
      ▼
Preprocessing
      │
      ├── Validation
      ├── Scaling
      ├── Label Generation
      ├── Feature Selection
      └── Windowing
      │
      ▼
Prognostics Model
      │
      ├── LSTM
      ├── GRU
      ├── Transformer
      └── Hybrid
      │
      ▼
RUL + Health Index
      │
      ├── Explainability
      ├── Uncertainty
      └── Decision Support
               │
               ▼
        Human Feedback
```

Keep new components consistent with this modular flow.

---

## 5. Development Workflow

Before making changes:

```bash
git pull
```

Create a feature branch:

```bash
git checkout -b feature/your-feature-name
```

Examples:

```text
feature/shap-explanations
feature/conformal-prediction
feature/new-model
feature/decision-engine
fix/preprocessing-validation
```

Make your changes, test them, and commit them:

```bash
git add .
git commit -m "Add SHAP explanation module"
```

Push your branch:

```bash
git push origin feature/your-feature-name
```

Then open a Pull Request.

---

## 6. Adding a New Model

New prognostics models should be placed in:

```text
src/models/
```

For example:

```text
src/models/
├── lstm.py
├── gru.py
├── transformer.py
├── hybrid.py
└── new_model.py
```

### Recommended Process

1. Implement the model in `src/models/`.
2. Follow the existing model structure.
3. Support the RUL and HI prediction tasks where applicable.
4. Add configuration parameters to the YAML configuration.
5. Add the model option to the training workflow.
6. Add appropriate tests.
7. Train the model on at least one C-MAPSS subset.
8. Evaluate its performance.
9. Document the model and its configuration.

Example:

```bash
python scripts/train.py --model new_model --subset FD001
```

---

## 7. Adding a New Preprocessing Component

Preprocessing modules belong in:

```text
src/preprocessing/
```

The current pipeline follows:

```text
Loader
   ↓
Validator
   ↓
Label Generator
   ↓
Scaler
   ↓
Feature Selection
   ↓
Window Generator
```

A new preprocessing component should:

- Have a clearly defined responsibility.
- Avoid modifying unrelated pipeline stages.
- Support configuration where appropriate.
- Validate its inputs.
- Produce predictable outputs.
- Include tests.

---

## 8. Adding Explainability Methods

Explainability components belong in:

```text
src/explainability/
```

The project plans to support:

- SHAP
- Integrated Gradients
- Temporal attention visualization

New XAI methods should clearly specify:

- Model input requirements
- Explanation output
- Local/global scope
- Feature or temporal importance
- Visualization requirements
- Computational limitations

Example structure:

```text
src/explainability/
├── shap_explainer.py
├── integrated_gradients.py
└── attention.py
```

---

## 9. Adding Uncertainty Methods

Uncertainty-related components belong in:

```text
src/uncertainty/
```

The research framework includes conformal prediction and calibration.

New uncertainty methods should document:

- Type of uncertainty
- Required model outputs
- Calibration procedure
- Prediction interval format
- Coverage or reliability metrics

Example:

```text
src/uncertainty/
├── conformal.py
├── calibration.py
└── metrics.py
```

---

## 10. Decision Engine Development

Maintenance decision logic belongs in:

```text
src/decision_engine/
```

Decision components may use:

- RUL predictions
- Health Index
- Uncertainty
- Explainability information
- Operational constraints
- Maintenance cost or risk

Decision outputs should be understandable and traceable.

A recommendation should provide enough information to explain why an action was selected.

---

## 11. Human-in-the-Loop Development

Human-in-the-loop components belong in:

```text
src/hitl/
```

The intended workflow is:

```text
AI Prediction
     ↓
Explanation + Uncertainty
     ↓
Maintenance Recommendation
     ↓
Expert Review
     ↓
Accept / Modify / Override
     ↓
Feedback Logging
     ↓
Model or Decision Refinement
```

Human feedback should be recorded with appropriate metadata so that decisions can be analyzed and reproduced.

---

## 12. Configuration Management

Experiment parameters should be stored in:

```text
configs/
```

Do not hard-code experiment-specific values inside source files when they can be configured externally.

Example:

```yaml
dataset: FD001
window_size: 30
max_rul: 125

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

This keeps experiments reproducible and makes comparisons easier.

---

## 13. Reproducibility

Use fixed random seeds when running experiments:

```python
SEED = 42
```

The project uses seeded randomness for:

- Python
- NumPy
- PyTorch
- CUDA when available

When reporting experimental results, record:

- Dataset subset
- Model
- Configuration
- Random seed
- Training settings
- Evaluation metrics

---

## 14. Testing

Tests are located in:

```text
tests/
```

Run the complete test suite:

```bash
pytest tests/ -v
```

Run a specific test module:

```bash
pytest tests/test_preprocessing.py -v
pytest tests/test_models.py -v
```

When adding a new feature, add or update tests where appropriate.

A Pull Request should not be submitted with known failing tests unless the failure is clearly documented.

---

## 15. Code Quality

Contributions should follow these principles:

- Keep modules focused on a single responsibility.
- Prefer readable code over clever code.
- Use meaningful variable and function names.
- Add type hints where practical.
- Document non-obvious logic.
- Avoid unnecessary duplication.
- Keep configuration outside the source code.
- Do not commit generated datasets or large model files.

---

## 16. Logging

Use Python's logging system rather than unnecessary `print()` statements for application and experiment logs.

Example:

```python
import logging

logger = logging.getLogger(__name__)

logger.info("Processing dataset")
logger.warning("Missing sensor values detected")
logger.error("Dataset file not found")
```

Logs should provide enough information to diagnose failures without exposing sensitive information.

---

## 17. Data and Generated Files

Do not commit:

- Raw NASA C-MAPSS datasets
- Large generated `.npy` files
- Model checkpoints
- Temporary files
- Local virtual environments
- Generated logs

Use the appropriate directories locally:

```text
data/raw/
data/processed/
outputs/checkpoints/
outputs/logs/
outputs/figures/
outputs/reports/
```

Ensure large or generated files are excluded through `.gitignore`.

---

## 18. Pull Request Guidelines

Before opening a Pull Request:

```bash
pytest tests/ -v
```

Check that:

- [ ] The code runs successfully.
- [ ] Relevant tests pass.
- [ ] New functionality has appropriate tests.
- [ ] Configuration is not unnecessarily hard-coded.
- [ ] Documentation has been updated.
- [ ] No datasets or large generated files are included.
- [ ] Experimental results are reproducible.
- [ ] The PR description clearly explains the change.

### Pull Request Description

Use:

```text
## What changed?

Brief description of the implementation.

## Why?

Explain the problem or research requirement.

## Testing

Describe tests performed.

## Results

Include relevant metrics or screenshots if applicable.

## Notes

Mention limitations, assumptions, or future work.
```

---

## 19. Commit Messages

Use short and descriptive commit messages.

Recommended format:

```text
<type>: <description>
```

Examples:

```text
feat: add SHAP explanation module
feat: add conformal prediction
fix: correct FD002 preprocessing
refactor: simplify model trainer
test: add transformer tests
docs: update contributor guide
```

Common types:

- `feat` — New functionality
- `fix` — Bug fix
- `refactor` — Code restructuring
- `test` — Tests
- `docs` — Documentation
- `perf` — Performance improvement

---

## 20. Research Contributions

Because PHM-XAI is a research project, contributions should distinguish between:

- New implementation
- Experimental improvement
- Research hypothesis
- Evaluation result
- Infrastructure change

For research-related changes, document the experiment configuration and evaluation methodology so results can be reproduced.

Do not claim performance improvements without supporting experimental evidence.

---

## 21. Current Development Areas

The core prognostics pipeline currently includes:

- Data loading and validation
- RUL and HI labeling
- Feature preprocessing
- Sliding-window generation
- LSTM
- GRU
- Transformer
- Hybrid models
- Training
- Evaluation
- Checkpointing
- Configuration management

The following areas are under active development:

- Explainability
- Uncertainty quantification
- Decision intelligence
- Human-in-the-loop learning
- Comprehensive multi-dimensional evaluation

Refer to `README.md` for the current project status.

---

## 22. Getting Help

Before opening an issue:

1. Check the README.
2. Check the `docs/` directory.
3. Review existing issues.
4. Verify the dataset and configuration.
5. Check logs under `outputs/logs/`.
6. Reproduce the problem with the smallest possible example.

When reporting a bug, include:

- Operating system
- Python version
- PyTorch version
- Dataset subset
- Model
- Configuration
- Error message
- Steps to reproduce

---

## 23. License

By contributing to PHM-XAI, you agree that your contributions will be licensed under the project's MIT License.

See `LICENSE` for details.
