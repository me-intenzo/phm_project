# PHM-XAI


## Dataset

NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) — 4 subsets: FD001, FD002, FD003, FD004.

Download the dataset and place the files in `data/raw/`:

```
data/raw/
├── train_FD001.txt
├── test_FD001.txt
└── RUL_FD001.txt
```

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Usage

### EDA

```bash
python main.py
```

### Preprocessing

```bash
python scripts/preprocess.py
```

### Training

```bash
python scripts/train.py
```

### Evaluation

```bash
python scripts/evaluate.py
```

### Explainability

```bash
python scripts/explain.py
```

### Uncertainty

```bash
python scripts/uncertainty.py
```

### Decision Support

```bash
python scripts/decision.py
```

## Configuration

Each subset has a YAML config in `configs/`. Example `configs/fd001.yaml`:

```yaml
dataset: FD001
window_size: 30
batch_size: 64
epochs: 50
learning_rate: 0.001
random_seed: 42
device: cuda  # or cpu
```

## Key Components

- **Preprocessing** — data loading, validation, min-max scaling, sliding window segmentation, RUL labeling
- **Models** — LSTM, GRU, Transformer, and hybrid architectures with custom loss functions
- **Explainability** — SHAP values, Integrated Gradients, attention visualization, Explainability Reliability Index (ERI)
- **Uncertainty** — conformal prediction intervals, calibration, coverage guarantees
- **Decision Engine** — rule-based maintenance recommendations with operational constraints
- **HITL** — human-in-the-loop feedback loop for model updating

## Requirements

- Python 3.9+
- PyTorch
- SHAP
- Captum
- scikit-learn
- pandas, numpy, matplotlib, seaborn

See `requirements.txt` for full list.
