# PHM-XAI

**Explainable Prognostics and Health Monitoring for Aircraft Engines**

PHM-XAI is a research framework for predictive maintenance using multivariate sensor time-series data from the **NASA C-MAPSS** dataset. It combines **Remaining Useful Life (RUL)** and **Health Index (HI)** prediction with explainability, uncertainty estimation, maintenance decision support, and human-in-the-loop learning.

## Key Features

- Multi-task RUL + Health Index prediction
- LSTM, GRU, Transformer, and Hybrid models
- NASA C-MAPSS FD001–FD004 support
- Configurable preprocessing and training
- SHAP, Integrated Gradients, and attention-based XAI *(in development)*
- Conformal prediction and calibration *(in development)*
- Explainable maintenance decisions *(in development)*
- Human-in-the-loop feedback *(in development)*

## Architecture

```text
NASA C-MAPSS
     │
     ▼
Preprocessing
     │
     ▼
RUL + HI Prediction
     │
     ├──────────────► Explainability
     │
     ├──────────────► Uncertainty
     │
     ▼
Maintenance Decision
     │
     ▼
Human Feedback
```

## Project Structure

```text
phm_project/
├── data/
│   ├── raw/
│   └── processed/
├── configs/
├── src/
│   ├── preprocessing/
│   ├── models/
│   ├── explainability/
│   ├── uncertainty/
│   ├── decision_engine/
│   ├── hitl/
│   └── utils/
├── scripts/
├── tests/
├── outputs/
├── requirements.txt
├── LICENSE
└── README.md
```

## Requirements

- Python 3.9+
- PyTorch 1.13+
- NumPy
- Pandas
- scikit-learn
- SciPy
- Matplotlib / Seaborn
- PyYAML
- SHAP / Captum

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Dataset

Download the **NASA C-MAPSS Turbofan Engine Degradation Dataset** and place the required files in:

```text
data/raw/
├── train_FD001.txt
├── test_FD001.txt
├── RUL_FD001.txt
└── ...
```

Supported subsets: **FD001, FD002, FD003, FD004**.

## Quick Start

### 1. Preprocess

```bash
python scripts/preprocess.py --subset FD001
```

### 2. Train

```bash
python scripts/train.py --model lstm --subset FD001
```

Supported models:

```text
lstm
gru
transformer
hybrid
```

### 3. Evaluate

```bash
python scripts/evaluate.py --model lstm --subset FD001
```

### 4. Run EDA

```bash
python main.py
```

## Configuration

Experiments are controlled through YAML files in `configs/`.

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

## Evaluation

### RUL

- MAE
- RMSE
- MAPE
- NASA scoring function

### Health Index

- Accuracy
- Precision
- Recall
- F1-score

Results are stored in:

```text
outputs/
├── reports/
├── figures/
├── checkpoints/
└── logs/
```

## Research Status

| Objective | Focus | Status |
|---|---|---|
| O1 | Hybrid deep learning for RUL + HI | ✅ Complete |
| O2 | Uncertainty quantification | 🔄 In Development |
| O3 | Multi-level explainability | 🔄 In Development |
| O4 | Explainable decision intelligence | 🔄 In Development |
| O5 | Human-in-the-loop learning | 🔄 In Development |
| O6 | Multi-dimensional evaluation | 🔄 In Development |

The core preprocessing, training, model, and evaluation pipeline is implemented. The XAI, uncertainty, decision-support, and HITL modules are being developed as the research progresses.

## Testing

```bash
pytest tests/ -v
```

## Citation

```bibtex
@software{phm_xai,
  title  = {PHM-XAI: Explainable Prognostics and Health Monitoring},
  author = {Nagesh Tiwari, Suraj Vishwakarma, Shani Vishwakarma, Vishwa Save},
  year   = {2026}
}
```

## 🤝 Contributing

PHM-XAI is designed to be modular and extensible. Developers can contribute
new models, preprocessing methods, XAI techniques, uncertainty methods,
decision policies, and human-in-the-loop components.

For development setup, project architecture, coding guidelines, testing,
and contribution workflow, see:

👉 [CONTRIBUTING.md](CONTRIBUTING.md)

### Development Areas

- New prognostics models
- Explainable AI methods
- Uncertainty quantification
- Maintenance decision intelligence
- Human-in-the-loop learning
- Evaluation and benchmarking
- Documentation and testing