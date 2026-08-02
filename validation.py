from pathlib import Path
from src.preprocessing.validator import DatasetValidator
from src.preprocessing.visualization import DatasetVisualizer

_REPORT_DIR = Path("outputs/reports")
_FIGURE_DIR = Path("outputs/figures")

def validate_dataset(df):
    DatasetValidator(output_dir=_REPORT_DIR).validate(df, "Training")

def dataset_summary(df, name="Training"):
    v = DatasetValidator(output_dir=_REPORT_DIR)
    summary = v.summarize(df, name)
    v.save_report(summary, f"{name.lower()}_summary.csv")
    print(summary.to_string(index=False))

def plot_engine_cycles(df):
    DatasetVisualizer(output_dir=_FIGURE_DIR).plot_engine_lifetime(df)

def plot_sensor_trends(df, sensor, engines=None):
    engines = engines or [1, 2, 3]
    DatasetVisualizer(output_dir=_FIGURE_DIR).plot_sensor_trend(df, sensor=sensor, engines=engines)

def plot_correlation(df):
    DatasetVisualizer(output_dir=_FIGURE_DIR).plot_correlation(df)

def check_sensor_variance(df):
    variance = DatasetVisualizer(output_dir=_FIGURE_DIR).sensor_variance(df)
    print(variance.to_string(index=False))
