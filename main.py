if __name__ == "__main__":

    from src.preprocessing.loader import CMAPSSLoader
    from src.preprocessing.validator import DatasetValidator
    from src.preprocessing.visualization import DatasetVisualizer
    from pathlib import Path

    _REPORT_DIR = Path("outputs/reports")
    _FIGURE_DIR = Path("outputs/figures")

    train, test, rul = CMAPSSLoader().load_dataset("FD001")

    validator = DatasetValidator(output_dir=_REPORT_DIR)
    validator.validate(train, "Training")

    summary = validator.summarize(train, "Training")
    validator.save_report(summary, "training_summary.csv")
    print(summary.to_string(index=False))

    viz = DatasetVisualizer(output_dir=_FIGURE_DIR)
    viz.plot_engine_lifetime(train)
    viz.plot_sensor_trend(train, sensor="sensor_2")
    viz.plot_sensor_trend(train, sensor="sensor_3")
    viz.plot_sensor_trend(train, sensor="sensor_4")
    viz.plot_sensor_trend(train, sensor="sensor_11")
    viz.plot_sensor_trend(train, sensor="sensor_15")
    viz.plot_correlation(train)
    viz.sensor_variance(train)

    print("\nEDA Complete")
