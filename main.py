if __name__ == "__main__":

    from src.preprocessing.loader import CMAPSSLoader
    from src.preprocessing.validator import DatasetValidator
    from src.preprocessing.visualization import DatasetVisualizer
    from src.utils.logger import get_script_logger
    from pathlib import Path

    _report_dir = Path("outputs/reports/EDA")
    _figures_dir = Path("outputs/figures/EDA")

    log = get_script_logger("eda", "eda_FD001")

    log.info("Starting EDA for FD001...")

    train, test, rul = CMAPSSLoader().load_dataset("FD001")

    validator = DatasetValidator(output_dir=_report_dir)
    validator.validate(train, "Training")

    summary = validator.summarize(train, "Training")
    validator.save_report(summary, "training_summary.csv")
    log.info("\n%s", summary.to_string(index=False))

    viz = DatasetVisualizer(output_dir=_figures_dir)
    viz.plot_engine_lifetime(train)
    viz.plot_sensor_trend(train, sensor="sensor_2")
    viz.plot_sensor_trend(train, sensor="sensor_3")
    viz.plot_sensor_trend(train, sensor="sensor_4")
    viz.plot_sensor_trend(train, sensor="sensor_11")
    viz.plot_sensor_trend(train, sensor="sensor_15")
    viz.plot_correlation(train)
    viz.sensor_variance(train)

    log.info("EDA complete. Figures → %s | Reports → %s", _figures_dir, _report_dir)
