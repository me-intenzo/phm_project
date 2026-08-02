if __name__ == "__main__":

    from loader import load_dataset
    from validation import (
        dataset_summary,
        validate_dataset,
        plot_engine_cycles,
        plot_sensor_trends,
        plot_correlation,
        check_sensor_variance,
    )

    train, test, rul = load_dataset()

    validate_dataset(train)

    dataset_summary(train, "Training")

    plot_engine_cycles(train)

    plot_sensor_trends(train, sensor="sensor_2")

    plot_sensor_trends(train, sensor="sensor_3")

    plot_sensor_trends(train, sensor="sensor_4")

    plot_sensor_trends(train, sensor="sensor_11")

    plot_sensor_trends(train, sensor="sensor_15")

    plot_correlation(train)

    check_sensor_variance(train)

    print("\nEDA Complete")
