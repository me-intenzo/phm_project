"""Windowing utilities."""


def create_windows(data, window_size: int = 10):
    """Split data into sliding windows."""
    return [data[i:i + window_size] for i in range(0, len(data), window_size)]
