"""Preprocessing regression tests."""

from __future__ import annotations

import pytest

from src.preprocessing.loader import CMAPSSLoader
from src.preprocessing.windowing import WindowGenerator


@pytest.mark.parametrize("subset", ["FD001", "FD002", "FD003", "FD004"])
def test_test_windows_keep_all_engines(subset: str):
    """Short test engines must not be dropped when forming final windows."""
    _, test_df, rul_df = CMAPSSLoader().load_dataset(subset)

    max_cycles = test_df.groupby("unit_number")["time_in_cycles"].transform("max")
    test_df["RUL"] = (
        max_cycles - test_df["time_in_cycles"]
        + rul_df["RUL"].reindex(test_df["unit_number"] - 1).values
    ).clip(upper=125)
    test_df["HI"] = (
        test_df.groupby("unit_number")["RUL"]
        .transform(lambda x: x / x.max())
        .fillna(0)
    )

    window_size = 40
    generator = WindowGenerator(window_size=window_size, stride=1)

    X, _, _, engine_ids = generator.create_test_windows(test_df)

    assert X.shape[0] == test_df["unit_number"].nunique()
    assert engine_ids.shape[0] == test_df["unit_number"].nunique()
    assert X.shape[1:] == (window_size, X.shape[2])
