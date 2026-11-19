"""
Hamza Ali - CSE 163 Final Project
Tests for data_pipeline.py using a small hand-made KOI table.
"""

import numpy as np
import pandas as pd

import data_pipeline as dp


def fake_koi():
    """Makes a 6-row KOI table whose correct answers are known."""
    return pd.DataFrame({
        "kepid": [1, 1, 2, 3, 4, 5],
        "kepoi_name": ["a", "b", "c", "d", "e", "f"],
        "koi_disposition": ["CONFIRMED", "CONFIRMED", "FALSE POSITIVE",
                            "CANDIDATE", "FALSE POSITIVE", "CONFIRMED"],
        "koi_prad": [1.0, 2.0, np.nan, 4.0, 5.0, 3.0],
        "koi_period": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        "koi_depth": [100.0, 200.0, 300.0, 400.0, np.nan, 600.0],
        "koi_time0bk": [1.0] * 6,
        "koi_duration": [2.0] * 6})


def test_filter_dispositions():
    """Only CONFIRMED and FALSE POSITIVE rows are kept."""
    result = dp.filter_dispositions(fake_koi())
    assert len(result) == 5
    assert set(result["koi_disposition"]) == {"CONFIRMED",
                                              "FALSE POSITIVE"}


def test_missing_summary():
    """Missing counts and percents match a hand count."""
    result = dp.missing_summary(fake_koi(), ["koi_prad", "koi_depth"])
    assert result.loc["koi_prad", "missing"] == 1
    assert result.loc["koi_depth", "missing"] == 1
    assert abs(result.loc["koi_prad", "percent"] - 100 / 6) < 1e-9


def test_drop_missing():
    """Dropping rows missing a variable of interest leaves none."""
    result = dp.drop_missing(fake_koi(), ["koi_prad", "koi_depth"])
    assert len(result) == 4
    assert result[["koi_prad", "koi_depth"]].isna().sum().sum() == 0


def test_summarize_numeric():
    """The seven-number summary of 1..5 has the expected values."""
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
    result = dp.summarize_numeric(df, ["x"])
    assert list(result.index) == ["mean", "std", "min", "25%", "50%",
                                  "75%", "max"]
    assert result.loc["mean", "x"] == 3.0
    assert result.loc["50%", "x"] == 3.0
    assert result.loc["max", "x"] == 5.0


def test_summarize_categorical():
    """Label counts match a hand count."""
    counts = dp.summarize_categorical(fake_koi())
    assert counts["CONFIRMED"] == 3
    assert counts["FALSE POSITIVE"] == 2


def test_make_stars_table():
    """The star table has one row per star, keeping the first KOI."""
    stars = dp.make_stars_table(fake_koi())
    assert len(stars) == 5
    assert stars["kepid"].is_unique
    assert stars.loc[stars["kepid"] == 1, "kepoi_name"].iloc[0] == "a"


def test_plots_are_saved(tmp_path):
    """Both plotting functions write an image file."""
    df = dp.filter_dispositions(fake_koi()).dropna()
    first = tmp_path / "radius.png"
    second = tmp_path / "depth.png"
    dp.plot_radius_by_disposition(df, str(first))
    dp.plot_depth_vs_period(df, str(second))
    assert first.stat().st_size > 0 and second.stat().st_size > 0


def main():
    """Runs every test in this file when run as a script."""
    import tempfile
    from pathlib import Path
    test_filter_dispositions()
    test_missing_summary()
    test_drop_missing()
    test_summarize_numeric()
    test_summarize_categorical()
    test_make_stars_table()
    with tempfile.TemporaryDirectory() as folder:
        test_plots_are_saved(Path(folder))
    print("All data_pipeline tests passed.")


if __name__ == "__main__":
    main()
