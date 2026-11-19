"""
Tests for data_pipeline.py using a tiny hand-made DataFrame whose
correct answers are known by inspection. Run with:
python test_data_pipeline.py
"""
import os
import tempfile

import matplotlib
import numpy as np
import pandas as pd

import data_pipeline as dp

matplotlib.use("Agg")


def make_toy():
    """Six KOIs: 2 confirmed, 2 false positive, 1 candidate, 1 junk."""
    return pd.DataFrame({
        "koi_disposition": ["CONFIRMED", "CONFIRMED", "FALSE POSITIVE",
                            "FALSE POSITIVE", "CANDIDATE", "CONFIRMED"],
        "koi_prad": [1.0, 2.0, 10.0, np.nan, 3.0, 4.0],
        "koi_period": [10.0, 20.0, 30.0, 40.0, 50.0, np.nan],
        "koi_depth": [100.0, 200.0, 300.0, 400.0, 500.0, 600.0],
    })


def test_filter_dispositions():
    result = dp.filter_dispositions(make_toy())
    assert len(result) == 5
    assert set(result["koi_disposition"]) == {"CONFIRMED",
                                              "FALSE POSITIVE"}
    assert "CANDIDATE" not in set(result["koi_disposition"])


def test_missing_summary():
    result = dp.missing_summary(make_toy(), dp.NUMERIC_COLS)
    assert result.loc["koi_prad", "missing"] == 1
    assert result.loc["koi_period", "missing"] == 1
    assert result.loc["koi_depth", "missing"] == 0
    assert abs(result.loc["koi_prad", "percent"] - 100 / 6) < 1e-9


def test_drop_missing():
    result = dp.drop_missing(make_toy(), dp.NUMERIC_COLS)
    assert len(result) == 4
    assert result[dp.NUMERIC_COLS].isna().sum().sum() == 0


def test_summarize_numeric():
    df = pd.DataFrame({"koi_prad": [1.0, 2.0, 3.0, 4.0, 5.0],
                       "koi_period": [10.0] * 5,
                       "koi_depth": [1.0, 1.0, 1.0, 1.0, 1.0]})
    result = dp.summarize_numeric(df)
    assert list(result.index) == ["mean", "std", "min", "25%",
                                  "50%", "75%", "max"]
    assert result.loc["mean", "koi_prad"] == 3.0
    assert result.loc["min", "koi_prad"] == 1.0
    assert result.loc["50%", "koi_prad"] == 3.0
    assert result.loc["max", "koi_prad"] == 5.0
    assert result.loc["std", "koi_period"] == 0.0


def test_summarize_categorical():
    result = dp.summarize_categorical(make_toy())
    assert result["CONFIRMED"] == 3
    assert result["FALSE POSITIVE"] == 2
    assert result["CANDIDATE"] == 1


def test_make_stars_table():
    toy = pd.DataFrame({
        "kepid": [1, 1, 2, 3],
        "kepoi_name": ["K1.01", "K1.02", "K2.01", "K3.01"],
        "koi_disposition": ["CONFIRMED", "CONFIRMED", "FALSE POSITIVE",
                            "CONFIRMED"],
        "koi_prad": [1.0, 2.0, 3.0, 4.0],
        "koi_period": [5.0, 6.0, 7.0, 8.0],
        "koi_depth": [100.0, 200.0, 300.0, 400.0],
        "koi_time0bk": [1.0, 1.0, 1.0, 1.0],
        "koi_duration": [2.0, 2.0, 2.0, 2.0],
        "extra": [0, 0, 0, 0],
    })
    stars = dp.make_stars_table(toy)
    assert len(stars) == 3
    assert stars["kepid"].is_unique
    assert list(stars.columns) == dp.STAR_COLS
    assert stars.loc[0, "kepoi_name"] == "K1.01"


def test_load_koi_reads_cached_file():
    toy = make_toy()
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "toy.csv")
        toy.to_csv(path, index=False)
        loaded = dp.load_koi(path=path)
    assert loaded.shape == toy.shape


def test_plots_create_files():
    clean = dp.drop_missing(dp.filter_dispositions(make_toy()),
                            dp.INTEREST_COLS)
    with tempfile.TemporaryDirectory() as folder:
        radius_path = os.path.join(folder, "r.png")
        depth_path = os.path.join(folder, "d.png")
        dp.plot_radius_by_disposition(clean, radius_path)
        dp.plot_depth_vs_period(clean, depth_path)
        assert os.path.getsize(radius_path) > 0
        assert os.path.getsize(depth_path) > 0


def main():
    test_filter_dispositions()
    test_missing_summary()
    test_drop_missing()
    test_summarize_numeric()
    test_summarize_categorical()
    test_make_stars_table()
    test_load_koi_reads_cached_file()
    test_plots_create_files()
    print("All tests passed.")


if __name__ == "__main__":
    main()
