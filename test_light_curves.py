"""
Tests for light_curves.py using synthetic light curves whose correct
answers are known by construction. Run with:
python test_light_curves.py
"""
import os
import tempfile

import matplotlib
import numpy as np
import pandas as pd

import light_curves as lc

matplotlib.use("Agg")

PERIOD = 5.0
T0 = 2.0
DEPTH = 0.01


def make_light_curve():
    """
    2000 cadences over ~40 days: slow sinusoidal trend, small noise,
    a 1% transit every 5 days, 20 NaN fluxes, and one huge outlier.
    """
    rng = np.random.default_rng(0)
    time = np.arange(2000) * 0.02043 + 1.0
    trend = 1000 * (1 + 0.02 * np.sin(time / 6))
    flux = trend * (1 + rng.normal(0, 0.0005, len(time)))
    phase = lc.fold_phase(time, PERIOD, T0)
    flux = np.where(np.abs(phase) < 0.1, flux * (1 - DEPTH), flux)
    flux[100:120] = np.nan
    flux[500] = flux[500] * 1.5
    return pd.DataFrame({"time": time, "flux": flux,
                         "flux_err": 1.0})


def test_clean_removes_nans_and_normalizes():
    raw = make_light_curve()
    clean = lc.clean_light_curve(raw)
    assert clean["flux"].isna().sum() == 0
    assert len(clean) <= len(raw) - 20
    assert abs(clean["flux"].median() - 1.0) < 0.001


def test_clean_removes_upward_outlier():
    raw = make_light_curve()
    clean = lc.clean_light_curve(raw)
    assert clean["flux"].max() < 1.05


def test_clean_keeps_transit_dip():
    raw = make_light_curve()
    clean = lc.clean_light_curve(raw)
    phase = lc.fold_phase(clean["time"], PERIOD, T0)
    in_transit = clean["flux"][np.abs(phase) < 0.05]
    out_transit = clean["flux"][np.abs(phase) > 0.5]
    drop = out_transit.median() - in_transit.median()
    assert abs(drop - DEPTH) < 0.002


def test_clean_flattens_trend():
    raw = make_light_curve()
    clean = lc.clean_light_curve(raw)
    phase = lc.fold_phase(clean["time"], PERIOD, T0)
    out_transit = clean["flux"][np.abs(phase) > 0.5]
    raw_spread = (raw["flux"] / raw["flux"].median()).std()
    assert out_transit.std() < raw_spread / 10


def test_light_curve_stats():
    raw = make_light_curve()
    clean = lc.clean_light_curve(raw)
    stats = lc.light_curve_stats(raw, clean)
    assert stats["n_raw"] == 2000
    assert stats["n_missing"] == 20
    assert abs(stats["pct_missing"] - 1.0) < 1e-9
    assert stats["n_clean"] == len(clean)
    assert stats["flux_min"] < 1 - DEPTH / 2


def test_fold_phase():
    result = lc.fold_phase([2.0, 7.0, 4.4, 5.1], 5.0, 2.0)
    expected = np.array([0.0, 0.0, 2.4, -1.9])
    assert np.allclose(result, expected)
    assert result.min() >= -2.5 and result.max() < 2.5


def test_select_sample():
    koi = pd.DataFrame({
        "kepid": list(range(10)) + [0, 1],
        "koi_disposition": ["CONFIRMED"] * 5 + ["FALSE POSITIVE"] * 5
        + ["CONFIRMED", "FALSE POSITIVE"],
    })
    sample = lc.select_sample(koi, n_per_class=3, seed=1)
    counts = sample["koi_disposition"].value_counts()
    assert counts["CONFIRMED"] == 3
    assert counts["FALSE POSITIVE"] == 3
    assert sample["kepid"].is_unique
    again = lc.select_sample(koi, n_per_class=3, seed=1)
    assert list(again["kepid"]) == list(sample["kepid"])


def test_download_reads_cache():
    raw = make_light_curve()
    with tempfile.TemporaryDirectory() as folder:
        raw.to_csv(os.path.join(folder, "123.csv"), index=False)
        loaded = lc.download_light_curve(123, cache_dir=folder)
    assert loaded.shape == raw.shape


def test_plots_create_files():
    raw = make_light_curve()
    clean = lc.clean_light_curve(raw)
    with tempfile.TemporaryDirectory() as folder:
        first = os.path.join(folder, "a.png")
        second = os.path.join(folder, "b.png")
        lc.plot_raw_vs_clean(raw, clean, 123, first)
        lc.plot_folded_transit(clean, PERIOD, T0, 5.0, 123, second)
        assert os.path.getsize(first) > 0
        assert os.path.getsize(second) > 0


def main():
    test_clean_removes_nans_and_normalizes()
    test_clean_removes_upward_outlier()
    test_clean_keeps_transit_dip()
    test_clean_flattens_trend()
    test_light_curve_stats()
    test_fold_phase()
    test_select_sample()
    test_download_reads_cache()
    test_plots_create_files()
    print("All tests passed.")


if __name__ == "__main__":
    main()
