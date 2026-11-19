"""
Hamza Ali - CSE 163 Final Project
Tests for light_curves.py using made-up light curves with known answers.
"""

import numpy as np
import pandas as pd

import light_curves as lc


def fake_raw(n=3000, seed=0):
    """
    Makes a raw light curve with a slow trend, small noise, a 1% transit
    every 5 days, 20 missing flux values and one large upward outlier.
    """
    rng = np.random.default_rng(seed)
    time = np.arange(n) * 0.02
    flux = 1000 * (1 + 0.002 * np.sin(time / 20))
    flux = flux * (1 + rng.normal(0, 0.0003, n))
    flux[(time % 5) < 0.3] *= 0.99
    flux[rng.choice(n, 20, replace=False)] = np.nan
    flux[1500] = 1500.0
    return pd.DataFrame({"time": time, "flux": flux,
                         "flux_err": np.full(n, 0.3)})


def test_cleaning_removes_missing_and_outlier():
    """Cleaning drops missing rows and the large outlier."""
    raw = fake_raw()
    clean = lc.clean_light_curve(raw, window=101)
    assert clean["flux"].isna().sum() == 0
    assert clean["flux"].max() < 1.01
    assert len(clean) <= len(raw) - 20


def test_cleaning_centers_flux_on_one():
    """The median cleaned brightness is 1.0."""
    clean = lc.clean_light_curve(fake_raw(), window=101)
    assert abs(clean["flux"].median() - 1.0) < 1e-3


def test_cleaning_keeps_transit_depth():
    """A 1% transit is still about 1% deep after cleaning."""
    raw = fake_raw()
    clean = lc.clean_light_curve(raw, window=301)
    inside = clean[(clean["time"] % 5) < 0.3]["flux"].median()
    outside = clean[(clean["time"] % 5) > 1.0]["flux"].median()
    assert abs((outside - inside) - 0.01) < 0.002


def test_cleaning_scales_uncertainty():
    """The uncertainty is divided by the median flux, about 1000."""
    clean = lc.clean_light_curve(fake_raw(), window=101)
    assert abs(clean["flux_err"].median() - 0.0003) < 0.0001


def test_light_curve_stats():
    """Missing counts and percents match the 20 missing values."""
    raw = fake_raw()
    clean = lc.clean_light_curve(raw, window=101)
    stats = lc.light_curve_stats(raw, clean)
    assert stats["n_raw"] == 3000
    assert stats["n_missing"] == 20
    assert abs(stats["pct_missing"] - 100 * 20 / 3000) < 1e-9
    assert stats["n_clean"] == len(clean)


def test_fold_phase():
    """Phases are days from the nearest transit centre, hand-checked."""
    phase = lc.fold_phase([2.0, 12.0, 14.5, 9.0], 10.0, 2.0)
    assert np.allclose(phase, [0.0, 0.0, 2.5, -3.0])


def test_select_sample_balanced_and_reproducible():
    """The sample is balanced with one KOI per star, same every time."""
    ids = np.arange(100)
    koi = pd.DataFrame({
        "kepid": np.concatenate([ids, ids[:10]]),
        "koi_disposition": ["CONFIRMED"] * 50 + ["FALSE POSITIVE"] * 60})
    first = lc.select_sample(koi, n_per_class=5, seed=1)
    again = lc.select_sample(koi, n_per_class=5, seed=1)
    assert len(first) == 10
    assert first["kepid"].is_unique
    assert (first["koi_disposition"] == "CONFIRMED").sum() == 5
    assert first["kepid"].tolist() == again["kepid"].tolist()


def test_saved_column_names():
    """Saving and loading renames columns both ways."""
    df = pd.DataFrame({"time": [1.0], "flux": [1.0], "flux_err": [0.1]})
    saved = lc.to_saved_format(df)
    assert list(saved.columns) == ["time_days", "brightness",
                                   "brightness_err"]
    assert list(lc.from_saved_format(saved).columns) == list(df.columns)


def test_plots_are_saved(tmp_path):
    """Both plotting functions write an image file."""
    raw = fake_raw()
    clean = lc.clean_light_curve(raw, window=101)
    first = tmp_path / "raw_vs_clean.png"
    second = tmp_path / "folded.png"
    lc.plot_raw_vs_clean(raw, clean, 1, str(first))
    lc.plot_folded_transit(clean, 5.0, 0.0, 7.2, 1, str(second))
    assert first.stat().st_size > 0 and second.stat().st_size > 0


def main():
    """Runs every test in this file when run as a script."""
    import tempfile
    from pathlib import Path
    test_cleaning_removes_missing_and_outlier()
    test_cleaning_centers_flux_on_one()
    test_cleaning_keeps_transit_depth()
    test_cleaning_scales_uncertainty()
    test_light_curve_stats()
    test_fold_phase()
    test_select_sample_balanced_and_reproducible()
    test_saved_column_names()
    with tempfile.TemporaryDirectory() as folder:
        test_plots_are_saved(Path(folder))
    print("All light_curves tests passed.")


if __name__ == "__main__":
    main()
