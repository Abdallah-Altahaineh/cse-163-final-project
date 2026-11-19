"""
Hamza Ali - CSE 163 Final Project
Tests for join_eda.py using small made-up stars with known answers.
"""

import numpy as np
import pandas as pd

import join_eda


def fake_curve(depth, period=10.0, t0=2.0, duration_hours=6.0):
    """
    Makes a flat light curve (brightness 1.0) sampled every 0.02 days
    for 100 days, with a dip of the given depth during each transit.
    """
    time = np.arange(0, 100, 0.02)
    phase = join_eda.fold_phase(time, period, t0)
    brightness = np.ones_like(time)
    brightness[np.abs(phase) <= duration_hours / 48] -= depth
    return pd.DataFrame({"time_days": time, "brightness": brightness})


def test_observed_depth_matches_known_dip():
    """A 1000 ppm dip must be measured as 1000 ppm."""
    df = fake_curve(0.001)
    depth = join_eda.observed_depth_ppm(df, 10.0, 2.0, 6.0)
    assert abs(depth - 1000) < 1


def test_observed_depth_flat_star_is_zero():
    """A star with no dip has an observed depth of zero."""
    df = fake_curve(0.0)
    assert abs(join_eda.observed_depth_ppm(df, 10.0, 2.0, 6.0)) < 1e-6


def test_observed_depth_too_few_points_is_nan():
    """A curve that never sees the transit gives NaN, not a number."""
    df = fake_curve(0.001).iloc[:50]
    depth = join_eda.observed_depth_ppm(df, 10.0, 2.0, 6.0)
    assert np.isnan(depth)


def test_measure_depths_skips_missing_files(tmp_path):
    """Stars with no cleaned light curve file are left out."""
    fake_curve(0.002).to_csv(tmp_path / "1.csv", index=False)
    stars = pd.DataFrame({
        "kepid": [1, 2], "koi_period": [10.0, 10.0],
        "koi_time0bk": [2.0, 2.0], "koi_duration": [6.0, 6.0]})
    result = join_eda.measure_depths(stars, str(tmp_path))
    assert list(result["kepid"]) == [1]
    assert abs(result["observed_depth_ppm"][0] - 2000) < 1


def test_build_joined_one_row_per_star():
    """Extra KOIs for the same star are dropped, keeping the first."""
    stars = pd.DataFrame({
        "kepid": [1, 1, 2], "koi_disposition": ["CONFIRMED"] * 3,
        "koi_depth": [100.0, 999.0, 300.0]})
    sonified = pd.DataFrame({"star_id": [1, 2], "audible": [True, False]})
    depths = pd.DataFrame({"kepid": [1, 2],
                           "observed_depth_ppm": [90.0, 280.0]})
    joined = join_eda.build_joined(stars, sonified, depths)
    assert len(joined) == 2
    assert joined.loc[joined["kepid"] == 1, "koi_depth"].iloc[0] == 100.0


def test_depth_correlation_perfect_ranking():
    """Depths in the same order give a Spearman correlation of 1."""
    joined = pd.DataFrame({"koi_depth": [1.0, 10.0, 100.0],
                           "observed_depth_ppm": [2.0, 20.0, 50.0]})
    assert join_eda.depth_correlation(joined) == 1.0


def test_plots_are_saved(tmp_path):
    """Both plotting functions write a PNG file."""
    joined = pd.DataFrame({
        "koi_disposition": ["CONFIRMED", "FALSE POSITIVE"] * 2,
        "koi_depth": [100.0, 1000.0, 200.0, 5000.0],
        "observed_depth_ppm": [90.0, 900.0, 180.0, 4000.0]})
    first = tmp_path / "a.png"
    second = tmp_path / "b.png"
    join_eda.plot_catalog_vs_observed(joined, str(first))
    join_eda.plot_depth_by_class(joined, str(second))
    assert first.stat().st_size > 0 and second.stat().st_size > 0


def main():
    """Runs every test in this file when run as a script."""
    import tempfile
    from pathlib import Path
    test_observed_depth_matches_known_dip()
    test_observed_depth_flat_star_is_zero()
    test_observed_depth_too_few_points_is_nan()
    test_build_joined_one_row_per_star()
    test_depth_correlation_perfect_ranking()
    with tempfile.TemporaryDirectory() as folder:
        test_measure_depths_skips_missing_files(Path(folder))
        test_plots_are_saved(Path(folder))
    print("All join_eda tests passed.")


if __name__ == "__main__":
    main()
