"""
Hari - CSE 163 Final Project
Tests for audio_features.py using fake stars with known answers.
"""
import os

import numpy as np
import pandas as pd
from sonify import load_periods


import audio_features as af


def fake_star(depth: float, period: float = 10.0) -> pd.DataFrame:
    """
    Makes a fake light curve with the columns time_days and brightness.
    Every period days the brightness drops by depth for 0.25 days,
    with a little random noise added.
    """
    np.random.seed(0)
    time = np.arange(0, 90, 0.0204)
    brightness = np.ones(len(time))
    for start in np.arange(period, 90, period):
        in_dip = (time > start) & (time < start + 0.25)
        brightness[in_dip] = 1 - depth
    brightness = brightness + np.random.normal(0, 0.0001, len(time))
    return pd.DataFrame({'time_days': time, 'brightness': brightness})


def test_star_features() -> None:
    """
    A star that dims by 0.005 should drop about 5 semitones, since
    the pitch drops 1000 semitones per unit of brightness.
    """
    stats = af.star_features(fake_star(0.005), 10.0)
    assert abs(stats['pitch_drop_semitones'] - 5) < 0.5
    assert abs(stats['baseline_hz'] - 440) < 1
    assert stats['audible']


def test_no_dip() -> None:
    """
    A star with no dip should only drop by its noise, far less than
    the 5 semitones of a real dip.
    """
    stats = af.star_features(fake_star(0.0), 10.0)
    assert stats['pitch_drop_semitones'] < 0.5


def test_build_features_table() -> None:
    """
    Builds the table from the real star folder and checks there is
    one row for each of the 50 stars, with no repeats.
    """
    periods = load_periods('data/stars.csv')
    table = af.build_features_table('data/lightcurves/clean', periods)
    assert len(table) == 50
    assert table['kepid'].nunique() == 50


def main() -> None:
    """Runs every test."""
    test_star_features()
    test_no_dip()
    test_build_features_table()
    print('All tests passed!')


if __name__ == '__main__':
    main()
