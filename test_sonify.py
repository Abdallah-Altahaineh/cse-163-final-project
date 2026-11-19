"""
Abdallah Altahaineh
CSE 163 Final Project

Tests for lightcurves.py and sonify.py. Uses a small made-up light
curve in tests/sample_lightcurve.csv and made-up curves built in code,
where the right answers are known by hand.

Run from the top of the repository: python test_sonify.py
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd

from lightcurves import (count_missing, load_light_curve,
                         seven_number_summary, window_around_dip)
from sonify import (AMPLITUDE, BASE_HZ, brightness_to_hz,
                    brightness_to_semitones, clearest_confirmed_star,
                    dip_center, fold_light_curve, pitch_stats,
                    semitones_to_hz, synthesize, transit_center)

SAMPLE_PATH = Path("tests") / "sample_lightcurve.csv"


def close(a: float, b: float, tol: float = 1e-6) -> bool:
    """
    Returns True if a and b are within tol of each other.
    """
    return math.isclose(a, b, abs_tol=tol)


def test_count_missing() -> None:
    """
    The sample file has exactly one blank brightness value and no other
    blanks.
    """
    df = load_light_curve(SAMPLE_PATH)
    missing = count_missing(df)
    assert missing["brightness"] == 1
    assert missing["brightness_err"] == 0
    assert missing["time_days"] == 0


def test_seven_number_summary() -> None:
    """
    Brightness values in the sample are 1.0, 0.99, 1.01 and 1.0, so the
    mean and median are 1.0, the minimum is 0.99 and the maximum 1.01.
    """
    df = load_light_curve(SAMPLE_PATH)
    summary = seven_number_summary(df, ["brightness"])["brightness"]
    assert list(summary.index) == ["mean", "std", "min", "25%", "50%",
                                   "75%", "max"]
    assert close(summary["mean"], 1.0)
    assert close(summary["50%"], 1.0)
    assert close(summary["min"], 0.99)
    assert close(summary["max"], 1.01)


def test_window_around_dip() -> None:
    """
    A flat light curve with one dip at day 5 should be cut down to the
    days around day 5 only.
    """
    times = np.arange(0, 10, 0.1)
    brightness = np.where(np.abs(times - 5) < 0.25, 0.99, 1.0)
    df = pd.DataFrame({"time_days": times, "brightness": brightness})
    window = window_around_dip(df, days=1.0)
    assert window["time_days"].min() > 3.9
    assert window["time_days"].max() < 6.1
    assert window["brightness"].min() == 0.99


def test_window_around_given_center() -> None:
    """
    When a center time is given, the window is built around it even if
    the lowest point is somewhere else.
    """
    times = np.arange(0, 10, 0.1)
    brightness = np.where(np.abs(times - 5) < 0.25, 0.99, 1.0)
    df = pd.DataFrame({"time_days": times, "brightness": brightness})
    window = window_around_dip(df, days=1.0, center=2.0)
    assert window["time_days"].min() > 0.9
    assert window["time_days"].max() < 3.1
    assert window["brightness"].min() == 1.0


def test_transit_center() -> None:
    """
    A star with period 2 days that dims between phase 0.3 and 0.35 of
    each orbit has transits at about 0.65, 2.65, 4.65 and so on. With
    data from day 0 to day 10, the transit closest to the middle is the
    one near day 4.65.
    """
    times = np.arange(0, 10, 0.001)
    phase = (times % 2.0) / 2.0
    brightness = np.where((phase >= 0.3) & (phase < 0.35), 0.99, 1.0)
    df = pd.DataFrame({"time_days": times, "brightness": brightness})
    center = transit_center(df, 2.0, n_bins=100)
    assert 4.55 < center < 4.75
    assert close(brightness[int(center * 1000)], 0.99)


def test_dip_center() -> None:
    """
    A dip covering indexes 4, 5 and 6 has its middle at index 5.
    """
    values = np.array([1, 1, 1, 1, 0.99, 0.99, 0.99, 1, 1, 1])
    assert dip_center(values) == 5


def test_brightness_to_semitones() -> None:
    """
    Normal brightness is no change, a 1% dip is 10 semitones, and a 3%
    dip (30 semitones) is capped at 24.
    """
    assert close(brightness_to_semitones(1.0), 0.0)
    assert close(brightness_to_semitones(0.99), 10.0)
    assert close(brightness_to_semitones(0.97), 24.0)
    assert close(brightness_to_semitones(1.001), -1.0)


def test_semitones_to_hz() -> None:
    """
    0 semitones is the base pitch, 12 is one octave lower (half the
    frequency), and 24 is two octaves lower (a quarter).
    """
    assert close(semitones_to_hz(0), BASE_HZ)
    assert close(semitones_to_hz(12), BASE_HZ / 2)
    assert close(semitones_to_hz(24), BASE_HZ / 4)


def test_brightness_to_hz() -> None:
    """
    Brightness 0.99 is 10 semitones below 440 Hz, about 246.94 Hz.
    """
    assert close(brightness_to_hz(1.0), 440.0)
    assert close(brightness_to_hz(0.99), 246.94, tol=0.01)
    assert close(brightness_to_hz(0.5), 110.0)


def test_fold_light_curve() -> None:
    """
    A star with period 1 day that dims to 0.99 for part of every orbit
    should fold into one dip of 0.99, moved to the middle of the orbit.
    """
    times = np.arange(0, 10, 0.001)
    phase = times % 1.0
    brightness = np.where((phase >= 0.3) & (phase < 0.35), 0.99, 1.0)
    df = pd.DataFrame({"time_days": times, "brightness": brightness})
    folded = fold_light_curve(df, 1.0, n_bins=100)
    assert len(folded) == 100
    assert close(folded["brightness"][50], 0.99, tol=1e-4)
    assert close(folded["brightness"].min(), 0.99, tol=1e-4)
    assert close(folded["brightness"][0], 1.0)
    assert close(folded["brightness"].max(), 1.0)
    assert close(folded["phase"].min(), -0.495)
    assert close(folded["phase"].max(), 0.495)


def test_synthesize() -> None:
    """
    One second of audio at 1000 samples per second has 1000 samples,
    never louder than AMPLITUDE, and starts and ends silent.
    """
    audio = synthesize(np.full(10, 440.0), duration_s=1.0,
                       sample_rate=1000)
    assert len(audio) == 1000
    assert np.abs(audio).max() <= AMPLITUDE
    assert close(audio[0], 0.0)
    assert close(audio[-1], 0.0)


def test_pitch_stats() -> None:
    """
    A pitch curve that sits at 440 Hz and drops to 220 Hz at the transit
    has a 220 Hz, 12 semitone drop, no noise, and is audible.
    """
    phase = np.linspace(-0.5, 0.5, 101)
    pitch = np.where(np.abs(phase) < 0.01, 220.0, 440.0)
    folded = pd.DataFrame({"phase": phase, "pitch_hz": pitch})
    stats = pitch_stats(folded)
    assert close(stats["baseline_hz"], 440.0)
    assert close(stats["pitch_drop_hz"], 220.0)
    assert close(stats["pitch_drop_semitones"], 12.0)
    assert close(stats["noise_hz"], 0.0)
    assert stats["dip_to_noise"] == float("inf")
    assert stats["audible"]


def test_pitch_stats_too_small() -> None:
    """
    A drop of 1 Hz from 440 Hz is about 4 cents, below the hearing
    threshold, so it is not audible.
    """
    phase = np.linspace(-0.5, 0.5, 101)
    pitch = np.where(np.abs(phase) < 0.01, 439.0, 440.0)
    folded = pd.DataFrame({"phase": phase, "pitch_hz": pitch})
    assert not pitch_stats(folded)["audible"]


def test_clearest_confirmed_star() -> None:
    """
    Star 2 has the highest dip_to_noise overall but is a false positive,
    so the clearest confirmed planet is star 3.
    """
    sonified = pd.DataFrame({"star_id": [1, 2, 3],
                             "dip_to_noise": [5.0, 100.0, 20.0]})
    stats = pd.DataFrame({"kepid": [1, 2, 3],
                          "koi_disposition": ["CONFIRMED", "FALSE POSITIVE",
                                              "CONFIRMED"]})
    assert clearest_confirmed_star(sonified, stats) == 3


def main() -> None:
    """
    Runs every test in this file.
    """
    test_count_missing()
    test_seven_number_summary()
    test_window_around_dip()
    test_window_around_given_center()
    test_transit_center()
    test_dip_center()
    test_brightness_to_semitones()
    test_semitones_to_hz()
    test_brightness_to_hz()
    test_fold_light_curve()
    test_synthesize()
    test_pitch_stats()
    test_pitch_stats_too_small()
    test_clearest_confirmed_star()
    print("All tests passed!")


if __name__ == "__main__":
    main()
