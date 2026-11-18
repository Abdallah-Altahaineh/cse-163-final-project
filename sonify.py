"""
Abdallah Altahaineh
CSE 163 Final Project

Turns Kepler light curves into sound. Each light curve is folded on its
planet's orbital period, so every transit lines up into one dip, and
brightness is mapped to pitch:

- normal brightness (1.0) plays at 440 Hz
- every 0.1% the star dims lowers the pitch by 1 semitone
- the change is capped at 24 semitones (two octaves)

Writes one WAV file per star, a table of pitch measurements for the
audio feature analysis, and a plot of one star's light curve next to
its pitch curve.

Run from the top of the repository: python sonify.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import soundfile as sf

from lightcurves import CLEAN_DIR, DATA_DIR, FIGURES_DIR, load_light_curve

STARS_PATH = DATA_DIR / "stars.csv"
OUTPUT_DIR = Path("output")
AUDIO_DIR = OUTPUT_DIR / "audio"
SONIFIED_PATH = OUTPUT_DIR / "sonified.csv"

METHOD = "semitone_fold"
BASE_HZ = 440.0
SEMITONES_PER_UNIT = 1000.0
MAX_SEMITONES = 24.0
N_BINS = 2000
SMOOTH_BINS = 5
DURATION_S = 10.0
SAMPLE_RATE = 22050
AMPLITUDE = 0.3
FADE_S = 0.05
OUT_OF_TRANSIT_PHASE = 0.1

# Smallest pitch change, in cents (hundredths of a semitone), that we
# treat as audible. Cite the hearing threshold source in the report.
JND_CENTS = 10.0

ID_COLUMNS = ("star_id", "kepid")
PERIOD_COLUMNS = ("period_days", "koi_period")


def brightness_to_semitones(brightness: np.ndarray) -> np.ndarray:
    """
    Converts brightness (1.0 = normal) to how many semitones the pitch
    drops. Brightening raises the pitch instead. Either way the change
    is capped at MAX_SEMITONES.
    """
    semitones = (1 - np.asarray(brightness, dtype=float)) * SEMITONES_PER_UNIT
    return np.clip(semitones, -MAX_SEMITONES, MAX_SEMITONES)


def semitones_to_hz(semitones: np.ndarray) -> np.ndarray:
    """
    Converts a drop of the given number of semitones below BASE_HZ into
    a frequency in Hz.
    """
    return BASE_HZ * 2 ** (-np.asarray(semitones, dtype=float) / 12)


def brightness_to_hz(brightness: np.ndarray) -> np.ndarray:
    """
    Converts brightness directly to pitch in Hz.
    """
    return semitones_to_hz(brightness_to_semitones(brightness))


def dip_center(values: np.ndarray) -> int:
    """
    Returns the index of the middle of the dip: the average position of
    the values that are closer to the lowest value than to the median.
    Assumes the dip does not wrap around the ends of the array.
    """
    midpoint = (values.min() + np.median(values)) / 2
    return int(round(np.flatnonzero(values < midpoint).mean()))


def fold_light_curve(df: pd.DataFrame, period_days: float,
                     n_bins: int = N_BINS) -> pd.DataFrame:
    """
    Folds a light curve on the given orbital period so every transit
    lands on top of the others, then averages brightness into n_bins
    equal slices of the orbit. Each slice is then averaged with its
    neighbors (SMOOTH_BINS slices in all) to reduce noise, so random
    noise is less likely to be mistaken for the dip. The result is
    shifted so the dip sits in the middle, at phase 0. Returns a
    DataFrame with the columns phase (from -0.5 to 0.5) and brightness.
    """
    phase = (df["time_days"] % period_days) / period_days
    bins = np.minimum((phase * n_bins).astype(int), n_bins - 1)
    binned = df.groupby(bins)["brightness"].mean()
    binned = binned.reindex(range(n_bins)).interpolate(
        limit_direction="both")
    smooth = binned.rolling(SMOOTH_BINS, center=True, min_periods=1).mean()
    values = smooth.to_numpy()
    values = np.roll(values, n_bins // 2 - int(values.argmin()))
    values = np.roll(values, n_bins // 2 - dip_center(values))
    centers = (np.arange(n_bins) + 0.5) / n_bins - 0.5
    return pd.DataFrame({"phase": centers, "brightness": values})


def synthesize(pitch_hz: np.ndarray, duration_s: float = DURATION_S,
               sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """
    Plays the given pitch curve from start to end over duration_s
    seconds as a smooth sine tone, and returns the audio samples. The
    sound fades in and out briefly so it does not click.
    """
    n_samples = int(duration_s * sample_rate)
    positions = np.linspace(0, len(pitch_hz) - 1, n_samples)
    freqs = np.interp(positions, np.arange(len(pitch_hz)), pitch_hz)
    phase = 2 * np.pi * np.cumsum(freqs) / sample_rate
    audio = AMPLITUDE * np.sin(phase)
    fade = min(n_samples // 2, int(FADE_S * sample_rate))
    if fade > 0:
        ramp = np.linspace(0, 1, fade)
        audio[:fade] *= ramp
        audio[-fade:] *= ramp[::-1]
    return audio


def pitch_stats(folded: pd.DataFrame) -> dict:
    """
    Measures a folded pitch curve with the columns phase and pitch_hz.
    Returns the baseline pitch (median outside the transit), how far the
    pitch drops in Hz and in semitones, the noise (standard deviation
    outside the transit), how many times bigger the drop is than the
    noise, and whether the drop is audible.
    """
    outside = folded[folded["phase"].abs() > OUT_OF_TRANSIT_PHASE]
    baseline = outside["pitch_hz"].median()
    lowest = folded["pitch_hz"].min()
    drop_hz = baseline - lowest
    drop_semitones = 12 * np.log2(baseline / lowest)
    noise_hz = outside["pitch_hz"].std()
    if noise_hz > 0:
        dip_to_noise = drop_hz / noise_hz
    else:
        dip_to_noise = np.inf
    return {
        "baseline_hz": baseline,
        "pitch_drop_hz": drop_hz,
        "pitch_drop_semitones": drop_semitones,
        "noise_hz": noise_hz,
        "dip_to_noise": dip_to_noise,
        "audible": bool(drop_semitones * 100 >= JND_CENTS),
    }


def find_column(df: pd.DataFrame, options: tuple) -> str:
    """
    Returns the first column name from options that the DataFrame has.
    """
    for name in options:
        if name in df.columns:
            return name
    raise KeyError(f"None of these columns were found: {options}")


def load_periods(path: Path) -> dict:
    """
    Reads the star table and returns a dictionary from each Kepler ID to
    its planet's orbital period in days. For stars with more than one
    planet, the first one listed is used.
    """
    stars = pd.read_csv(path)
    id_col = find_column(stars, ID_COLUMNS)
    period_col = find_column(stars, PERIOD_COLUMNS)
    stars = stars.dropna(subset=[period_col]).drop_duplicates(id_col)
    return dict(zip(stars[id_col].astype(int), stars[period_col]))


def plot_pitch_curve(folded: pd.DataFrame, star_id: int, path: Path) -> None:
    """
    Plots a folded light curve above its pitch curve, sharing the same
    phase axis, and saves the plot to the given path.
    """
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    sns.lineplot(data=folded, x="phase", y="brightness", ax=top)
    top.set_title("Folded light curve")
    top.set_ylabel("Brightness (1.0 = normal)")
    sns.lineplot(data=folded, x="phase", y="pitch_hz", ax=bottom)
    bottom.set_title("Pitch curve")
    bottom.set_xlabel("Phase of orbit (transit at 0)")
    bottom.set_ylabel("Pitch (Hz)")
    fig.suptitle(f"Kepler star {star_id}: brightness and pitch")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def sonify_star(df: pd.DataFrame, period_days: float) -> tuple:
    """
    Folds one star's light curve, converts it to pitch, and returns the
    folded DataFrame (with a pitch_hz column) and the audio samples.
    """
    folded = fold_light_curve(df, period_days)
    folded["pitch_hz"] = brightness_to_hz(folded["brightness"])
    audio = synthesize(folded["pitch_hz"].to_numpy())
    return folded, audio


def main() -> None:
    """
    Sonifies every cleaned light curve that has a known period, saves the
    audio files and sonified.csv, and plots one star's pitch curve.
    """
    periods = load_periods(STARS_PATH)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    curves = {}
    for path in sorted(CLEAN_DIR.glob("*.csv")):
        star_id = int(path.stem)
        if star_id not in periods:
            print("Skipping", star_id, "because it has no period")
            continue
        folded, audio = sonify_star(load_light_curve(path), periods[star_id])
        audio_path = AUDIO_DIR / f"{star_id}.wav"
        sf.write(audio_path, audio, SAMPLE_RATE)
        row = {"star_id": star_id, "method": METHOD}
        row.update(pitch_stats(folded))
        row["audio_path"] = str(audio_path)
        rows.append(row)
        curves[star_id] = folded

    sonified = pd.DataFrame(rows)
    sonified.to_csv(SONIFIED_PATH, index=False)
    print("Sonified", len(sonified), "stars; saved", SONIFIED_PATH)
    print(sonified.describe())
    print("Audible:", int(sonified["audible"].sum()), "of", len(sonified))

    best = sonified.sort_values("pitch_drop_semitones").iloc[-1]
    star_id = int(best["star_id"])
    path = FIGURES_DIR / f"pitch_curve_{star_id}.png"
    plot_pitch_curve(curves[star_id], star_id, path)
    print("Saved pitch curve plot to", path)


if __name__ == "__main__":
    main()
