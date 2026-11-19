"""
Abdallah Altahaineh
CSE 163 Final Project

Exploratory data analysis of the Kepler light curves. Loads the raw and
cleaned light curves that the data pipeline saved, counts missing data,
and summarizes the brightness measurements. The plotting function here
is used by sonify.py, which knows each planet's orbital period and so
can center the plot on a real transit.

Run from the top of the repository: python lightcurves.py
"""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "lightcurves" / "raw"
CLEAN_DIR = DATA_DIR / "lightcurves" / "clean"
STATS_PATH = DATA_DIR / "light_curve_stats.csv"
FIGURES_DIR = Path("figures")

VARIABLES = ["brightness", "brightness_err"]
DAYS_AROUND_DIP = 1.5


def load_light_curve(path: Path) -> pd.DataFrame:
    """
    Reads one light curve CSV with the columns time_days, brightness and
    brightness_err, and returns it as a DataFrame.
    """
    return pd.read_csv(path)


def load_all(folder: Path) -> pd.DataFrame:
    """
    Reads every light curve CSV in the given folder and returns them as
    one DataFrame, with a star_id column taken from each file name.
    """
    frames = []
    for path in sorted(folder.glob("*.csv")):
        df = load_light_curve(path)
        df["star_id"] = int(path.stem)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def count_missing(df: pd.DataFrame) -> pd.Series:
    """
    Returns the number of missing values in each column of the given
    DataFrame.
    """
    return df.isna().sum()


def percent_missing(df: pd.DataFrame) -> pd.Series:
    """
    Returns the percent of rows that are missing in each column of the
    given DataFrame.
    """
    return count_missing(df) / len(df) * 100


def seven_number_summary(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Returns the mean, standard deviation, minimum, first quartile,
    median, third quartile and maximum of each given column.
    """
    return df[columns].describe().drop("count")


def window_around_dip(df: pd.DataFrame, days: float = DAYS_AROUND_DIP,
                      center: Optional[float] = None) -> pd.DataFrame:
    """
    Returns the rows within the given number of days of center, the time
    of a transit. If center is not given, the light curve's lowest point
    is used instead, found with a rolling median so a single noisy
    measurement does not count as the dip.
    """
    if center is None:
        smooth = df["brightness"].rolling(5, center=True).median()
        center = df.loc[smooth.idxmin(), "time_days"]
    in_window = (df["time_days"] > center - days) & \
        (df["time_days"] < center + days)
    return df[in_window]


def plot_light_curve(df: pd.DataFrame, star_id: int, path: Path,
                     center: Optional[float] = None) -> None:
    """
    Plots brightness over the days around one transit of one star and
    saves the plot to the given path. center is the time of the transit;
    if it is not given, the light curve's lowest point is used.
    """
    window = window_around_dip(df, center=center)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.lineplot(data=window, x="time_days", y="brightness", ax=ax)
    ax.set_title(f"Light curve of Kepler star {star_id} during a transit")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Brightness (1.0 = normal)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    """
    Prints the size, missing data and summaries of the light curves.
    """
    clean = load_all(CLEAN_DIR)
    raw = load_all(RAW_DIR)
    stats = pd.read_csv(STATS_PATH)

    print("Number of stars:", clean["star_id"].nunique())
    print("Cleaned rows and columns:", clean.shape)
    print("Raw rows and columns:", raw.shape)
    print()

    print("Missing values in the raw light curves:")
    print(count_missing(raw))
    print("Percent missing:")
    print(percent_missing(raw))
    print("Missing values after cleaning:")
    print(count_missing(clean))
    print()

    print("Seven-number summary of the cleaned light curves:")
    print(seven_number_summary(clean, VARIABLES))
    print()

    print("Missing data per star:")
    print(seven_number_summary(stats, ["n_raw", "n_missing", "pct_missing"]))
    print()

    print("Stars by disposition:")
    print(stats["koi_disposition"].value_counts())


if __name__ == "__main__":
    main()
