"""
Abdallah Altahaineh
CSE 163 Final Project

Exploratory data analysis of the Kepler light curves. Loads the raw and
cleaned light curves that the data pipeline saved, counts missing data,
summarizes the brightness measurements, and plots one light curve with
a visible transit dip.

Run from the top of the repository: python lightcurves.py
"""

from pathlib import Path

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


def deepest_confirmed_star(stats: pd.DataFrame) -> int:
    """
    Returns the Kepler ID of the confirmed planet whose star reaches the
    lowest brightness, which makes its transit dip the easiest to see.
    """
    confirmed = stats[stats["koi_disposition"] == "CONFIRMED"]
    return int(confirmed.sort_values("flux_min").iloc[0]["kepid"])


def window_around_dip(df: pd.DataFrame,
                      days: float = DAYS_AROUND_DIP) -> pd.DataFrame:
    """
    Returns the rows within the given number of days of the light
    curve's lowest point. A rolling median is used to find that point,
    so a single noisy measurement does not count as the dip.
    """
    smooth = df["brightness"].rolling(5, center=True).median()
    center = df.loc[smooth.idxmin(), "time_days"]
    in_window = (df["time_days"] > center - days) & \
        (df["time_days"] < center + days)
    return df[in_window]


def plot_light_curve(df: pd.DataFrame, star_id: int, path: Path) -> None:
    """
    Plots brightness over time around the deepest transit dip of one
    star and saves the plot to the given path.
    """
    window = window_around_dip(df)
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
    Prints the size, missing data and summaries of the light curves and
    saves a plot of one light curve with a transit dip.
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

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    star_id = deepest_confirmed_star(stats)
    star = clean[clean["star_id"] == star_id]
    path = FIGURES_DIR / f"light_curve_{star_id}.png"
    plot_light_curve(star, star_id, path)
    print()
    print("Saved light curve plot to", path)


if __name__ == "__main__":
    main()
