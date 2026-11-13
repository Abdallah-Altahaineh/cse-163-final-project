"""
Hamsa Ali: data pipeline and EDA for the Kepler KOI cumulative table.

Loads the NASA Exoplanet Archive Kepler cumulative table, keeps only
CONFIRMED and FALSE POSITIVE rows, reports size / missingness /
summary statistics for the variables of interest, and saves two
EDA figures. Run with:  python data_pipeline.py
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

KOI_URL = (
    "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
    "?query=select+*+from+cumulative&format=csv"
)
LOCAL_PATH = os.path.join("data", "koi_cumulative.csv")
FIG_DIR = "figures"
STARS_PATH = os.path.join("data", "stars.csv")

LABEL_COL = "koi_disposition"
KEEP_LABELS = ["CONFIRMED", "FALSE POSITIVE"]
RADIUS_COL = "koi_prad"
PERIOD_COL = "koi_period"
DEPTH_COL = "koi_depth"
NUMERIC_COLS = [RADIUS_COL, PERIOD_COL, DEPTH_COL]
INTEREST_COLS = [LABEL_COL] + NUMERIC_COLS
STAR_COLS = ["kepid", "kepoi_name", LABEL_COL, RADIUS_COL, PERIOD_COL,
             DEPTH_COL, "koi_time0bk", "koi_duration"]


def load_koi(path=LOCAL_PATH, url=KOI_URL):
    """
    Return the raw KOI cumulative table as a DataFrame.

    Reads the cached CSV at `path` if it exists. Otherwise downloads
    the table from `url` and saves a copy at `path`. Lines starting
    with '#' (comments in manual archive downloads) are skipped.
    """
    if os.path.exists(path):
        return pd.read_csv(path, comment="#")
    df = pd.read_csv(url, comment="#")
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def filter_dispositions(df):
    """
    Return a copy of `df` containing only CONFIRMED and
    FALSE POSITIVE rows (candidates are dropped).
    """
    return df[df[LABEL_COL].isin(KEEP_LABELS)].copy()


def missing_summary(df, cols=None):
    """
    Return a DataFrame with the count and percent of missing values
    for each column in `cols` (all columns if None), sorted from
    most to least missing.
    """
    if cols is None:
        cols = list(df.columns)
    counts = df[cols].isna().sum()
    percent = counts / len(df) * 100
    result = pd.DataFrame({"missing": counts, "percent": percent})
    return result.sort_values("missing", ascending=False)


def drop_missing(df, cols):
    """Return `df` without rows that are missing a value in `cols`."""
    return df.dropna(subset=cols).copy()


def summarize_numeric(df, cols=None):
    """
    Return the seven-number summary (mean, std, min, quartiles,
    median, max) for each numeric column in `cols`.
    """
    if cols is None:
        cols = NUMERIC_COLS
    stats = df[cols].describe()
    return stats.loc[["mean", "std", "min", "25%", "50%", "75%", "max"]]


def summarize_categorical(df, col=LABEL_COL):
    """Return the unique values of `col` with their counts."""
    return df[col].value_counts()


def make_stars_table(df):
    """
    Return one row per star (the first KOI listed for each kepid) with
    the columns needed to choose which stars to download and
    sonify: star ID, KOI name, disposition, radius, period, depth,
    transit epoch, and transit duration.
    """
    return df.drop_duplicates(subset="kepid")[STAR_COLS].reset_index(
        drop=True)


def plot_radius_by_disposition(df, out_path):
    """
    Save overlaid histograms of planet radius (log scale) for
    CONFIRMED vs FALSE POSITIVE rows to `out_path`.
    """
    data = df[df[RADIUS_COL] > 0]
    bins = np.logspace(
        np.log10(data[RADIUS_COL].min()),
        np.log10(data[RADIUS_COL].max()),
        40,
    )
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label in KEEP_LABELS:
        subset = data[data[LABEL_COL] == label]
        ax.hist(subset[RADIUS_COL], bins=bins, alpha=0.6, label=label)
    ax.set_xscale("log")
    ax.set_xlabel("Planet radius (Earth radii, log scale)")
    ax.set_ylabel("Number of KOIs")
    ax.set_title("Planet Radius by Disposition")
    ax.legend(title="Disposition")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_depth_vs_period(df, out_path):
    """
    Save a log-log scatter plot of transit depth vs. orbital period,
    colored by disposition, to `out_path`.
    """
    data = df[(df[DEPTH_COL] > 0) & (df[PERIOD_COL] > 0)]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label in KEEP_LABELS:
        subset = data[data[LABEL_COL] == label]
        ax.scatter(subset[PERIOD_COL], subset[DEPTH_COL],
                   s=6, alpha=0.5, label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Orbital period (days, log scale)")
    ax.set_ylabel("Transit depth (ppm, log scale)")
    ax.set_title("Transit Depth vs. Orbital Period")
    ax.legend(title="Disposition")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    raw = load_koi()
    print("Raw table shape (rows, cols):", raw.shape)

    df = filter_dispositions(raw)
    print("After keeping CONFIRMED / FALSE POSITIVE:", df.shape)

    print("\nMissing data, variables of interest:")
    print(missing_summary(df, INTEREST_COLS))

    clean = drop_missing(df, INTEREST_COLS)
    print("\nAfter dropping rows missing a variable of interest:",
          clean.shape)

    print("\nSeven-number summary:")
    print(summarize_numeric(clean))
    print("\nDisposition counts:")
    print(summarize_categorical(clean))

    os.makedirs(FIG_DIR, exist_ok=True)
    plot_radius_by_disposition(
        clean, os.path.join(FIG_DIR, "radius_by_disposition.png"))
    plot_depth_vs_period(
        clean, os.path.join(FIG_DIR, "depth_vs_period.png"))
    clean.to_csv(os.path.join("data", "koi_clean.csv"), index=False)
    stars = make_stars_table(clean)
    stars.to_csv(STARS_PATH, index=False)
    print("\nSaved figures to", FIG_DIR, "and data/koi_clean.csv")
    print("Saved", len(stars), "stars to", STARS_PATH)


if __name__ == "__main__":
    main()
