"""
Hamza Ali - CSE 163 Final Project
EDA of the joined dataset: the KOI catalog (data/stars.csv), the cleaned
light curves (data/lightcurves/clean) and the sonified output
(output/sonified.csv), joined on the Kepler ID.

For each star we measure the transit depth actually seen in the cleaned
light curve (the "observed" depth) and compare it with the depth NASA
lists in the catalog. Run with: python join_eda.py
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

from light_curves import fold_phase

STARS_PATH = os.path.join("data", "stars.csv")
SONIFIED_PATH = os.path.join("output", "sonified.csv")
CLEAN_DIR = os.path.join("data", "lightcurves", "clean")
JOINED_PATH = os.path.join("output", "joined.csv")
FIG_DIR = "figures"
LABEL_COL = "koi_disposition"
COLORS = {"CONFIRMED": "tab:blue", "FALSE POSITIVE": "tab:orange"}
MIN_POINTS = 3


def observed_depth_ppm(df, period, t0, duration_hours):
    """
    Returns the observed transit depth in parts per million: mean
    brightness outside the transit minus mean brightness inside it.
    "Inside" is within half the KOI transit duration of the transit
    center; "outside" is more than one full duration away. Means are
    used because the median of a noisy star can sit at exactly 1.0.
    Returns NaN if either side has fewer than MIN_POINTS measurements.
    """
    duration_days = duration_hours / 24
    phase = pd.Series(fold_phase(df["time_days"], period, t0)).abs()
    inside = df.loc[(phase <= duration_days / 2).to_numpy(), "brightness"]
    outside = df.loc[(phase > duration_days).to_numpy(), "brightness"]
    if len(inside) < MIN_POINTS or len(outside) < MIN_POINTS:
        return float("nan")
    return (outside.mean() - inside.mean()) * 1e6


def measure_depths(stars, clean_dir=CLEAN_DIR):
    """
    Returns a DataFrame with columns kepid and observed_depth_ppm, one
    row per star in stars that has a cleaned light curve in clean_dir.
    """
    rows = []
    for row in stars.itertuples():
        path = os.path.join(clean_dir, f"{row.kepid}.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        depth = observed_depth_ppm(df, row.koi_period, row.koi_time0bk,
                                   row.koi_duration)
        rows.append({"kepid": row.kepid, "observed_depth_ppm": depth})
    return pd.DataFrame(rows)


def build_joined(stars, sonified, depths):
    """
    Joins the first KOI of each star, the sonified measurements and the
    observed depths into one table with one row per star. Raises a
    ValueError if the join is not one-to-one.
    """
    first = stars.drop_duplicates("kepid")
    sonified = sonified.rename(columns={"star_id": "kepid"})
    joined = first.merge(sonified, on="kepid", how="inner",
                         validate="1:1")
    return joined.merge(depths, on="kepid", how="inner", validate="1:1")


def depth_correlation(joined):
    """Returns the Spearman correlation of catalog vs observed depth."""
    valid = joined.dropna(subset=["koi_depth", "observed_depth_ppm"])
    return stats.spearmanr(valid["koi_depth"],
                           valid["observed_depth_ppm"])[0]


def plot_catalog_vs_observed(joined, path):
    """Scatter of NASA's catalog depth vs depth seen in our light curves."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for label, group in joined.groupby(LABEL_COL):
        ax.scatter(group["koi_depth"], group["observed_depth_ppm"],
                   label=label.title(), color=COLORS[label])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("NASA's catalog depth (ppm)")
    ax.set_ylabel("Observed depth in cleaned light curve (ppm)")
    ax.set_title("Catalog vs. Observed Transit Depth")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_depth_by_class(joined, path):
    """Box plot of observed depth for confirmed vs false positive stars."""
    labels = ["CONFIRMED", "FALSE POSITIVE"]
    data = [joined.loc[joined[LABEL_COL] == lab, "observed_depth_ppm"]
            for lab in labels]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.boxplot(data)
    ax.set_xticks([1, 2], ["Confirmed Planet", "False Positive"])
    ax.set_yscale("log")
    ax.set_xlabel("NASA's Label")
    ax.set_ylabel("Observed depth (ppm)")
    ax.set_title("Observed Transit Depth by NASA's Label")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    """Builds the joined table, prints summaries and saves two figures."""
    stars = pd.read_csv(STARS_PATH)
    sonified = pd.read_csv(SONIFIED_PATH)
    first = stars[stars["kepid"].isin(sonified["star_id"])]
    first = first.drop_duplicates("kepid")
    depths = measure_depths(first)
    joined = build_joined(stars, sonified, depths)
    print(f"Joined shape (rows, columns): {joined.shape}")
    print(f"Missing values:\n{joined.isna().sum().loc[lambda s: s > 0]}")
    print(joined.groupby(LABEL_COL)["observed_depth_ppm"].describe())
    print(f"Spearman (catalog vs observed depth): "
          f"{depth_correlation(joined):.3f}")
    os.makedirs(FIG_DIR, exist_ok=True)
    joined.to_csv(JOINED_PATH, index=False)
    plot_catalog_vs_observed(
        joined, os.path.join(FIG_DIR, "catalog_vs_observed_depth.png"))
    plot_depth_by_class(
        joined, os.path.join(FIG_DIR, "observed_depth_by_class.png"))


if __name__ == "__main__":
    main()
