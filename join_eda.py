"""
Hamza Ali
CSE 163 Final Project

EDA of the joined dataset. Joins the sampled KOI table to the
light curve statistics and to a new column, the observed transit depth,
measured from each star's cleaned light curve at the transit times in the
KOI table. Compares the catalog depth to the observed depth and saves
two plots and the joined table.

Run from the top of the repository: python join_eda.py
(after python data_pipeline.py and python light_curves.py)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from light_curves import fold_phase

DATA_DIR = Path("data")
SAMPLE_PATH = DATA_DIR / "sample_kois.csv"
STATS_PATH = DATA_DIR / "light_curve_stats.csv"
CLEAN_DIR = DATA_DIR / "lightcurves" / "clean"
OUTPUT_DIR = Path("output")
JOINED_PATH = OUTPUT_DIR / "joined.csv"
FIGURES_DIR = Path("figures")
DPI = 200

LABELS = ["CONFIRMED", "FALSE POSITIVE"]
KOI_COLUMNS = ["kepid", "kepoi_name", "koi_disposition", "koi_prad",
               "koi_period", "koi_depth", "koi_time0bk", "koi_duration"]
STATS_COLUMNS = ["kepid", "n_raw", "n_missing", "pct_missing", "n_clean",
                 "flux_mean", "flux_std", "flux_min", "flux_max"]

# Measurements within this fraction of the transit duration of the
# transit time count as in transit. Measurements farther than one full
# duration away count as out of transit.
IN_TRANSIT_FRACTION = 0.4
MIN_IN_TRANSIT = 3
MIN_OUT_OF_TRANSIT = 20
PPM = 1e6


def observed_depth_ppm(df: pd.DataFrame, period: float, t0: float,
                       duration_days: float) -> tuple:
    """
    Measures how deep the transits are in a cleaned light curve with the
    columns time_days and brightness. Every transit is lined up using
    the period and first transit time t0 from the KOI table. Returns
    the median brightness out of transit minus the median brightness in
    transit, in parts per million, and the number of in-transit
    measurements. The depth is NaN if there are too few measurements
    in or out of transit to trust it.
    """
    offset = np.abs(fold_phase(df["time_days"], period, t0))
    brightness = df["brightness"].to_numpy()
    inside = brightness[offset <= IN_TRANSIT_FRACTION * duration_days]
    outside = brightness[offset > duration_days]
    if len(inside) < MIN_IN_TRANSIT or len(outside) < MIN_OUT_OF_TRANSIT:
        return np.nan, len(inside)
    return (np.median(outside) - np.median(inside)) * PPM, len(inside)


def measure_depths(sample: pd.DataFrame, clean_dir: Path) -> pd.DataFrame:
    """
    Measures the observed transit depth of every star in the sample.
    Returns a DataFrame with the columns kepid, observed_depth_ppm and
    n_in_transit, one row per star.
    """
    rows = []
    for _, koi in sample.iterrows():
        kepid = int(koi["kepid"])
        light_curve = pd.read_csv(clean_dir / f"{kepid}.csv")
        depth, n_inside = observed_depth_ppm(
            light_curve, koi["koi_period"], koi["koi_time0bk"],
            koi["koi_duration"] / 24)
        rows.append({"kepid": kepid, "observed_depth_ppm": depth,
                     "n_in_transit": n_inside})