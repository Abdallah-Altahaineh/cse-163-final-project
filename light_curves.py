"""
Kepler light curve download, cleaning, and EDA.

Takes the cleaned KOI table from data_pipeline.py (data/koi_clean.csv),
picks a balanced sample of stars, downloads their Kepler light curves
from MAST through Lightkurve, cleans them, summarizes them, and saves
two EDA figures. Cleaned light curves are written to
data/lightcurves_clean/<kepid>.csv for the sonification code.
Run with:  python light_curves.py   (after python data_pipeline.py)
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import data_pipeline as dp

RAW_DIR = os.path.join("data", "lightcurves", "raw")
CLEAN_DIR = os.path.join("data", "lightcurves", "clean")
SAMPLE_PATH = os.path.join("data", "sample_kois.csv")
FIG_DIR = "figures"

ID_COL = "kepid"
NAME_COL = "kepoi_name"
EPOCH_COL = "koi_time0bk"
DURATION_COL = "koi_duration"

SAVED_NAMES = {"time": "time_days", "flux": "brightness",
               "flux_err": "brightness_err"}

N_PER_CLASS = 25
SEED = 163
DETREND_WINDOW = 301
CLIP_SIGMA = 4.0
STAT_COLS = ["n_raw", "pct_missing", "flux_mean", "flux_std",
             "flux_min", "flux_max"]


def to_saved_format(df):
    """Rename internal columns to the names used in saved CSV files."""
    return df.rename(columns=SAVED_NAMES)


def from_saved_format(df):
    """
    Rename saved-file columns (time_days, brightness, brightness_err)
    back to the internal names. Columns that already use the internal
    names pass through unchanged, so older cached files still load.
    """
    reverse = {new: old for old, new in SAVED_NAMES.items()}
    return df.rename(columns=reverse)


def select_sample(koi, n_per_class=N_PER_CLASS, seed=SEED):
    """
    Return a balanced sample of KOIs: up to `n_per_class` rows per
    disposition, with at most one KOI per star so no light curve is
    downloaded twice. Sampling is random but reproducible via `seed`.
    """
    one_per_star = koi.drop_duplicates(subset=ID_COL)
    parts = []
    for label in dp.KEEP_LABELS:
        group = one_per_star[one_per_star[dp.LABEL_COL] == label]
        count = min(n_per_class, len(group))
        parts.append(group.sample(n=count, random_state=seed))
    return pd.concat(parts).reset_index(drop=True)


def download_light_curve(kepid, cache_dir=RAW_DIR):
    """
    Return a raw Kepler long-cadence light curve for one star as a
    DataFrame with columns time (BKJD days), flux, flux_err. Reads
    the cached CSV if present, otherwise downloads every quarter
    with Lightkurve, stitches them, and caches the result. Returns
    None if MAST has no light curve for the star.
    """
    path = os.path.join(cache_dir, f"{kepid}.csv")
    if os.path.exists(path):
        return from_saved_format(pd.read_csv(path))
    import lightkurve as lk
    results = lk.search_lightcurve(
        f"KIC {kepid}", mission="Kepler", author="Kepler",
        cadence="long")
    if len(results) == 0:
        return None
    lc = results.download_all(flux_column="pdcsap_flux").stitch()
    df = pd.DataFrame({
        "time": np.asarray(lc.time.value, dtype=float),
        "flux": np.ma.filled(lc.flux.value.astype(float), np.nan),
        "flux_err": np.ma.filled(lc.flux_err.value.astype(float),
                                 np.nan),
    })
    os.makedirs(cache_dir, exist_ok=True)
    to_saved_format(df).to_csv(path, index=False)
    return df


def clean_light_curve(df, window=DETREND_WINDOW, sigma=CLIP_SIGMA):
    """
    Return a cleaned copy of a raw light curve with columns time, flux
    (and flux_err, scaled the same way, when the raw data has it).

    Steps: drop rows with missing time or flux and sort by time;
    divide by the median flux; flatten slow stellar trends by dividing
    by a centered rolling median of `window` cadences (long enough to
    keep transit dips, which are much shorter); remove only upward
    outliers more than `sigma` robust standard deviations above the
    median, so real transit dips are never clipped.
    """
    out = df.dropna(subset=["time", "flux"])
    out = out.sort_values("time").reset_index(drop=True)
    flux = out["flux"] / out["flux"].median()
    trend = flux.rolling(window, center=True, min_periods=1).median()
    flat = flux / trend
    center = flat.median()
    spread = 1.4826 * (flat - center).abs().median()
    if spread > 0:
        keep = flat <= center + sigma * spread
    else:
        keep = pd.Series(True, index=flat.index)
    result = pd.DataFrame({"time": out["time"], "flux": flat})
    if "flux_err" in out.columns:
        result["flux_err"] = out["flux_err"] / out["flux"].median() / trend
    return result[keep].reset_index(drop=True)


def light_curve_stats(raw, clean):
    """
    Return a dict of size, missingness, and flux statistics for one
    star: raw row count, missing-flux count and percent, cleaned row
    count, and mean/std/min/max of the cleaned flux.
    """
    n_raw = len(raw)
    n_missing = int(raw["flux"].isna().sum())
    return {
        "n_raw": n_raw,
        "n_missing": n_missing,
        "pct_missing": 100 * n_missing / n_raw,
        "n_clean": len(clean),
        "flux_mean": clean["flux"].mean(),
        "flux_std": clean["flux"].std(),
        "flux_min": clean["flux"].min(),
        "flux_max": clean["flux"].max(),
    }


def fold_phase(time, period, t0):
    """
    Return each time's offset in days from the nearest transit
    center, in the range [-period/2, period/2).
    """
    time = np.asarray(time, dtype=float)
    return (time - t0 + 0.5 * period) % period - 0.5 * period


def plot_raw_vs_clean(raw, clean, kepid, out_path):
    """
    Save a two-panel figure of one star's brightness over time: the
    raw light curve on top and the cleaned light curve below.
    """
    fig, axes = plt.subplots(2, 1, figsize=(8, 5.5), sharex=True)
    axes[0].plot(raw["time"], raw["flux"], ".", ms=1.5,
                 label="Raw flux")
    axes[1].plot(clean["time"], clean["flux"], ".", ms=1.5,
                 color="tab:orange", label="Cleaned flux")
    axes[0].set_ylabel("Flux")
    axes[1].set_ylabel("Normalized flux")
    axes[1].set_xlabel("Time (BKJD, days)")
    axes[0].set_title(f"Raw vs. Cleaned Light Curve (KIC {kepid})")
    for ax in axes:
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_folded_transit(clean, period, t0, duration_hours, kepid,
                        out_path):
    """
    Save a phase-folded plot of the cleaned light curve centered on
    the transit, with binned medians drawn over the individual
    points so the transit dip is easy to see.
    """
    hours = fold_phase(clean["time"], period, t0) * 24
    limit = 3 * duration_hours
    near = np.abs(hours) <= limit
    x = hours[near]
    y = clean["flux"].to_numpy()[near]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(x, y, ".", ms=2, alpha=0.3, label="Cleaned flux")
    edges = np.linspace(-limit, limit, 41)
    idx = np.digitize(x, edges)
    centers, medians = [], []
    for i in range(1, len(edges)):
        in_bin = y[idx == i]
        if len(in_bin) > 0:
            centers.append((edges[i - 1] + edges[i]) / 2)
            medians.append(np.median(in_bin))
    ax.plot(centers, medians, "-", color="tab:red", lw=2,
            label="Binned median")
    ax.set_xlabel("Hours from transit center")
    ax.set_ylabel("Normalized flux")
    ax.set_title(f"Phase-Folded Transit (KIC {kepid})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    koi = pd.read_csv(os.path.join("data", "koi_clean.csv"))
    sample = select_sample(koi)
    sample.to_csv(SAMPLE_PATH, index=False)
    os.makedirs(CLEAN_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    rows = []
    best = None
    for _, koi_row in sample.iterrows():
        kepid = int(koi_row[ID_COL])
        try:
            raw = download_light_curve(kepid)
        except Exception as err:
            print(f"Skipping KIC {kepid}: {err}")
            continue
        if raw is None or raw.empty:
            print(f"No light curve for KIC {kepid}")
            continue
        clean = clean_light_curve(raw)
        to_saved_format(clean).to_csv(
            os.path.join(CLEAN_DIR, f"{kepid}.csv"), index=False)
        stats = light_curve_stats(raw, clean)
        stats[ID_COL] = kepid
        stats[dp.LABEL_COL] = koi_row[dp.LABEL_COL]
        rows.append(stats)
        is_confirmed = koi_row[dp.LABEL_COL] == "CONFIRMED"
        deeper = best is None or koi_row[dp.DEPTH_COL] > best[0]
        if is_confirmed and deeper:
            best = (koi_row[dp.DEPTH_COL], koi_row, raw, clean)

    table = pd.DataFrame(rows)
    table.to_csv(os.path.join("data", "light_curve_stats.csv"),
                 index=False)
    print("Stars with light curves:", len(table), "of", len(sample))
    print("Light curve rows per star, % missing flux, cleaned flux:")
    print(dp.summarize_numeric(table, STAT_COLS))

    if best is not None:
        _, koi_row, raw, clean = best
        kepid = int(koi_row[ID_COL])
        plot_raw_vs_clean(
            raw, clean, kepid,
            os.path.join(FIG_DIR, "raw_vs_clean_light_curve.png"))
        plot_folded_transit(
            clean, koi_row[dp.PERIOD_COL], koi_row[EPOCH_COL],
            koi_row[DURATION_COL], kepid,
            os.path.join(FIG_DIR, "folded_transit.png"))
        print("Saved figures for KIC", kepid)


if __name__ == "__main__":
    main()
