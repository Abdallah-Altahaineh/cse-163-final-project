"""
Hari - CSE 163 Final Project
Turns a star's brightness into pitch drops and measures each dip
(how many, how deep, how long, what shape) so we can compare
confirmed planets and false positives by sound.
"""
import os
import pandas as pd
from sonify import (N_BINS, brightness_to_hz, fold_light_curve, load_periods,
                    pitch_stats)
min_dip_to_noise = 5
min_transit_bins = 15


def dip_shape(folded: pd.DataFrame, period_days: float) -> dict:
    """
    Measures the folded dip's duration (hours the brightness stays below
    half the dip's depth) and flatness (average drop inside the dip
    divided by the deepest drop: about 1 for flat-bottomed, lower for V).
    """
    outside = folded[folded['phase'].abs() > 0.1]
    drop = outside['brightness'].median() - folded['brightness']
    depth = drop.max()
    in_dip = (drop >= depth / 2) & (folded['phase'].abs() <= 0.1)
    duration_hr = in_dip.sum() / len(folded) * period_days * 24
    flatness = drop[in_dip].mean() / depth
    return {'duration_hr': duration_hr, 'flatness': flatness}


def star_features(df: pd.DataFrame, period_days: float) -> dict:
    """
    Folds one star's light curve and returns its pitch statistics.
    Duration and flatness are set to NaN when dip is below a minimum
    threshold and flatness is also set to NaN when below min threshold
    of slices to measure the shape
    """
    folded = fold_light_curve(df, period_days)
    folded['pitch_hz'] = brightness_to_hz(folded['brightness'])
    stats = pitch_stats(folded)
    shape = dip_shape(folded, period_days)
    transit_bins = shape['duration_hr'] / (period_days * 24) * N_BINS
    if stats['dip_to_noise'] < min_dip_to_noise:
        shape = {'duration_hr': float('nan'), 'flatness': float('nan')}
    elif transit_bins < min_transit_bins:
        shape['flatness'] = float('nan')
    stats.update(shape)
    return stats


def build_features_table(folder: str, periods: dict) -> pd.DataFrame:
    """Folds every star in the folder and returns one row of features each."""
    rows = []
    for name in os.listdir(folder):
        if not name.endswith('.csv'):
            continue
        kepid = int(name[:-4])
        df = pd.read_csv(os.path.join(folder, name))
        summary = star_features(df, periods[kepid])
        summary['kepid'] = kepid
        rows.append(summary)
    return pd.DataFrame(rows)


def main() -> None:
    """Builds the folded audio feature table with NASA labels."""
    periods = load_periods('data/stars.csv')
    features = build_features_table('data/lightcurves/clean', periods)
    labels = pd.read_csv('data/light_curve_stats.csv')
    labeled = features.merge(labels[['kepid', 'koi_disposition']], on='kepid')
    print(labeled)
    labeled.to_csv('audio_features.csv', index=False)


if __name__ == '__main__':
    main()
