"""
Hari - CSE 163 Final Project
Turns a star's brightness into pitch drops and measures each dip
(how many, how deep, how long, what shape) so we can compare
confirmed planets and false positives by sound.
"""

import os

import pandas as pd
from sonify import (brightness_to_hz, fold_light_curve, load_periods,
                    pitch_stats)


def star_features(df: pd.DataFrame, period_days: float) -> dict:
    """Folds one star's light curve and returns its pitch statistics."""
    folded = fold_light_curve(df, period_days)
    folded['pitch_hz'] = brightness_to_hz(folded['brightness'])
    return pitch_stats(folded)


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
