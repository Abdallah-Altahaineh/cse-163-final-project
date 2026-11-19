"""
Hari - CSE 163 Final Project
EDA of the audio feature table: size, missing data, summaries,
and how features differ between confirmed planets and false positives.
"""

import os

import matplotlib.pyplot as plt
import pandas as pd

FEATURES = ['pitch_drop_semitones', 'noise_hz', 'dip_to_noise']
FIG_DIR = 'figures'


def plot_depth(df: pd.DataFrame, path: str) -> None:
    """
    Saves a box plot comparing the pitch drop of confirmed planets and
    false positives.
    """
    confirmed = df[df['koi_disposition'] == 'CONFIRMED']
    false_pos = df[df['koi_disposition'] == 'FALSE POSITIVE']
    plt.figure()
    plt.boxplot([confirmed['pitch_drop_semitones'],
                 false_pos['pitch_drop_semitones']])
    plt.xticks([1, 2], ['Confirmed Planet', 'False Positive'])
    plt.xlabel('NASA Label')
    plt.ylabel('Pitch drop (semitones)')
    plt.title("Pitch Drop in NASA's Confirmed and False Positives")
    plt.savefig(path, dpi=200)
    plt.close()


def plot_noise_vs_depth(df: pd.DataFrame, path: str) -> None:
    """
    Saves a scatter plot of noise against pitch drop, one color per class.
    """
    confirmed = df[df['koi_disposition'] == 'CONFIRMED']
    false_pos = df[df['koi_disposition'] == 'FALSE POSITIVE']
    plt.figure()
    plt.scatter(confirmed['noise_hz'], confirmed['pitch_drop_semitones'],
                label='Confirmed planets')
    plt.scatter(false_pos['noise_hz'], false_pos['pitch_drop_semitones'],
                label='False positives')
    plt.xlabel('Noise (Hz)')
    plt.ylabel('Pitch drop (semitones)')
    plt.title("Pitch Drop vs Noise for NASA's Confirmed and False Positives")
    plt.legend()
    plt.savefig(path, dpi=200)
    plt.close()


def main() -> None:
    """
    Saves both figures to figures/ and prints a summary of the audio
    features table.
    """
    df = pd.read_csv('audio_features.csv')
    os.makedirs(FIG_DIR, exist_ok=True)
    plot_depth(df, os.path.join(FIG_DIR, 'depth_by_class.png'))
    plot_noise_vs_depth(df, os.path.join(FIG_DIR, 'noise_vs_depth.png'))

    print(f"1. Size (rows, columns): {df.shape}")
    print(f"2. Missing values per column:\n{df.isna().sum()}")
    print(f"3. Seven-number summary:\n{df[FEATURES].describe()}")
    print('4. Label counts:')
    print(df['koi_disposition'].value_counts())
    print('5. Median of each feature, by class:')
    print(df.groupby('koi_disposition')[FEATURES].median())
    print('6. Pitch drop summary by class:')
    print(df.groupby('koi_disposition')['pitch_drop_semitones'].describe())
    print(f"7. Duplicate stars: {df['kepid'].duplicated().sum()}")


if __name__ == '__main__':
    main()
