"""
Hari - CSE 163 Final Project
EDA of the audio feature table: size, missing data, summaries,
and how features differ between confirmed planets and false positives.
"""

import pandas as pd
import os
import matplotlib.pyplot as plt
from sonify import brightness_to_hz, fold_light_curve, load_periods, pitch_stats


def plot_depth(df: pd.DataFrame) -> None:
    """
    Creates a box plot that compares the pitch drop of confirmed and false positive planets
    """
    confirmed = df[df['koi_disposition'] == 'CONFIRMED']['pitch_drop_semitones']
    false_positives = df[df['koi_disposition'] == 'FALSE POSITIVE']['pitch_drop_semitones']
    plt.figure()
    plt.boxplot([confirmed, false_positives])
    plt.xticks([1, 2], ['Confirmed Planet', 'False Positive'])
    plt.xlabel('NASA Label')
    plt.ylabel('Pitch drop (semitones)')
    plt.title("Pitch Drop in Nasa's Confirmed and False Positives")
    plt.savefig('depth_by_class.png')

def plot_noise_vs_depth(df: pd.DataFrame) -> None:
    """
    Creates a scatter plot with points of data from confirmed and false positives and their noise and pitch drop
    """
    confirmed = df[df['koi_disposition'] == 'CONFIRMED']
    false_pos = df[df['koi_disposition'] == 'FALSE POSITIVE']
    plt.figure()
    plt.scatter(confirmed['noise_hz'], confirmed['pitch_drop_semitones'], label = "Confirmed planets noise and depth")
    plt.scatter(false_pos['noise_hz'], false_pos['pitch_drop_semitones'], label = "False positive planets noise and depth")
    plt.xlabel("Noise (Hz)")
    plt.ylabel("Pitch drop (semitones)")
    plt.title("Pitch drop vs Noise for Nasa's Confirmed and False Positives")
    plt.legend()
    plt.savefig("noise_vs_depth.png")




FEATURES = ['pitch_drop_semitones', 'noise_hz', 'dip_to_noise']


def main() -> None:
    """
    Runs all the graph creation functions and saves them as PNG images, prints a summary of the audio features table.
    """
    df = pd.read_csv('audio_features.csv')

    plot_depth(df)
    plot_noise_vs_depth(df)
    print(f"1. Size (rows, columns): {df.shape}")


    print(f"2. Missing values per column: {df.isna().sum()}")


    print(f"3. Seven-number summary: {df[FEATURES].describe()}")

    print("Result Validity")

    print('4. Label counts:')
    print(df['koi_disposition'].value_counts())

    print(f"5. Median of each feature, by class:")
    print(df.groupby('koi_disposition')[FEATURES].median())

    print(f"6. Duplicate stars: {df['kepid'].duplicated().sum()}")



if __name__ == '__main__':
    main()
