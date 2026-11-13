"""
Hari - CSE 163 Final Project
Turns a star's brightness into pitch drops and measures each dip
(how many, how deep, how long, what shape) so we can compare
confirmed planets and false positives by sound.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def fake_light_curve():
    time = np.arange(0, 90, 0.0204)
    flux = np.ones(len(time))
    df = pd.DataFrame({'time': time, 'flux': flux})
    for start in range(10, 90, 10):
        df.loc[(df['time'] > start) & (df['time'] < start + 0.25), 'flux'] = 0.995

    np.random.seed(0)
    df['flux'] = df['flux'] + np.random.normal(0, 0.0001, len(df))
    return df


def flux_to_drop(df):
    df['drop'] = (1 - df['flux'])* 1000 
    df['drop'] = (df['drop'].clip(upper=24))
    return df


def mark_dips(df):
    # Converting median absolute deviation (MAD) to standard deviation for ideal bell curve noise 1/(.6745)
    MAD_TO_STD = 1.4826
    dip_threshold = 5
    middle = df['drop'].median()
    noise = (df['drop'] - middle).abs().median() * MAD_TO_STD
    df['in_dip'] = df['drop'] > middle + dip_threshold * noise

    return df



def count_dips(df):
    count = 0
    previous = False
    for value in df['in_dip']:
        if value == True and previous == False:
            count +=1
        previous = value
    return count

def label_dips(df):
    count = 0
    previous = False
    labels = []
    for value in df['in_dip']:
        if value > previous:
            count +=1
        if value == True:
            labels.append(count)
        else:
            labels.append(0)

        previous = value
    df['dip_id'] = labels
    return df

def measure_dips(df):
    hours_per_row = 0.49
    in_dips = df[df['in_dip']]
    groups = in_dips.groupby('dip_id')
    depth = groups['drop'].max()
    duration = groups.size() * hours_per_row
    return pd.DataFrame({'depth': depth, "duration_in_hours": duration})

def build_features_table(folder):
    rows = []
    for name in os.listdir(folder):
        path = os.path.join(folder, name)
        df = load_star(path)
        df = flux_to_drop(df)
        df = mark_dips(df)
        df = label_dips(df)
        summary = summarize_dips(df, measure_dips(df))
        summary['kepid'] = int(name[:-4])
        rows.append(summary)
    return pd.DataFrame(rows)

    

def summarize_dips(df, dips):
    noise = df[df['dip_id'] == 0]['drop'].std()
    return {
        'num_dips': len(dips),
        'depth': float(dips['depth'].mean()),
        'duration_hr': float(dips['duration_in_hours'].mean()),
        'noise': float(noise),
        'ratio': float(dips['depth'].mean() / noise),
    }


def load_star(path):
    df = pd.read_csv(path)
    df = df.rename(columns={'time_days': 'time', 'brightness': 'flux'})
    return df


def main():
    features = build_features_table('data/lightcurves/clean')
    stars = pd.read_csv('data/stars.csv')
    labels = stars[['kepid', 'koi_disposition']]
    labeled = features.merge(labels, on='kepid')
    print(labeled)
    print(len(labeled))
    print(labeled['koi_disposition'].value_counts())

  
    


if __name__ == '__main__':
    main()
    
