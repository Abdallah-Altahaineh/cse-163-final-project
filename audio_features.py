"""
Hari - CSE 163 Final Project
Turns a star's brightness into pitch drops and measures each dip
(how many, how deep, how long, what shape) so we can compare
confirmed planets and false positives by sound.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def fake_light_curve():
    time = np.arange(0, 90, 0.0204)
    flux = np.ones(len(time))
    df = pd.DataFrame({'time': time, 'flux': flux})
    for start in range(10, 90, 10):
        df.loc[(df['time'] > start) & (df['time'] < start + 0.25), 'flux'] = 0.995
    return df


def flux_to_drop(df):
    df['drop'] = (1 - df['flux'])* 1000 
    df['drop'] = df['drop'].clip(upper=24)
    return df


def mark_dips(df):
    df['in_dip'] = df['drop'] > 1
    return df



def count_dips(df):
    count = 0
    previous = False
    for value in df['in_dip']:
        if value == True and previous == False:
            count +=1
        previous = value
    return count


def main():
    df = fake_light_curve()
    df = flux_to_drop(df)     
    df = mark_dips(df)           
    print(f"Number of dips: {count_dips(df)}")
    print(f"Rows in dips: {df['in_dip'].sum()}")
    plt.plot(df['time'], df['drop'])     
    plt.xlabel('Time (days)')
    plt.ylabel('Pitch drop(semitones)')
    plt.show()
    


if __name__ == '__main__':
    main()
    
