import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def fake_light_curve():
    time = np.arange(0, 90, 0.0204)
    flux = np.ones(len(time))
    df = pd.DataFrame({'time': time, 'flux': flux})
    df.loc[(df['time'] > 10) & (df['time'] < 10.25), 'flux'] = 0.995
    return df


def flux_to_drop(df):
    df['drop'] = (1 - df['flux'])* 1000 
    df['drop'] = df['drop'].clip(upper=24)
    return df


def main():
    df = fake_light_curve()
    df = flux_to_drop(df)                
    plt.plot(df['time'], df['drop'])     
    plt.xlabel('Time (days)')
    plt.ylabel('Pitch drop(semitones)')
    plt.show()


if __name__ == '__main__':
    main()
    
