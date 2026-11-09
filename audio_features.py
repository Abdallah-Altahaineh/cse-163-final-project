import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def fake_light_curve():
    time = np.arange(0, 90, 0.0204)
    flux = np.ones(len(time))
    df = pd.DataFrame({'time': time, 'flux': flux})
    df.loc[(df['time'] > 10) & (df['time'] < 10.25), 'flux'] = 0.995
    return df



def main():
    df = fake_light_curve()
    plt.plot(df['time'], df['flux'])
    plt.xlabel('Time (days)')
    plt.ylabel('Brightness')
    plt.show()


if __name__ == '__main__':
    main()

