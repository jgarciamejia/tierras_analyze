from astropy.timeseries import LombScargle
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
plt.ion()
from scipy.stats import sigmaclip

def periodogram(x, y, y_err):

    use_inds = ~np.isnan(y)
    x = x[use_inds]
    y = y[use_inds]
    y_err = y_err[use_inds]

    v, l, h = sigmaclip(y)
    use_inds = np.where((y>l)&(y<h))[0]
    x = x[use_inds]
    y = y[use_inds]
    y_err = y_err[use_inds]

    v, l, h = sigmaclip(y_err)
    use_inds = np.where(y_err<h)[0]
    x = x[use_inds]
    y = y[use_inds]
    y_err = y_err[use_inds] 
    
    x -= x[0]
    pers = np.arange(0.26, 0.29, 0.5/86400)
    freqs = 1/pers
    power = LombScargle(x, y, y_err).power(freqs)

    return x, y, y_err, pers, freqs, power

def periodogram_plot(x, y, y_err, per, power, phase=False, color_by_time=False):
    if phase:
        fig, ax = plt.subplots(3,1,figsize=(10,8))
    else:
        fig, ax = plt.subplots(2,1,figsize=(10,6))

    for a in ax:
        a.tick_params(labelsize=12)

    if color_by_time: 
        cmap = plt.get_cmap('viridis')
        inds = np.arange(len(x))
        color_inds = np.array([int(i) for i in inds*255/len(x)])
        color = cmap(color_inds)
    else:
        color='#b0b0b0'

    ax[0].scatter(x, y, c=color, s=2)
    ax[0].errorbar(x, y, y_err, linestyle="None",marker='',color=color,zorder=0)
    ax[0].set_xlabel('BJD TDB', fontsize=14)
    ax[0].set_ylabel('Normalized Flux', fontsize=14)
    
    ax[1].plot(per, power, marker='.')
    ax[1].set_xscale('log')
    best_per = per[np.argmax(power)]
    ax[1].plot(best_per, np.max(power), marker='o')
    ax[1].set_xlabel('Period (d)', fontsize=14)
    ax[1].set_ylabel('Power', fontsize=14)

    best_per = 0.2759
    if phase:
        phased_x = (x % best_per) / best_per 
        sort = np.argsort(phased_x)
        phased_x = phased_x[sort]
        phased_y = y[sort]
        phased_y_err = y_err[sort]

        ax[2].scatter(phased_x, phased_y, c=color[sort], s=5)
        ax[2].errorbar(phased_x, phased_y, phased_y_err, linestyle="None",marker='',color=color[sort],zorder=0)
        ax[2].set_xlabel('Phase', fontsize=14)
        ax[2].set_ylabel('Normalized Flux', fontsize=14)
        ax[2].grid(alpha=0.7)
    plt.tight_layout()
    breakpoint()
    return 

if __name__ == '__main__':
    df = pd.read_csv('/data/tierras/targets/Gaia DR3 1450064354510643456/Gaia DR3 1450064354510643456_global_lc.csv', comment='#')
    x = np.array(df['BJD TDB'])
    y = np.array(df['Flux'])
    y_err = np.array(df['Flux Error'])

    x, y, y_err, per, freq, power = periodogram(x, y, y_err)
    periodogram_plot(x, y, y_err, per, power, phase=True, color_by_time=True)

   

    breakpoint()