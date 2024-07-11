import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
plt.ion() 
from scipy.stats import sigmaclip 
from median_filter import median_filter_uneven
from astrobase import periodbase 
import matplotlib
matplotlib.use('qt5agg')

def theta_aov(times, flux, flux_err, period, binsize=0.05, min_bin=9):
    # phase on the period 
    phased_times = times % period / period
    sort = np.argsort(phased_times)
    phased_times = phased_times[sort]
    phased_flux = flux[sort]
    phased_flux_err = flux_err[sort]

    bins = np.arange(0, 1, binsize)
    n_dets = len(phased_times)
    bin_inds = np.digitize(phased_times, bins)

    bin_s1_numerators = []
    bin_s2_numerators = []
    bin_ndets = []
    good_bins = 0 
    all_xbar = np.nanmedian(phased_flux)

    for x in np.unique(bin_inds):
        inds = bin_inds == x
        bin_flux = phased_flux[inds]
        if len(bin_flux) > min_bin: 
            bin_ndet = len(bin_flux)
            bin_xbar = np.nanmedian(bin_flux)
            bin_s1_numerator = bin_ndet * (bin_xbar - all_xbar)**2 # see eqn 1 from Schwarzenberg-Czerny 1989
            bin_s2_numerator = np.nansum((bin_flux-all_xbar)**2)
            bin_s1_numerators.append(bin_s1_numerator)
            bin_s2_numerators.append(bin_s2_numerator)
            bin_ndets.append(bin_ndet)
            good_bins += 1

    bin_s1_numerators = np.array(bin_s1_numerators)
    bin_s2_numerators = np.array(bin_s2_numerators)
    s1 = np.nansum(bin_s1_numerators)/(good_bins-1)
    s2 = np.nansum(bin_s2_numerators)/(n_dets)

    return s1/s2

def aov(times, flux, flux_err):

    times -= times[0] 
    periods = np.arange(1.9, 2.1, 1/86400)
    thetas = np.zeros(len(periods))
    plt.figure()
    for i in range(len(periods)):
        period = periods[i]
        thetas[i] = theta_aov(times, flux, flux_err, period)
        if i % 100 == 0:
            print(f'{period:.4f}, {thetas[i]:.3f} ({i+1} of {len(periods)})')
        #plt.plot(periods[0:i+1], thetas[0:i+1])
        #plt.pause(0.1)
        # plt.clf()

    return periods, thetas

if __name__ == '__main__':

    target ='Gaia DR3 4146925275198041216'

    lc_path = f'/data/tierras/fields/TIC362144730/sources/lightcurves/{target}_global_lc.csv'
    df = pd.read_csv(lc_path, comment='#')
    times = np.array(df['BJD TDB'])
    flux = np.array(df['Flux'])
    flux_err = np.array(df['Flux Error'])

    nan_inds = ~np.isnan(flux)
    times = times[nan_inds]
    flux = flux[nan_inds]
    flux_err = flux_err[nan_inds]

    # clear major outliers
    v, l, h = sigmaclip(flux)
    sc_inds = np.where((flux > l) & (flux < h))[0]
    times = times[sc_inds]
    flux = flux[sc_inds]
    flux_err = flux_err[sc_inds]

    # do a 5-minute median filter and flag more outliers 
    x_filter, y_filter = median_filter_uneven(times, flux, 5/(60*24))
    y_filtered = flux / y_filter

    v, l, h = sigmaclip(y_filtered, 3, 3)
    sc_inds = np.where((y_filtered > l) & (y_filtered < h))[0]
    times = times[sc_inds]
    flux = flux[sc_inds]
    flux_err = flux_err[sc_inds]


    # per, theta = aov(times, flux, flux_err)
    # plt.plot(per, theta)

    # best_per = per[np.argmax(theta)]

    # phased_times = times % best_per / best_per
    # sort = np.argsort(phased_times)
    # phased_times = phased_times[sort]
    # phased_flux = flux[sort]
    # phased_flux_err = flux_err[sort]

    fig, ax = plt.subplots(4,1,figsize=(10,10))
    ax[0].plot(times, flux, marker='.', ls='')
    # ax[1].plot(per, theta)

    # ax[2].plot(phased_times, phased_flux, marker='.', ls='')
    
    # ax[2].set_title(f'Folded on {best_per:.5} d')
    res_dict = periodbase.saov.aov_periodfind(times, flux, flux_err, magsarefluxes=True, startp=0.3, endp=1.1, stepsize=1/86400, autofreq=False)
    ax[1].plot(res_dict['periods'], res_dict['lspvals'], marker='.')

    best_per = res_dict['bestperiod']
    # best_per = 1.0125
    phased_times = times % best_per / best_per
    sort = np.argsort(phased_times)
    phased_times = phased_times[sort]
    phased_flux = flux[sort]
    phased_flux_err = flux_err[sort]
    ax[3].plot(phased_times, phased_flux, marker='.', ls='')
    ax[3].set_title(f'Folded on {best_per} d')
    plt.tight_layout()

    breakpoint()