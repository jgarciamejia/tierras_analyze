import numpy as np 
import matplotlib.pyplot as plt 
plt.ioff()
import pandas as pd 
import argparse 
import os 
from scipy.stats import sigmaclip, loguniform
from astropy.timeseries import LombScargle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from ap_phot import set_tierras_permissions
import pickle 

def main(raw_args=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("-field", required=True, help="Name of observed target field exactly as shown in raw FITS files.")
    ap.add_argument("-p_min", required=False, default=1/24, help="Minimum period in days from which to draw injection signals", type=float)
    ap.add_argument("-p_max_ratio", required=False, default=1.0, help="Maximum fraction of injected period to length of the time series", type=float)
    ap.add_argument("-a_min", required=False, default=0.0001, help='Minimum amplitude of injected signals', type=float)
    ap.add_argument('-a_max', required=False, default=0.1, help='Maximum amplitude of injected signals', type=float)
    ap.add_argument('-detection_threshold', required=False, default=0.01, help='Maximum relative difference between injected and recovered period to consider simulation loop a success', type=float)
    ap.add_argument('-n_simulations', required=False, default=50000, help='Number of injections to perform', type=int)
    args = ap.parse_args(raw_args)
    field = args.field
    p_min = args.p_min
    p_max_ratio = args.p_max_ratio
    a_min = args.a_min
    a_max = args.a_max
    n_sims = args.n_simulations
    detection_threshold = args.detection_threshold
    
    lc_path = f'/data/tierras/fields/{field}/sources/lightcurves/{field}_global_lc.csv'
    if not os.path.exists(lc_path):
        print(f'{lc_path} does not exist! Returning.')
        return 
    else:
        lc_df = pd.read_csv(lc_path, comment='#')

    times = np.array(lc_df['BJD TDB'])
    flux = np.array(lc_df['Flux'])
    flux_err = np.array(lc_df['Flux Error'])

    # mask out flagged exposures 
    flux_flag = np.array(lc_df['Low Flux Flag']).astype(bool)
    wcs_flag = np.array(lc_df['WCS Flag']).astype(bool)
    pos_flag = np.array(lc_df['Position Flag']).astype(bool)
    fwhm_flag = np.array(lc_df['FWHM Flag']).astype(bool)
    mask = ~(flux_flag | wcs_flag | pos_flag | fwhm_flag)

    times = times[mask]
    flux = flux[mask]
    flux_err = flux_err[mask]

    x_offset = times[0]
    times -= x_offset

    # toss any nans
    nan_inds =  ~(np.isnan(times) | np.isnan(flux) | np.isnan(flux_err))
    times = times[nan_inds]
    flux = flux[nan_inds]
    flux_err = flux_err[nan_inds]

    # sigma clip 
    # TODO: set threshold dynamically?
    v, l, h = sigmaclip(flux, 4, 4)
    sc_inds = np.where((flux > l) & (flux < h))[0]
    times = times[sc_inds]
    flux = flux[sc_inds]
    flux_err = flux_err[sc_inds] #NOTE: should this be replaced with standard deviation?

    p_max = times[-1] * p_max_ratio # calculate the maximum period of injected signals using extent of offset-adjusted times

    # model_times = np.linspace(times[0], times[-1], 10000)
    # do injection and recovery
    min_freq = 1/p_max
    max_freq = 1/p_min

    # do an initial run of the periodogram to get the frequency grid so you don't have to compute it each simulation step
    freq, pow = LombScargle(times, flux, flux_err).autopower(minimum_frequency=min_freq, maximum_frequency=max_freq)

    injected_per = []
    injected_amp = []
    success = []
    injected_per = loguniform.rvs(p_min, p_max, size=n_sims)
    injected_amp = loguniform.rvs(a_min, a_max, size=n_sims)
    injected_phase = np.random.uniform(0, 2*np.pi, size=n_sims)
    
    for i in range(n_sims):
        amp = injected_amp[i]
        per = injected_per[i]
        phase = injected_phase[i]
        
        signal = 1+amp*np.sin(2*np.pi/per*times+phase)
        injected_flux = flux * signal

        power = LombScargle(times, injected_flux, flux_err).power(freq)
        per_max_power = 1/freq[np.argmax(power)]
        if abs(per-per_max_power)/per < detection_threshold: 
            success.append(True)
            # fig, ax = plt.subplots(2,1,figsize=(9,9))
            # ax[0].plot(times, injected_flux, 'k.')
            # ax[0].plot(times, signal, 'r')
            # ax[1].plot(1/freq, power)
            # ax[1].set_xscale('log')
            # ax[1].plot(per_max_power, power[np.argmax(power)], color='tab:orange', marker='o')
            # ax[1].axvline(per, ls='--', color='m')
            # breakpoint()
        else:
            success.append(False)

        if i % 100 == 0 and i != 0:
            print(f'{i} of {n_sims}')
    
    injected_per = np.array(injected_per)
    injected_amp = np.array(injected_amp)
    success = np.array(success)
    success_inds = np.where(success)[0]

    amp_grid_edges = np.logspace(np.log10(a_min), np.log10(a_max), 11)
    per_grid_edges = np.logspace(np.log10(p_min), np.log10(p_max), 11)

    injection_counts, _, _ = np.histogram2d(injected_per, injected_amp, bins=[per_grid_edges, amp_grid_edges])
    injection_counts = injection_counts.T
    recovery_counts, _, _ = np.histogram2d(injected_per[success_inds], injected_amp[success_inds], bins=[per_grid_edges, amp_grid_edges])
    recovery_counts = recovery_counts.T

    # plot injected/recovered pers/amplitudes separately
    fig, ax = plt.subplots(2,1,figsize=(9,9))
    ax[0].hist(injected_per, bins=per_grid_edges)
    ax[0].set_xscale('log')
    ax[0].hist(injected_per[success_inds], bins=per_grid_edges)
    ax[0].set_xlabel('Per')

    ax[1].hist(injected_amp, bins=amp_grid_edges)
    ax[1].set_xscale('log')
    ax[1].hist(injected_amp[success_inds], bins=amp_grid_edges)
    ax[1].set_xlabel('Amp')

    fig, ax = plt.subplots(1,1,figsize=(9,9))
    im = ax.imshow(recovery_counts/injection_counts, interpolation='none', origin='lower', extent=[np.log10(per_grid_edges[0]), np.log10(per_grid_edges[-1]), np.log10(amp_grid_edges[0]), np.log10(amp_grid_edges[-1])])
    ax.axis('equal')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.1)
    cb = fig.colorbar(im, cax=cax, orientation='vertical')
    cb.set_label('Detection Fraction', rotation=270, fontsize=12)
    cb.ax.get_yaxis().labelpad = 15
    ax.tick_params(labelsize=12)
    ax.set_xlabel('log$_{10}$(P) (days)', fontsize=14)
    ax.set_ylabel('log$_{10}$(Amplitude)', fontsize=14)
    plt.tight_layout()

    # save the plot and injection/recovery data 
    output_dir = f'/data/tierras/fields/{field}/period_injection_recovery/'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        set_tierras_permissions(output_dir)
    
    plt.savefig(output_dir+f'{field}_period_injection_recovery.png', dpi=300)
    set_tierras_permissions(output_dir+f'{field}_period_injection_recovery.png')

    output_df = pd.DataFrame(np.array([injected_amp, injected_per, injected_phase, success.astype(bool)]).T, columns=['Injected Amplitude', 'Injected Period', 'Injected Phase', 'Success'])
    output_df.to_csv(output_dir+f'{field}_period_injection_recovery_data.csv', index=0)
    set_tierras_permissions(output_dir+f'{field}_period_injection_recovery_data.csv')

    plt.close('all')

    return 

if __name__ == '__main__':
    main()