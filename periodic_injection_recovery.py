import numpy as np 
import matplotlib.pyplot as plt 
plt.ioff()
import pandas as pd 
import argparse 
import os 
from scipy.stats import sigmaclip, loguniform
from astropy.timeseries import LombScargle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import FormatStrFormatter

def plot_injection_recovery(times, injected_flux, signal, freq, power, per_max_power, per, window_fn_power, alias_lowers, alias_uppers):
    #TODO: simplify inputs 

    fig = plt.figure(figsize=(9,9))
    ax1 = plt.subplot(311)
    ax2 = plt.subplot(312)
    ax3 = plt.subplot(313, sharex=ax2)

    ax1.plot(times, injected_flux, 'k.')
    ax1.plot(times, signal, 'r')
    ax2.plot(1/freq, power)
    ax2.set_xscale('log')
    ax2.plot(per_max_power, power[np.argmax(power)], color='tab:orange', marker='o')
    ax2.axvline(per, ls='--', color='m')
    ax2.set_ylabel('LS Periodogram Power', fontsize=14)

    ax3.plot(1/freq, window_fn_power)
    ax3.set_xscale('log')
    ax3.set_ylabel('Window Fn. Power', fontsize=14)
    ax3.set_xlabel('Period (d)', fontsize=14)
    ax2.set_ylim(ax3.get_ylim())

    alias_colors = ['tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan']

    for i in range(len(alias_lowers)):
        ax2.fill_between([alias_lowers[i], alias_uppers[i]], 0, 1, color=alias_colors[i], alpha=0.2)
    breakpoint()
    return 

def main(raw_args=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("-field", required=True, help="Name of observed target field exactly as shown in raw FITS files.")
    ap.add_argument("-ffname", required=False, default='flat0000', help='Name of flat directory for light curves.')
    ap.add_argument("-p_min", required=False, default=1/24, help="Minimum period in days from which to draw injection signals", type=float)
    ap.add_argument("-p_max_ratio", required=False, default=1.0, help="Maximum fraction of injected period to length of the time series", type=float)
    ap.add_argument("-a_min", required=False, default=0.001, help='Minimum amplitude of injected signals', type=float)
    ap.add_argument('-a_max', required=False, default=0.1, help='Maximum amplitude of injected signals', type=float)
    ap.add_argument('-detection_threshold', required=False, default=0.01, help='Maximum relative difference between injected and recovered period to consider simulation loop a success', type=float)
    ap.add_argument('-n_simulations', required=False, default=50000, help='Number of injections to perform', type=int)
    args = ap.parse_args(raw_args)
    field = args.field
    ffname = args.ffname
    p_min = args.p_min
    p_max_ratio = args.p_max_ratio
    a_min = args.a_min
    a_max = args.a_max
    n_sims = args.n_simulations
    detection_threshold = args.detection_threshold

    window_fn_threshold_factor = 0.5 # peak power must be greater than this factor times the window function power at the same frequecy to be accepted 
    aliases_to_check = [1/3, 1/2, 1, 2, 3,]

    lc_path = f'/data/tierras/fields/{field}/sources/lightcurves/{ffname}/{field}_global_lc.csv'
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
    
    # set up a coarse initial grid of period and amplitude cells
    amp_grid_edges = np.logspace(np.log10(a_min), np.log10(a_max), 21)
    per_grid_edges = np.logspace(np.log10(p_min), np.log10(p_max), 31)

    per_grid, amp_grid = np.meshgrid(per_grid_edges, amp_grid_edges)
    
    n_cells = (len(amp_grid_edges) - 1) * (len(per_grid_edges) - 1)
    if n_sims / n_cells < 100: 
        print(f'WARNING: n_sims/n_cells = {n_sims/n_cells:.2f}')

    # do an initial run of the periodogram to get the frequency grid so you don't have to compute it each simulation step
    freq, pow = LombScargle(times, flux, flux_err).autopower(minimum_frequency=min_freq, maximum_frequency=max_freq)

    # calculate the window function power of the data over the frequency grid 
    window_fn_power = LombScargle(times, np.ones_like(times), fit_mean=False, center_data=False).power(freq)

    # METHOD 1
    injected_per = []
    injected_amp = []
    success = []
    injected_per = loguniform.rvs(p_min, p_max, size=n_sims)
    injected_amp = loguniform.rvs(a_min, a_max, size=n_sims)
    injected_phase = np.random.uniform(0, 2*np.pi, size=n_sims)

    # injection_counts, _, _ = np.histogram2d(injected_per, injected_amp, bins=[per_grid_edges, amp_grid_edges])
    # injection_counts = injection_counts.T
    
    injected_amp = []
    injected_per = []
    injected_phase = []

    injection_counts = np.zeros_like(per_grid, dtype='int')
    recovery_counts = np.zeros_like(per_grid, dtype='int')
    success_rate = np.zeros_like(per_grid)
    n_amp_cells = len(amp_grid_edges) - 1
    n_per_cells = len(per_grid_edges) - 1
    n_injected = 0
    while n_injected < n_sims:
        # draw a random cell 
        i = np.random.choice(n_per_cells)
        j = np.random.choice(n_amp_cells)

        per_lower = per_grid_edges[i]
        per_upper = per_grid_edges[i+1]
        amp_lower = amp_grid_edges[j]
        amp_upper = amp_grid_edges[j+1]


        # for first 10000 sims, sample cells with equal probability
        if n_injected <= 10000:
            P = 1 
        else: 
            # after first 10000 sims, sample cells with a Gaussian centered on 0.5 and standard deviation of 0.15 (so cells with success rates near 0 and 1 are rarely sampled)
            P = 1/np.sqrt(2*0.3) * np.exp(-(0.5-success_rate[j,i])**2 / (0.3**2))
        
        if np.random.uniform() < P: 
            per = loguniform.rvs(per_lower, per_upper)
            amp = loguniform.rvs(amp_lower, amp_upper)
            phase = np.random.uniform(0, 2*np.pi)

            injected_per.append(per)
            injected_amp.append(amp)
            injected_phase.append(phase)

            injection_counts[j,i] += 1
            
            signal = 1+amp*np.sin(2*np.pi/per*times+phase)
            injected_flux = flux * signal

            power = LombScargle(times, injected_flux, flux_err).power(freq)
            per_max_power = 1/freq[np.argmax(power)]

            # consider the injection recovered if the fractional difference between the injected and recovered period are less than detection_threshold AND the power at the detected period is greater than window_fn_threshold_factor times the window function power at that period
            # this also accounts for the period aliases in aliases_to_check
            alias_uppers = [j*per+detection_threshold*j*per for j in aliases_to_check]
            alias_lowers = [j*per-detection_threshold*j*per for j in aliases_to_check]
            detection_in_range = [alias_lowers[i] < per_max_power < alias_uppers[i] for i in range(len(alias_lowers))]
            # if sum(detection_in_range) > 0 and not detection_in_range[int((len(aliases_to_check)-1)/2)]:
            #     plot_injection_recovery(times, injected_flux, signal, freq, power, per_max_power, per, window_fn_power, alias_lowers, alias_uppers)
            #     breakpoint()

            if sum(detection_in_range) > 0 and (power[np.argmax(power)] > window_fn_power[np.argmax(power)] * window_fn_threshold_factor): 
                recovery_counts[j,i] += 1

            n_injected += 1   
        success_rate = recovery_counts / injection_counts
        if n_injected % 500 == 0 and n_injected != 0:
            print(f'{n_injected} of {n_sims}')


    fig, ax = plt.subplots(2,1,figsize=(9,9))
    im = ax[0].imshow(injection_counts, origin='lower')
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes('right', size='5%', pad=0.15)
    cb = fig.colorbar(im, cax=cax, orientation='vertical')
    cb.set_label('Injections', rotation=270, fontsize=12)
    cb.ax.get_yaxis().labelpad = 15

    ax[0].set_yticks(np.arange(-0.5, n_amp_cells+0.5, 1))
    ax[0].set_xticks(np.arange(-0.5, n_per_cells+0.5, 1))
    ax[0].set_ylim(-0.5, n_amp_cells-0.5)
    ax[0].set_xlim(-0.5, n_per_cells-0.5)
    ax[0].set_yticklabels(FormatStrFormatter('%.2f').format_ticks(amp_grid_edges*100))
    ax[0].set_xticklabels([])
    ax[0].grid(alpha=0.3)
    # ax[0].set_aspect('equal')
    ax[0].tick_params(labelsize=12)
    ax[0].tick_params(axis='x',rotation=45)
    ax[0].set_ylabel('Amplitude (%)', fontsize=14)
    
    im = ax[1].imshow(success_rate, origin='lower')
    cs = ax[1].contour(success_rate, [0.5, 0.75, 0.95], cmap=plt.get_cmap('Greys'))
    ax[1].clabel(cs, cs.levels, inline=True, fontsize=12)


    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes('right', size='5%', pad=0.15)
    cb = fig.colorbar(im, cax=cax, orientation='vertical')
    cb.set_label('Detection Fraction', rotation=270, fontsize=12)
    cb.ax.get_yaxis().labelpad = 15

    ax[1].set_yticks(np.arange(-0.5, n_amp_cells+0.5, 1))
    ax[1].set_xticks(np.arange(-0.5, n_per_cells+0.5, 1))
    ax[1].set_ylim(-0.5, n_amp_cells-0.5)
    ax[1].set_xlim(-0.5, n_per_cells-0.5)
    ax[1].set_yticklabels(FormatStrFormatter('%.2f').format_ticks(amp_grid_edges*100))
    ax[1].set_xticklabels(FormatStrFormatter('%.2f').format_ticks(per_grid_edges))
    ax[1].grid(alpha=0.3)
    ax[1].tick_params(labelsize=12)
    ax[1].tick_params(axis='x',rotation=65)
    
    ax[1].set_xlabel('P (days)', fontsize=14)
    ax[1].set_ylabel('Amplitude (%)', fontsize=14)
    ax[0].set_title(field, fontsize=14)
    plt.tight_layout()

    # save the plot and injection/recovery data 
    output_dir = f'/data/tierras/fields/{field}/period_injection_recovery/'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    plt.savefig(output_dir+f'{field}_period_injection_recovery.png', dpi=300)
    plt.close('all')

    return 

if __name__ == '__main__':
    main()