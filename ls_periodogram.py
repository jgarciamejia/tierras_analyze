from astropy.timeseries import LombScargle
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
plt.ioff()
from scipy.stats import sigmaclip
from median_filter import median_filter_uneven
from scipy.optimize import curve_fit 
import argparse 
from ap_phot import t_or_f
from scipy.signal import find_peaks
import os 
import matplotlib
from scipy.signal import argrelextrema
from scipy.optimize import curve_fit 

def linear_model(x, m, b):
    return m*x+b

def sine_model(x, a, c):
    return a*np.sin(2*np.pi*x+c)+1

def periodogram(x, y, y_err, pers=None, sc=False):
    # remove NaNs
    use_inds = ~np.isnan(y)
    x = x[use_inds]
    y = y[use_inds]
    y_err = y_err[use_inds]

    if sc:
        # get times of each night 
        x_deltas = np.array([x[i]-x[i-1] for i in range(1,len(x))])
        x_breaks = np.where(x_deltas > 0.4)[0]
        x_list = []
        for i in range(len(x_breaks)):
            if i == 0:
                x_list.append(x[0:x_breaks[i]+1])
            else:
                x_list.append(x[x_breaks[i-1]+1:x_breaks[i]+1])
        x_list.append(x[x_breaks[-1]+1:len(x)])

        x_list_2 = [] 
        y_list = []
        y_err_list = []
        # sigmaclip each night 
        for i in range(len(x_list)):
            use_inds = np.where((x>=x_list[i][0])&(x<=x_list[i][-1]))[0]
            v, l, h = sigmaclip(y[use_inds], 3, 3) # TODO: should median filter y first 
            sc_inds = np.where((y[use_inds] > l) & (y[use_inds] < h))[0]
            x_list_2.extend(x[use_inds][sc_inds])
            y_list.extend(y[use_inds][sc_inds])
            y_err_list.extend(y_err[use_inds][sc_inds])
        
        x = np.array(x_list_2)
        y = np.array(y_list)
        y_err = np.array(y_err_list)

        v, l, h = sigmaclip(y[~np.isnan(y)])
        use_inds = np.where((y>l)&(y<h))[0]
        x = x[use_inds]
        y = y[use_inds]
        y_err = y_err[use_inds]

        # v, l, h = sigmaclip(y_err[~np.isnan(y_err)])
        # use_inds = np.where(y_err<h)[0]
        # x = x[use_inds]
        # y = y[use_inds]
        # y_err = y_err[use_inds] 

    x_offset = x[0]
    x -= x_offset

    if pers is None:
        freqs, power = LombScargle(x, y, y_err).autopower(maximum_frequency=1/per_lower)
        pers = 1/freqs
    else:
        freqs = 1/pers
        power = LombScargle(x, y, y_err).power(freqs)

    return x, y, y_err, pers, freqs, power, x_offset

def periodogram_plot(x, y, y_err, per, power, window_fn_power, x_offset, target, baseline_restarts=False, color_by_time=False):
    def on_click(event):
        ''' allow the user to click on different periodogram peaks and phase on them '''

        global highlight 

        ax = event.inaxes
        if ax is not None: 
            label = axes_mapping.get(ax, 'Unknown axis')
        else:
            return 
        
        if label != 'ax2':
            return

        xdata = event.xdata 
        ydata = event.ydata 
        
        dists = np.sqrt((np.log10(xdata) - np.log10(per))**2 + (ydata - power)**2)
        point = np.argmin(dists)

        print(f'Per = {per[point]:.2f} d, pow = {power[point]:.2f}')

        save_ax2_xlim = ax2.get_xlim()
        save_ax2_ylim = ax2.get_ylim()

        if highlight:
            ax1.lines[-1].remove()
            ax2.lines[-1].remove()

        highlight = ax2.plot(per[point], power[point], marker='o', color='m', label=f'P = {per[point]:.2f} d')
        ax2.legend()
        ax2.set_xlim(save_ax2_xlim)
        ax2.set_ylim(save_ax2_ylim)

        ax4.cla()

        best_per = per[point]

        phased_x = (x % best_per) / best_per 

        sort = np.argsort(phased_x)
        phased_x = phased_x[sort]
        phased_y = y[sort]
        phased_y_err = y_err[sort]

        ax4.scatter(phased_x, phased_y, c=color[sort], s=5)
        ax4.errorbar(phased_x, phased_y, phased_y_err, linestyle="None",marker='',ecolor=color[sort],zorder=0)
        ax4.set_xlabel('Phase', fontsize=14)
        ax4.set_ylabel('Normalized Flux', fontsize=14)
        ax4.grid(alpha=0.7)

        model_times = np.linspace(phased_x[0], phased_x[-1], 10000)
        model_amp = 0.05
        model_phase = 0
        model_offset = 1

        params, params_covariance = curve_fit(sine_model, phased_x, phased_y, sigma=phased_y_err, p0=[model_amp, model_phase]) 
        print(f'Amplitude: {abs(params[0])*1e3:.1f} ppt')
        
        phase_bin = 0.05
        n_bin = int(1/phase_bin)
        bx = np.zeros(n_bin)
        by = np.zeros(n_bin)
        bye = np.zeros(n_bin)
        for i in range(n_bin):
            phase_start = i*phase_bin
            phase_end = (i+1)*phase_bin
            bx[i] = (phase_start + phase_end)/2

            inds = np.where((phased_x >= phase_start) & (phased_x < phase_end))[0]
            if len(inds) == 0:
                by[i] = np.nan
                bye[i] = np.nan
            else:
                #by[i] = np.nanmean(phased_y[inds])
                by[i] = np.nansum((1/phased_y_err[inds])**2*phased_y[inds])/np.nansum((1/phased_y_err[inds])**2)
                bye[i] = np.nanstd(phased_y[inds])/np.sqrt(len(~np.isnan(phased_y[inds])))
        ax4.errorbar(bx, by, bye, marker='o', color='#FF0000', zorder=4, ls='', ms=7, mew=2, mfc='none', mec='#FF0000', ecolor='#FF0000')
        
        ax4.plot(model_times, sine_model(model_times, params[0], params[1]), lw=2, color='#b0b0b0', label='Best-fit sine model')

        unphased_model = params[0]*np.sin(2*np.pi*unphased_model_times/best_per+params[1])+1
        ax1.plot(unphased_model_times, unphased_model, color='#b0b0b0', zorder=0)

        return 

    fig = plt.figure(figsize=(15,10))
    ax1 = plt.subplot(311)
    ax2 = plt.subplot(312)
    #ax3 = plt.subplot(413, sharex=ax2)
    ax4 = plt.subplot(313)

    global highlight
    cid = fig.canvas.mpl_connect('button_press_event', on_click)
    axes_mapping = {ax1: 'ax1', ax2: 'ax2', ax4: 'ax4'}
    # axes_mapping = {ax1: 'ax1', ax2: 'ax2', ax3: 'ax3', ax4: 'ax4'}
    ax1.tick_params(labelsize=12)
    ax2.tick_params(labelsize=12)
    # ax3.tick_params(labelsize=12)
    ax4.tick_params(labelsize=12)

    if color_by_time: 
        cmap = matplotlib.colormaps['viridis']
        inds = np.arange(len(x))
        color_inds = np.array([int(i) for i in inds*255/len(x)])
        color = cmap(color_inds)
    else:
        color='#b0b0b0'

    ax1.set_title(target, fontsize=14)
    ax1.scatter(x, y, c=color, s=2)
    ax1.errorbar(x, y, y_err, linestyle="None",marker='',ecolor=color,zorder=0)
    if baseline_restarts:
        for i in range(len(baseline_dates)):
            if x[0] < baseline_dates[i]-x_offset < x[-1]: # only plot if it falls within the time range of observations 
                ax1.axvline(baseline_dates[i]-x_offset, color='r', alpha=0.6)
    ax1.set_xlabel(f'BJD TDB - {x_offset:.4f}', fontsize=14)
    ax1.set_ylabel('Normalized Flux', fontsize=14)
    ax1.grid(alpha=0.7)
    
    ax2.plot(per, window_fn_power, marker='.', color='tab:orange', label='Window fn.')
    ax2.plot(per, power, marker='.', color='tab:blue', label='Data')
    ax2.set_xscale('log')

    # peaks = find_peaks(power, prominence=0.02)
    # peak_pers = per[peaks[0]]
    # peak_pows = power[peaks[0]]

    # ax2.plot(peak_pers, peak_pows, marker='o', color='tab:pink', mew=1.5, mfc='none', ls='')

    best_per = per[np.argmax(power)]
    # ax2.plot(best_per, np.max(power), marker='o', label=f'P={best_per:.2f} d')
    highlight = ax2.plot(best_per, np.max(power), marker='o', color='m', label=f'P = {best_per:.2f} d')
    ax2.set_xlabel('Period (d)', fontsize=14)
    ax2.set_ylabel('Power', fontsize=14)
    ax2.legend() 
    ax2.grid(alpha=0.7)

    # best_per = 2.48978
    # ax3.plot(per, window_fn_power, marker='.')
    # ax3.set_xscale('log')
    # ax3.set_ylabel('Window fn. power', fontsize=14)
    # ax3.set_xlabel('Period (d)', fontsize=14)
    # ax3.grid(alpha=0.7)

    phased_x = (x % best_per) / best_per 
    sort = np.argsort(phased_x)
    phased_x = phased_x[sort]
    phased_y = y[sort]
    phased_y_err = y_err[sort]

    ax4.scatter(phased_x, phased_y, c=color[sort], s=5)
    ax4.errorbar(phased_x, phased_y, phased_y_err, linestyle="None",marker='',ecolor=color[sort],zorder=0)
    ax4.set_xlabel('Phase', fontsize=14)
    ax4.set_ylabel('Normalized Flux', fontsize=14)
    ax4.grid(alpha=0.7)

    model_times = np.arange(phased_x[0], phased_x[-1], 0.001)
    model_amp = 0.05
    model_phase = 0
    model_offset = 1

    params, params_covariance = curve_fit(sine_model, phased_x, phased_y, sigma=phased_y_err, p0=[model_amp, model_phase]) 
    print(f'Amplitude: {abs(params[0])*1e3:.1f} ppt')
    
    phase_bin = 0.05
    n_bin = int(1/phase_bin)
    bx = np.zeros(n_bin)
    by = np.zeros(n_bin)
    bye = np.zeros(n_bin)
    for i in range(n_bin):
        phase_start = i*phase_bin
        phase_end = (i+1)*phase_bin
        bx[i] = (phase_start + phase_end)/2

        inds = np.where((phased_x >= phase_start) & (phased_x < phase_end))[0]
        if len(inds) == 0:
            by[i] = np.nan
            bye[i] = np.nan
        else:
            #by[i] = np.nanmean(phased_y[inds])
            by[i] = np.nansum((1/phased_y_err[inds])**2*phased_y[inds])/np.nansum((1/phased_y_err[inds])**2)
            bye[i] = np.nanstd(phased_y[inds])/np.sqrt(len(~np.isnan(phased_y[inds])))
    ax4.errorbar(bx, by, bye, marker='o', color='#FF0000', zorder=4, ls='', ms=7, mew=2, mfc='none', mec='#FF0000', ecolor='#FF0000')
    
    ax4.plot(model_times, sine_model(model_times, params[0], params[1]), lw=2, color='#b0b0b0', label='Best-fit sine model')

    # add plot of unphased model to first panel
    unphased_model_times = np.arange(x[0], x[-1], 0.001)
    unphased_model = params[0]*np.sin(2*np.pi*unphased_model_times/best_per+params[1])+1
    ax1.plot(unphased_model_times, unphased_model, color='#b0b0b0', zorder=0)

    plt.tight_layout()


    breakpoint()
    return fig, (ax1, ax2, ax4)

def main(raw_args=None):
    ap = argparse.ArgumentParser()

    ap.add_argument('-field', required=True, help='Name of field')
    ap.add_argument('-gaia_id', required=False, default=None, help='Gaia source_id of target in field for which to run periodogram. If None passed, will use the field name as the target.')
    ap.add_argument('-ffname', required=False, default='flat0000', help='Name of flat directory to use for reading light curves.')
    ap.add_argument('-median_filter_w', required=False, type=float, default=0, help='Width of median filter in days to regularize data')
    ap.add_argument('-quality_mask', required=False, default='True', type=str)
    ap.add_argument('-flux_flag_level', required=False, default=0.8, type=float, help="Should be between 0 and 1. If passed, apply custom masking on flux levels using normalized ALC. Points below flux_flag_level will be ignored.")
    ap.add_argument('-sigmaclip', required=False, default='True', type=str, help='Whether or not to sigma clip the data.')
    ap.add_argument('-autofreq', required=False, default='True', type=str, help='Whether or not to use astropys default algorithm to establish the frequency grid. If False, per_low, per_hi, and per_resolution will be used. ')
    ap.add_argument('-per_low', required=False, default=1/24, type=float, help='Lower period (in days) to use to establish frequency grid IF autofreq is False.')
    ap.add_argument('-per_hi', required=False, default=100, type=float, help='Upper period (in days) to use to establish frequency grid IF autofreq is False.')
    ap.add_argument('-per_resolution', required=False, default=15/86400, type=float, help='Period resolution (in days) to use to establish frequency grid IF autofreq is False.')
    ap.add_argument('-baseline_restarts', required=False, default='True', type=str, help='Re-baseline the data using the camera restart dates in /data/tierras/fields/camera_restart_dates.csv')
    args = ap.parse_args(raw_args)
    field = args.field
    gaia_id = args.gaia_id
    ffname = args.ffname 
    median_filter_w = args.median_filter_w
    quality_mask = t_or_f(args.quality_mask)
    flux_flag_level = args.flux_flag_level
    sc = t_or_f(args.sigmaclip)
    autofreq = t_or_f(args.autofreq)
    global per_lower, per_upper 
    per_lower = args.per_low
    per_upper = args.per_hi
    per_res = args.per_resolution
    baseline_restarts = t_or_f(args.baseline_restarts)
    if gaia_id is None:
        target = field 
    else:
        target = f'Gaia DR3 {gaia_id}'

    if autofreq: 
        pers = None 
    else:
        pers = np.arange(per_lower, per_upper, per_res)

    try:
        df = pd.read_csv(f'/data/tierras/fields/{field}/sources/lightcurves/{ffname}/{target}_global_lc.csv', comment='#')
    except:
        return None, None, None
    x = np.array(df['BJD TDB'])
    y = np.array(df['Flux'])
    y_err = np.array(df['Flux Error'])
    alc = np.array(df['ALC'])
    # flux_flag = np.array(df['Low Flux Flag']).astype(bool)
    wcs_flag = np.array(df['WCS Flag']).astype(bool)
    pos_flag = np.array(df['Position Flag']).astype(bool)
    fwhm_flag = np.array(df['FWHM Flag']).astype(bool)
    flux_flag = np.array(df['Flux Flag']).astype(bool)

    # check for file indicating start/end times of transits; if it exists, use it to mask out in-transit points   
    if os.path.exists(f'/data/tierras/fields/{field}/{field}_transit_times.csv') and target == field:
        transit_time_df = pd.read_csv(f'/data/tierras/fields/{field}/{field}_transit_times.csv')
        start_times = np.array(transit_time_df['tstart'])
        end_times = np.array(transit_time_df['tend'])
        transit_inds = np.ones_like(x, dtype='bool')
        for i in range(len(start_times)):
            transit_inds[np.where((x >= start_times[i]) & (x <= end_times[i]))[0]] = False
        print(f'Masking {len(x)-sum(transit_inds)} in-transit_points.')
        x = x[transit_inds]
        y = y[transit_inds]
        y_err = y_err[transit_inds]
        wcs_flag = wcs_flag[transit_inds]
        pos_flag = pos_flag[transit_inds]
        fwhm_flag = fwhm_flag[transit_inds]
        flux_flag = flux_flag[transit_inds]
        alc = alc[transit_inds]

    # alternatively, the user can declare a list of linear ephemerides for planets in the system; if this file exists, use it to mask in-transit points 
    if os.path.exists(f'/data/tierras/fields/{field}/{field}_transit_ephemerides.csv') and target == field:
        ephem_df = pd.read_csv(f'/data/tierras/fields/{field}/{field}_transit_ephemerides.csv')
        for i in range(len(ephem_df)):
            t0 = ephem_df['t0'][i]
            per = ephem_df['period'][i]
            dur = ephem_df['duration'][i]
            start_n = int(np.ceil((x[0]-t0)/per))
            end_n = int(np.floor(x[-1]-t0)/per)
            tn = t0 + per*np.arange(start_n, end_n+2)
            transit_inds = np.ones_like(x, dtype='bool')
            for j in range(len(tn)):
                transit_inds[np.where((x > tn[j]-dur/2) & (x < tn[j] + dur/2))[0]] = False
            print(f'Masking {len(x)-sum(transit_inds)} in-transit_points.')
            x = x[transit_inds]
            y = y[transit_inds]
            y_err = y_err[transit_inds]
            wcs_flag = wcs_flag[transit_inds]
            pos_flag = pos_flag[transit_inds]
            fwhm_flag = fwhm_flag[transit_inds]
            flux_flag = flux_flag[transit_inds]
            alc = alc[transit_inds]
    
    
         

    # fit the alc to correct for flux loss due to mirror getting dirtier 
    # do it for each camera restart season 
    mirror_df = pd.read_csv('/data/tierras/fields/mirror_cleaning_dates.csv', comment='#')
    global mirror_dates
    mirror_dates = np.array(mirror_df['jd'])
    mirror_fit = np.zeros(len(x))

    fig, ax = plt.subplots(2, figsize=(8,6), sharex=True)
    ax[0].plot(x, alc/np.median(alc), label='Normalized ALC')
    ax[0].tick_params(labelsize=12)
    ax[0].grid(alpha=0.5)
    ax[0].set_ylabel('Norm. ALC flux', fontsize=14)

    for i in range(len(mirror_dates)):
        labeled = False
        if mirror_dates[i] > x[0]:
            if not labeled:
                ax[0].axvline(mirror_dates[i], color='k', ls='--', label='Mirror cleaning dates')
                labeled = True
            else:
                ax[0].axvline(mirror_dates[i], color='k', ls='--')

    labeled = False
    for i in range(len(mirror_dates)+1):
        if i == 0:
            x_start = 0 
        else:
            x_start = mirror_dates[i-1]
        if i == len(mirror_dates):
            x_end = x[-1]
        else:
            x_end = mirror_dates[i]
        inds = np.where((x >= x_start) & (x <= x_end))[0]
        if len(inds) == 0:
            continue 
        
        order = max([int(len(inds)/100), 10])
        maxima_indices = inds[argrelextrema(alc[inds]/np.median(alc[inds]), np.greater, order=order)[0]]

        # coeffs = np.polyfit(x[maxima_indices] - x[maxima_indices[0]], alc[maxima_indices]/np.median(alc), 1)

        coeffs, pcov = curve_fit(linear_model, x[maxima_indices] - x[maxima_indices[0]], alc[maxima_indices]/np.median(alc), bounds=([-np.inf, -np.inf], [0, np.inf]))

        
        fit = coeffs[0]*(x[inds] - x[maxima_indices[0]]) +coeffs[1]
       

        if not labeled:
            ax[0].plot(x[maxima_indices], alc[maxima_indices]/np.median(alc), 'rx', label='Fit points')
        else:
            ax[0].plot(x[maxima_indices], alc[maxima_indices]/np.median(alc), 'rx')
        

        mirror_fit[inds] = fit 
    
        if not labeled:
            ax[0].plot(x[inds], fit, color='tab:orange', label='Mirror dirtying fit')
            labeled = True
        else:
            ax[0].plot(x[inds], fit, color='tab:orange')
    
    ax[0].legend()

    mirror_corrected_flux = (alc/np.median(alc))/mirror_fit

    ax[1].plot(x, mirror_corrected_flux, label='Dirtying-corrected ALC flux')
    ax[1].axhline(flux_flag_level, color='tab:red', ls='--', label='Flux flag level')
    ax[1].legend()
    ax[1].set_xlabel('Time (BJD)', fontsize=14)
    ax[1].grid(alpha=0.5)
    ax[1].set_ylabel('Corrected Flux', fontsize=14)
    ax[1].tick_params(labelsize=12)
    fig.tight_layout()

    # now mask out low flux nights 
    # breakpoint()


    if quality_mask: 
        flux_flag = np.zeros_like(wcs_flag)
        inds = np.where(mirror_corrected_flux < flux_flag_level)
        flux_flag[inds] = True

        mask = np.where(~(wcs_flag | pos_flag | fwhm_flag | flux_flag))[0]

        x = x[mask]
        y = y[mask]
        y_err = y_err[mask]

    nan_inds = ~np.isnan(y) & ~np.isnan(y_err)
    x = x[nan_inds]
    y = y[nan_inds]
    y_err = y_err[nan_inds]
    
    # renormalize with masks applied 
    norm = np.nanmedian(y)
    y /= norm 
    y_err /= norm 

    if baseline_restarts:
        baseline_df = pd.read_csv('/data/tierras/fields/camera_restart_dates.csv', comment='#')
        global baseline_dates
        baseline_dates = np.array(baseline_df['jd'])
        for i in range(len(baseline_dates)+1):
            if i == 0:
                x_start = x[0]
            else:
                x_start = baseline_dates[i-1]
            if i == len(baseline_dates):
                x_end = x[-1]
            else:
                x_end = baseline_dates[i]
            inds = np.where((x >= x_start) & (x <= x_end))[0]
            norm = np.nanmedian(y[inds])
            y[inds] /= norm 
            y_err[inds] /= norm
    
    
    if median_filter_w != 0:
        print(f'Median filtering target flux with a filter width of {median_filter_w} days.')
        x_filter, y_filter = median_filter_uneven(x, y, median_filter_w)
        mu = np.median(y)

        # plt.figure()
        # plt.plot(x,y)
        # plt.plot(x,y_filter)
        # plt.plot(x,mu*y/y_filter)
        y =  mu*y/(y_filter)

    x, y, y_err, per, freq, power, x_offset = periodogram(x, y, y_err, pers=pers, sc=sc)

    # calculate the window function power of the data over the frequency grid 
    window_fn_power = LombScargle(x, np.ones_like(x), fit_mean=False, center_data=False).power(freq)
    fig, ax = periodogram_plot(x, y, y_err, per, power, window_fn_power, x_offset, target, baseline_restarts, color_by_time=True)

    return fig, ax, power


if __name__ == '__main__':
    main()
