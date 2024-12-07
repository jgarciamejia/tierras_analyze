from astropy.timeseries import LombScargle
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
plt.ion()
from scipy.stats import sigmaclip
from median_filter import median_filter_uneven
from scipy.optimize import curve_fit 

def sine_model(x, a, c, d):
    return a*np.sin(2*np.pi*x+c)+d

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

        v, l, h = sigmaclip(y_err[~np.isnan(y_err)])
        use_inds = np.where(y_err<h)[0]
        x = x[use_inds]
        y = y[use_inds]
        y_err = y_err[use_inds] 

    x -= x[0]

    if pers is None:
        freqs, power = LombScargle(x, y, y_err).autopower()
        pers = 1/freqs
    else:
        freqs = 1/pers
        power = LombScargle(x, y, y_err).power(freqs)

    return x, y, y_err, pers, freqs, power

def periodogram_plot(x, y, y_err, per, power, window_fn_power, color_by_time=False):
    fig = plt.figure(figsize=(9,9))
    ax1 = plt.subplot(411)
    ax2 = plt.subplot(412)
    ax3 = plt.subplot(413, sharex=ax2)
    ax4 = plt.subplot(414)

    ax1.tick_params(labelsize=12)
    ax2.tick_params(labelsize=12)
    ax3.tick_params(labelsize=12)
    ax4.tick_params(labelsize=12)

    if color_by_time: 
        cmap = plt.get_cmap('viridis')
        inds = np.arange(len(x))
        color_inds = np.array([int(i) for i in inds*255/len(x)])
        color = cmap(color_inds)
    else:
        color='#b0b0b0'

    ax1.scatter(x, y, c=color, s=2)
    ax1.errorbar(x, y, y_err, linestyle="None",marker='',color=color,zorder=0)
    ax1.set_xlabel('BJD TDB', fontsize=14)
    ax1.set_ylabel('Normalized Flux', fontsize=14)
    
    ax2.plot(per, power, marker='.')
    ax2.set_xscale('log')
    best_per = per[np.argmax(power)]
    ax2.plot(best_per, np.max(power), marker='o')
    ax2.set_xlabel('Period (d)', fontsize=14)
    ax2.set_ylabel('Power', fontsize=14)
    
    # best_per = 2.48978
    ax3.plot(per, window_fn_power, marker='.')
    ax3.set_xscale('log')
    ax3.set_ylabel('Window fn. power', fontsize=14)
    ax3.set_xlabel('Period (d)', fontsize=14)

    phased_x = (x % best_per) / best_per 
    sort = np.argsort(phased_x)
    phased_x = phased_x[sort]
    phased_y = y[sort]
    phased_y_err = y_err[sort]

    ax4.scatter(phased_x, phased_y, c=color[sort], s=5)
    ax4.errorbar(phased_x, phased_y, phased_y_err, linestyle="None",marker='',color=color[sort],zorder=0)
    ax4.set_xlabel('Phase', fontsize=14)
    ax4.set_ylabel('Normalized Flux', fontsize=14)
    ax4.grid(alpha=0.7)

    model_times = np.linspace(phased_x[0], phased_x[-1], 10000)
    model_amp = 0.05
    model_phase = 0
    model_offset = 1

    params, params_covariance = curve_fit(sine_model, phased_x, phased_y, sigma=phased_y_err, p0=[model_amp, model_phase, model_offset]) 
    
    phase_bin = 0.05
    n_bin = int(1/phase_bin)
    bx = np.zeros(n_bin)
    by = np.zeros(n_bin)
    bye = np.zeros(n_bin)
    for i in range(n_bin):
        phase_start = i*phase_bin
        phase_end = (i+1)*phase_bin
        inds = np.where((phased_x >= phase_start) & (phased_x < phase_end))[0]
        bx[i] = (phase_start + phase_end)/2
        by[i] = np.nanmean(phased_y[inds])
        bye[i] = np.nanstd(phased_y[inds])/np.sqrt(len(~np.isnan(phased_y[inds])))
    ax4.errorbar(bx, by, bye, marker='o', color='#FF0000', zorder=4, ls='', ms=7, mew=2, mfc='none', mec='#FF0000', ecolor='#FF0000')
    
    ax4.plot(model_times, sine_model(model_times, params[0], params[1], params[2]), lw=2, color='k', label='Best-fit sine model')
    plt.tight_layout()
    return 

if __name__ == '__main__':
    field = 'WASP-156'
    ffname = 'flat0000'
    median_filter_w = 0
    quality_mask = False 
    target = field
    pers = np.arange(1/24, 100, 15/86400)
    # pers = np.arange(1/24, 1, 1/86400)
    # pers = None
    sc = True

    df = pd.read_csv(f'/data/tierras/fields/{field}/sources/lightcurves/{ffname}/{target}_global_lc.csv', comment='#')
    x = np.array(df['BJD TDB'])
    y = np.array(df['Flux'])
    y_err = np.array(df['Flux Error'])
    # flux_flag = np.array(df['Low Flux Flag']).astype(bool)
    wcs_flag = np.array(df['WCS Flag']).astype(bool)
    pos_flag = np.array(df['Position Flag']).astype(bool)
    fwhm_flag = np.array(df['FWHM Flag']).astype(bool)
    flux_flag = np.array(df['Flux Flag']).astype(bool)

    if quality_mask: 
        mask = np.where(~(wcs_flag | pos_flag | fwhm_flag | flux_flag))[0]

        x = x[mask]
        y = y[mask]
        y_err = y_err[mask]

    nan_inds = ~np.isnan(y)
    x = x[nan_inds]
    y = y[nan_inds]
    y_err = y_err[nan_inds]
    
    if median_filter_w != 0:
        print(f'Median filtering target flux with a filter width of {median_filter_w} days.')
        x_filter, y_filter = median_filter_uneven(x, y, median_filter_w)
        mu = np.median(y)

        # plt.figure()
        # plt.plot(x,y)
        # plt.plot(x,y_filter)
        # plt.plot(x,mu*y/y_filter)
        y =  mu*y/(y_filter)

    x, y, y_err, per, freq, power = periodogram(x, y, y_err, pers=pers, sc=sc)

    # calculate the window function power of the data over the frequency grid 
    window_fn_power = LombScargle(x, np.ones_like(x), fit_mean=False, center_data=False).power(freq)
    periodogram_plot(x, y, y_err, per, power, window_fn_power, color_by_time=True)

   

    breakpoint()
