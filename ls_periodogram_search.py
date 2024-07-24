from astropy.timeseries import LombScargle
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
plt.ioff()
from scipy.stats import sigmaclip
from median_filter import median_filter_uneven
from scipy.optimize import curve_fit 
import argparse 
import glob
import os 
from ap_phot import set_tierras_permissions
import warnings 
warnings.filterwarnings('ignore')
import matplotlib.gridspec as gridspec

def sine_model(x, a, c, d):
    return a*np.sin(2*np.pi*x+c)+d

def periodogram(x, y, y_err, target, field, sc=False):
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

    ls = LombScargle(x, y, y_err)

    freqmin = 1/x[-1] # search for periods up to the duration of the time coverage...
    freqmax = 1/(10/(24*60)) # ...down to 10 minutes
    nfreq = int(np.ceil(freqmax*(x[-1]-x[0])*20))
    freqs = np.linspace(freqmin, freqmax, nfreq)

    power = ls.power(freqs)

    use_inds = ~np.isnan(power)
    power = power[use_inds]
    freqs = freqs[use_inds]

    # TODO check aliases following VanderPlas (2018)
    fap = ls.false_alarm_probability(power.max(), method='baluev')

    per_peak = 1/freqs[np.argmax(power)]
    if fap < 0.001:
        if per_peak < 0.98 or per_peak > 1.02:
            periodogram_plot(x, y, y_err, freqs, power, target, field, save=True, color_by_time=True)

    return x, y, y_err, freqs, power

def periodogram_plot(x, y, y_err, freq, power, target, field, color_by_time=False, save=False):

    per = 1/freq 

    plt.ioff()
    window_fn_power = LombScargle(x, np.ones_like(x), fit_mean=False, center_data=False).power(freq)

    fig = plt.figure(figsize=(10,10))
    gs = gridspec.GridSpec(4, 1, figure=fig)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2], sharex=ax2)
    ax4 = fig.add_subplot(gs[3])

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

    ax1.set_title(target, fontsize=14)
    ax1.scatter(x, y, c=color, s=2)
    ax1.errorbar(x, y, y_err, linestyle="None",marker='',color=color,zorder=0)
    ax1.set_xlabel('BJD TDB', fontsize=14)
    ax1.set_ylabel('Normalized flux', fontsize=14)
    ax1.grid(alpha=0.7)
    
    ax2.plot(per, power, marker='')
    ax2.set_xscale('log')
    best_freq = freq[np.argmax(power)]
    best_per = 1/best_freq

    # best_per = 1/(best_freq + 2)

    ax2.plot(best_per, np.max(power), marker='o')
    ax2.set_ylabel('LS power', fontsize=14)
    ax2.grid(alpha=0.7, which='both')

    ax3.plot(per, window_fn_power, marker='')
    ax3.set_xscale('log')
    # ax3.set_yscale('log')
    ax3.set_xlabel('Period (d)', fontsize=14)
    ax3.set_ylabel('Window fn. power', fontsize=14)
    ax3.grid(alpha=0.7, which='both')

    phased_x = (x % best_per) / best_per 
    sort = np.argsort(phased_x)
    phased_x = phased_x[sort]
    phased_y = y[sort]
    phased_y_err = y_err[sort]

    ax4.scatter(phased_x, phased_y, c=color[sort], s=5)
    ax4.errorbar(phased_x, phased_y, phased_y_err, linestyle="None",marker='',color=color[sort],zorder=0)
    ax4.set_xlabel('Phase', fontsize=14)
    ax4.set_ylabel('Normalized flux', fontsize=14)
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
    
    # print(params[0])

    variability_model = sine_model(phased_x, params[0], params[1], params[2])
    chi_2_var = np.nansum((phased_y-variability_model)**2/(phased_y_err**2))
    bic_var = chi_2_var + (len(params)+1)*np.log(len(phased_y))
    flat_line_model = np.nanmedian(phased_y)
    chi_2_flat = np.nansum((phased_y-flat_line_model)**2/(phased_y_err**2))
    bic_flat = chi_2_flat + np.log(len(phased_y))

    ax4.plot(model_times, sine_model(model_times, params[0], params[1], params[2]), lw=2, color='k', label='Best-fit sine model')

    plt.tight_layout()
    # if target == 'Gaia DR3 4146924622363117056':
    #     breakpoint()
    delta_bic = bic_flat-bic_var

    if save:
        output_dir = f'/data/tierras/fields/{field}/sources/plots/ls_periodograms'
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
            set_tierras_permissions(output_dir)
        plt.savefig(output_dir+f'/{delta_bic:.2f}_{target}_ls_periodogram.png', dpi=200)
        set_tierras_permissions(output_dir+f'/{delta_bic:.2f}_{target}_ls_periodogram.png')
    if delta_bic < 6:
        print('Flat model - variability model BIC < 6!')
    plt.close()
    return 

def main(raw_args=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("-field", required=True, help="Name of observed target field exactly as shown in raw FITS files.")

    args = ap.parse_args()
    field = args.field 

    lc_path = f'/data/tierras/fields/{field}/sources/lightcurves/'
    source_file = glob.glob(f'/data/tierras/photometry/**/{field}/flat0000/**_sources.csv')[0]
    sources = pd.read_csv(source_file)
    lightcurves = glob.glob(lc_path+'**global_lc.csv')
    n_lcs = len(lightcurves)
    for i in range(n_lcs):
        target = lightcurves[i].split('/')[-1].split('_')[0]
        if target == field:
            with open(f'/data/tierras/fields/{field}/{field}_gaia_dr3_id.txt', 'r') as f:
                source_id = f.readline()
            source_id = source_id.split(' ')[-1]
        else:
            source_id = target 
        
        source_ind = np.where(sources['source_id'] == int(source_id.split(' ')[-1]))[0][0]
        source_rp = sources['phot_rp_mean_mag'][source_ind]

        print(f'Doing {target} ({i+1} of {n_lcs})')
        df = pd.read_csv(lightcurves[i], comment='#')
        x = np.array(df['BJD TDB'])
        y = np.array(df['Flux'])
        y_err = np.array(df['Flux Error'])

        # inflate errors by estimated night-to-night lower bound
        # this was determined BY EYE in measurements of night-to-nights in TIC362144730 
        error_inflation = 0.0001*np.exp((source_rp-10)/1.4)+0.0004
        # y_err = np.sqrt(y_err**2 + error_inflation**2) # add in quadrature? 
        y_err += error_inflation

        x, y, y_err, freq, power = periodogram(x, y, y_err, target, field, sc=True)


if __name__ == '__main__':
    main()