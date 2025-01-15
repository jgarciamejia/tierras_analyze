import numpy as np 
import matplotlib.pyplot as plt 
plt.ion()
import argparse 
import pandas as pd 
from ap_phot import t_or_f
from median_filter import median_filter_uneven
import os
from scipy.stats import sigmaclip
import emcee 
from celerite import terms 
import celerite 
from scipy.optimize import minimize
from astropy.timeseries import LombScargle
from scipy.signal import find_peaks, peak_widths 

def main(raw_args=None):
    def gp_mcmc(x, y, y_err, nsteps):
        def neg_log_like(params, y, gp):
            gp.set_parameter_vector(params)
            return -gp.log_likelihood(y)
    
        class CustomTerm(terms.Term):
            parameter_names = ('log_a', 'log_b', 'log_c', 'log_P')
            def get_real_coefficients(self, params):
                log_a, log_b, log_c, log_P = params 
                b = np.exp(log_b)
                return (np.exp(log_a)*(1.0+b)/(2.0+b), np.exp(log_c))
            
            def get_complex_coefficients(self, params):
                log_a, log_b, log_c, log_P = params 
                b = np.exp(log_b)
                return (np.exp(log_a)/(2.0+b), 0.0, np.exp(log_c), 2*np.pi*np.exp(-log_P))
            
            def neg_log_like(params, y, gp):
                gp.set_parameter_vector(params)
                return -gp.log_likelihood(y)
            
        def gp_log_probability(params):
            global best_log_prob, theta_save_gp, prob_save 
            gp.set_parameter_vector(params)
            lp = gp.log_prior()

            if not np.isfinite(lp): #This is only getting -inf, why? 
                return -np.inf
    
            log_prob = gp.log_likelihood(y) + lp 
            if log_prob > best_log_prob:
                prob_save = log_prob
                theta_save_gp = params
                best_log_prob = log_prob
            return log_prob

        global best_log_prob, theta_save_gp, prob_save 

        # use a Lomb Scargle to get a constraint on P 
        freq, power = LombScargle(x, y, y_err).autopower()
        per = 1/freq
        P_guess = per[np.argmax(power)]

        # TODO: set bounds on P using width of peak instead of sqrt guess that I've used here
        bounds = dict(log_a=(-10, 0),
                log_b=(-20, 20), 
                log_c=(-20, -17),
                log_P=(np.log(P_guess-np.sqrt(P_guess)), np.log(P_guess+np.sqrt(P_guess)))) 

        kernel = CustomTerm(log_a=-2, log_b=0.5, log_c=-19, log_P=np.log(P_guess), bounds=bounds)
        gp = celerite.GP(kernel, mean=1.0)
        gp.compute(x, y_err)
        initial = gp.get_parameter_vector()

        gp.compute(x, y_err)
        initial_params = gp.get_parameter_vector()
        bounds = gp.get_parameter_bounds()
        r = minimize(neg_log_like, initial_params, method='L-BFGS-B', bounds=bounds, args=(y, gp))
        gp.set_parameter_vector(r.x)

        #Run MCMC with GP 
        best_log_prob = -999999 #Initialize
        initial = gp.get_parameter_vector()
        ndim, nwalkers = len(initial), 64
        gp_sampler = emcee.EnsembleSampler(nwalkers, ndim, gp_log_probability)

        print("     Running GP burn-in for {} steps...".format(int(nsteps*0.1)))
        p0 = initial + 1e-6 * np.random.randn(nwalkers, ndim)
        p0, lp, _ = gp_sampler.run_mcmc(p0, int(nsteps*0.1))

        print("     Running GP production for {} steps...".format(nsteps))
        gp_sampler.reset()
        gp_sampler.run_mcmc(p0, nsteps)

        return gp, gp_sampler

    ap = argparse.ArgumentParser()

    ap.add_argument('-field', required=True, help='Name of field')
    ap.add_argument('-gaia_id', required=False, default=None, help='Gaia source_id of target in field for which to run periodogram. If None passed, will use the field name as the target.')
    ap.add_argument('-ffname', required=False, default='flat0000', help='Name of flat directory to use for reading light curves.')
    ap.add_argument('-median_filter_w', required=False, type=float, default=0, help='Width of median filter in days to regularize data')
    ap.add_argument('-quality_mask', required=False, default='False', type=str)
    ap.add_argument('-sigmaclip', required=False, default='True', type=str, help='Whether or not to sigma clip the data.')

    args = ap.parse_args(raw_args)
    field = args.field
    gaia_id = args.gaia_id
    ffname = args.ffname 
    median_filter_w = args.median_filter_w
    quality_mask = t_or_f(args.quality_mask)
    sc = t_or_f(args.sigmaclip)

    if gaia_id is None:
        target = field 
    else:
        target = f'Gaia DR3 {gaia_id}'

    try:
        df = pd.read_csv(f'/data/tierras/fields/{field}/sources/lightcurves/{ffname}/{target}_global_lc.csv', comment='#')
    except:
        return None, None, None
    
    x = np.array(df['BJD TDB'])
    y = np.array(df['Flux'])
    y_err = np.array(df['Flux Error'])
    # flux_flag = np.array(df['Low Flux Flag']).astype(bool)
    wcs_flag = np.array(df['WCS Flag']).astype(bool)
    pos_flag = np.array(df['Position Flag']).astype(bool)
    fwhm_flag = np.array(df['FWHM Flag']).astype(bool)
    flux_flag = np.array(df['Flux Flag']).astype(bool)

    # check for file indicating start/end times of transits; if it exists, use it to mask out in-transit points   
    if os.path.exists(f'/data/tierras/fields/{field}/{field}_transit_times.csv'):
        transit_time_df = pd.read_csv(f'/data/tierras/fields/{field}/{field}_transit_times.csv')
        start_times = np.array(transit_time_df['tstart'])
        end_times = np.array(transit_time_df['tend'])
        transit_inds = np.ones_like(x, dtype='bool')
        for i in range(len(start_times)):
            transit_inds[np.where((x >= start_times[i]) & (x <= end_times[i]))[0]] = False
        x = x[transit_inds]
        y = y[transit_inds]
        y_err = y_err[transit_inds]
        wcs_flag = wcs_flag[transit_inds]
        pos_flag = pos_flag[transit_inds]
        fwhm_flag = fwhm_flag[transit_inds]
        flux_flag = flux_flag[transit_inds]

    if quality_mask: 
        mask = np.where(~(wcs_flag | pos_flag | fwhm_flag | flux_flag))[0]

        x = x[mask]
        y = y[mask]
        y_err = y_err[mask]

    nan_inds = ~np.isnan(y)
    x = x[nan_inds]
    y = y[nan_inds]
    y_err = y_err[nan_inds]
    
    # renormalize with masks applied 
    norm = np.nanmedian(y)
    y /= norm 
    y_err /= norm 
 
    if median_filter_w != 0:
        print(f'Median filtering target flux with a filter width of {median_filter_w} days.')
        x_filter, y_filter = median_filter_uneven(x, y, median_filter_w)
        mu = np.median(y)

        # plt.figure()
        # plt.plot(x,y)
        # plt.plot(x,y_filter)
        # plt.plot(x,mu*y/y_filter)
        y =  mu*y/(y_filter)
    
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
        bx = np.zeros(len(x_list))
        by = np.zeros_like(bx)
        bye = np.zeros_like(bx)
        # sigmaclip each night 
        for i in range(len(x_list)):
            use_inds = np.where((x>=x_list[i][0])&(x<=x_list[i][-1]))[0]
            v, l, h = sigmaclip(y[use_inds], 3, 3) 
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

    x_offset = x[0]
    x -= x_offset

    # get binned data 
    x_deltas = np.array([x[i]-x[i-1] for i in range(1,len(x))])
    x_breaks = np.where(x_deltas > 0.4)[0]
    x_list = []
    for i in range(len(x_breaks)):
        if i == 0:
            x_list.append(x[0:x_breaks[i]+1])
        else:
            x_list.append(x[x_breaks[i-1]+1:x_breaks[i]+1])
    x_list.append(x[x_breaks[-1]+1:len(x)])
    bx = np.zeros(len(x_list))
    by = np.zeros_like(bx)
    bye = np.zeros_like(bx)
    for i in range(len(x_list)):
        inds = np.where((x >= x_list[i][0]) & (x <= x_list[i][-1]))[0]
        bx[i] = np.median(x[inds])
        by[i] = np.median(y[inds])
        bye[i] = np.std(y[inds])/np.sqrt(len(y[inds]))

    gp, gp_sampler = gp_mcmc(x, y, y_err, 1000)
    samples = gp_sampler.flatchain
    random_samps = samples[np.random.randint(len(samples), size=200)]

    x_model = np.linspace(min(x), max(x), 10000)
    cm = plt.get_cmap('viridis')
    random_sample_color = 120
    fig, ax = plt.subplots(1, 1, figsize=(20,5), sharey=True, sharex='col')
    ax.set_ylabel('Normalized Flux', fontsize=16)
    ax.plot(x, y, marker='o', color='#C0C0C0', ls='')
    #Plot random samples

    for s in random_samps:
        gp.set_parameter_vector(s)
        mu = gp.predict(y, x_model, return_cov=False)
        ax.plot(x_model, mu, color=cm(random_sample_color), alpha=0.02, zorder=1)

    ax.errorbar(bx, by, bye, marker='o', mfc='none', mec='k', mew=1.5, ls='', zorder=4)
    ax.grid(alpha=0.5)
    ax.set_xlabel('BJD$_{TDB} -$'+f'{x_offset}', fontsize=14)
    ax.set_ylabel('Normalized Flux', fontsize=14)
    ax.tick_params(labelsize=12)
    plt.tight_layout()
    breakpoint()

if __name__ == '__main__':
    main()