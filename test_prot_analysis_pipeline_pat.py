import os
import sys
import pdb
import glob
import copy
import argparse 
import importlib
import numpy as np
import pandas as pd
from time import time
import matplotlib.pylab as plt
from astropy.timeseries import LombScargle
from astropy.time import Time
from scipy.stats import sigmaclip, pearsonr
import pickle 

import pymc3 as pm 
import pymc3_ext as pmx 
from celerite2.theano import terms, GaussianProcess 

import test_load_data as ld
from test_mearth_style_for_tierras_pat import mearth_style, mearth_style_pat, mearth_style_pat_weighted
from test_find_rotation_period import build_model, sigma_clip
from test_bin_lc import ep_bin 
from corrected_flux_plot import reference_flux_correction
from noise_calculation import noise_component_plot 
from median_filter import median_filter_uneven
from ap_phot import regression as regress
from flare_masker import flare_masker

# Contributors (so far): JIrwin, EPass, JGarciaMejia, PTamburo.

# Deal with command line
ap = argparse.ArgumentParser()
ap.add_argument("-target", required=True, help="Name of observed target exactly as shown in raw FITS files.")       
ap.add_argument("-ffname", required=False, default='flat0000', help="Name of folder in which to store reduced+flattened data. Convention is flatXXXX. XXXX=0000 means no flat was used.")
ap.add_argument("-N", required=False, type=int, default=1, help="Number of sinusoids to use in the Hartman et al 2018 spot model. Defaults to N=1")
ap.add_argument("-min_period", required=False, type=float, default=0.1, help="Minimum period limit for the Lomb Scargle search, units=days. Defaults to 0.1 days.")
ap.add_argument("-max_period", required=False, type=float, default=100, help="Maximum period limit for the Lomb Scargle search, units=days. Defaults to 100 days.")
ap.add_argument("-ls_resolution", required=False, type=int, default=1000000, help="Resolution of the frequency array for the Lomb Scargle search. Defaults to 1000000.")
ap.add_argument("-exclude_dates", nargs='*',type=str,help="Dates to exclude, if any. Write the dates separated by a space (e.g., 19950119 19901023)")
ap.add_argument("-exclude_comps", nargs='*',type=int,help="Comparison stars to exclude, if any. Write the comp/ref number assignments ONLY, separated by a space (e.g., 2 5 7 if you want to exclude references/comps R2,R5, and R7.) ")
ap.add_argument("-ap_radius", default='optimal',help='Aperture radius (in pixels) of data to be loaded. Write as an integer (e.g., 8 if you want to use the circular_fixed_ap_phot_8.csv files for all loaded dates of data). Defaults to the optimal radius. ')
ap.add_argument('-restore', default=False, help='If True, the code will try to restore the global photometry for the target with the selected aperture size.')
ap.add_argument('-median_filter_w', default=0, help='The width of the median filter (in days) to be applied to the data set. If 0, no filtering will be done.')
ap.add_argument('-regression', default=False, help='If True, the code will do regression on target flux on each night against target x/y pixel positions, average FWHM, and airmass.')
ap.add_argument('-airmass_limit', required=False, default=2, help='Maximum airmass of a cadence that will be retained in the analysis.')
ap.add_argument('-duration_limit', required=False, default=0.5, help='Minimum duration (in hours) that a night of observations must have to be retained in the analysis. Note that this limit refers to the total exposure time on a given night, the data do not have to be continuous.')
ap.add_argument('-fwhm_limit', required=False, default=3.5, help='Maximum FWHM (in arcseconds) that a cadence must have to be retained in the analysis.')

args = ap.parse_args()

target = args.target
ffname = args.ffname
N_sinusoids = args.N
min_period = args.min_period
max_period = args.max_period
ls_resolution = args.ls_resolution
exclude_dates = np.array(args.exclude_dates)
exclude_comps = np.array(args.exclude_comps)
ap_radius = args.ap_radius
restore = bool(args.restore)
regression = bool(args.regression)
airmass_limit = float(args.airmass_limit)
duration_limit = float(args.duration_limit) 
fwhm_limit = float(args.fwhm_limit)

median_filter_w = float(args.median_filter_w)

basepath = '/data/tierras/'
lcpath = os.path.join(basepath,'lightcurves')
lcfolderlist = np.sort(glob.glob(lcpath+"/**/"+target))
lcdatelist = [lcfolderlist[ind].split("/")[4] for ind in range(len(lcfolderlist))] 

# start the timer (to see how long the code takes)
start = time()

#load the reference star df 
targpath = os.path.join(basepath, 'targets')
# refdf_path = os.path.join(targpath, f'{target}/{target}_target_and_ref_stars.csv')
# refdf = pd.read_csv(refdf_path)


# load the list of comparison stars to use. Alt method: use same strategy as in ld.calc_rel_flux
compfname = os.path.join(lcfolderlist[0],ffname,"night_weights.csv")
compfname_df = pd.read_csv(compfname)
complist = compfname_df['Reference'].to_numpy()
complist = np.array([int(s.split()[-1]) for s in complist])
# complist = np.arange(len(refdf)-1)+1

global_flux_path = targpath+f'/{target}/global_uncorrected_circular_fixed_ap_phot_{ap_radius}.csv'
if restore and os.path.exists(global_flux_path):
    res = pickle.load(open(global_flux_path,'rb'))
    full_bjd = res['BJD']
    bjd_save = res['BJD List']
    full_flux = res['Target Flux']
    full_err = res['Target Flux Error']
    full_reg = res['Regressor Fluxes']
    full_reg_err = res['Regressor Flux Errors']
    full_flux_div_expt = res['Target Flux Div Exptime']
    full_err_div_expt = res['Target Flux Error Div Exptime']
    full_relflux = res['Target Relative Flux']
    full_exptime = res['Exposure Time']
    full_sky = res['Sky Background']
    full_x_pos = res['X']
    full_y_pos = res['Y']
    full_airmass = res['Airmass']
    full_fwhm = res['FWHM']
else:
    # Load raw target and reference fluxes into global lists
    full_bjd, bjd_save, full_flux, full_err, full_reg, full_reg_err, full_flux_div_expt, full_err_div_expt, full_relflux, full_exptime, full_sky, full_x_pos, full_y_pos, full_airmass, full_fwhm = ld.make_global_lists(lcpath,target,ffname,exclude_dates,complist,duration_limit,ap_radius=args.ap_radius, )
    
    # Write the global lists to global_flux_path 
    global_flux_dict = {'BJD':full_bjd, 'BJD List':bjd_save, 'Target Flux': full_flux, 'Target Flux Error':full_err, 'Regressor Fluxes':full_reg, 'Regressor Flux Errors':full_reg_err, 'Target Flux Div Exptime':full_flux_div_expt, 'Target Flux Error Div Exptime':full_err_div_expt, 'Target Relative Flux':full_relflux, 'Exposure Time':full_exptime, 'Sky Background':full_sky, 'X':full_x_pos, 'Y':full_y_pos, 'Airmass':full_airmass, 'FWHM':full_fwhm}
    #pickle.dump(global_flux_dict, open(global_flux_path,'wb'))


# Update the mask with any comp stars identified as outliers
mask = ~np.isin(complist,exclude_comps)

# Drop flagged comp stars from the full regressor array
full_reg_loop = copy.deepcopy(full_reg)[mask]
full_reg_err_loop = copy.deepcopy(full_reg_err)[mask]

# mask bad data and use comps to calculate frame-by-frame magnitude zero points
x, y, err, masked_reg, masked_reg_err, cs, c_unc, exp_times, skies, weights, x_pos, y_pos, airmass, fwhm, bjd_save = mearth_style_pat_weighted(full_bjd, full_flux_div_expt, full_err_div_expt, full_reg_loop, full_reg_err_loop, full_exptime, full_sky, full_x_pos, full_y_pos, full_airmass, full_fwhm, bjd_save, duration_limit, airmass_limit, fwhm_limit) 

plt.figure()
plt.hist(airmass)
plt.xlabel('Airmass')

plt.figure()
plt.hist(fwhm)
plt.xlabel('FWHM')

# Generate the corrected flux figure
resfig, resax, binned_fluxes, masked_reg_corr = reference_flux_correction(x, bjd_save, masked_reg, masked_reg_err, cs, c_unc, complist[mask], weights[mask], airmass, fwhm) 

reg_corr_stds = np.nanstd(masked_reg_corr,axis=1)[np.where(weights!=0)[0]]
print(np.nanmedian(reg_corr_stds))

# Optionally use a median filter to remove long-term trends from the corrected target flux
if median_filter_w != 0:
    print(f'Median filtering target flux with a filter width of {median_filter_w} days.')
    x_filter, y_filter = median_filter_uneven(x, y, median_filter_w)
    mu = np.median(y)

    plt.figure()
    plt.plot(x,y)
    plt.plot(x,y_filter)
    plt.plot(x,mu*y/y_filter)
    breakpoint()
    y =  mu*y/(y_filter)

if regression:
    print('Regressing target flux against x/y position, FWHM, and airmass.')
    # loop over each night and do regression on target flux against x/y position, fwhm, and airmass

    for ii in range(len(bjd_save)):
        use_inds = np.where((x>=bjd_save[ii][0])&(x<=bjd_save[ii][-1]))[0]
        ancillary_dict = {'Airmass':airmass[use_inds],'FWHM':fwhm[use_inds]}
        reg_flux, intercept, coeffs, ancillary_dict_return = regress(y[use_inds], ancillary_dict, pval_threshold=1e-3, verbose=True)
        if intercept != 0:
            # plt.figure()
            # plt.plot(x[use_inds], y[use_inds])
            # plt.plot(x[use_inds], reg_flux*np.mean(y[use_inds]))
            y[use_inds] = reg_flux*np.mean(y[use_inds]) # update flux with the regressed values
            # breakpoint()
    
flare_mask = flare_masker(bjd_save, x, y)

# save the flare data in case we want it later
flare_x = x[~flare_mask]
flare_y = y[~flare_mask]
flare_err = err[~flare_mask]

# mask out the flares
x = x[flare_mask]
y = y[flare_mask]
err = err[flare_mask]

# y_reg = regress(y, {'Airmass':airmass[flare_mask], 'FWHM':fwhm[flare_mask]})[0]
# y = y_reg * np.median(y)

# ref_dists = (np.array((refdf['x'][0]-refdf['x'][1:])**2+(refdf['y'][0]-refdf['y'][1:])**2)**(0.5))[mask]
# bp_rps = np.array(refdf['bp_rp'][1:][mask])
# G_mags = np.array(refdf['G'][1:][mask])

# detector_half = np.zeros(len(mask), dtype='int')
# for i in range(len(ref_dists)):
#     if refdf['y'][i+1] > 1023:
#         detector_half[i] = 1
# detector_half = detector_half[mask]

# plt.figure()
colors = ['tab:blue', 'tab:orange']
stddevs = np.std(binned_fluxes, axis=1)*1e3

print(f'Median night-to-night stddev of weighted ref stars: {np.median(stddevs[np.where(weights != 0)[0]]):.2f} ppt')

fwhm_correlations = np.zeros(len(masked_reg_corr))
fwhm_pvals = np.zeros_like(fwhm_correlations)
airmass_correlations = np.zeros_like(fwhm_correlations)
airmass_pvals = np.zeros_like(fwhm_correlations)
for i in range(len(masked_reg_corr)):
    v, l, h = sigmaclip(masked_reg_corr[i,:], 3, 3)
    use_inds = np.where((masked_reg_corr[i]>l)&(masked_reg_corr[i]<h))[0]
    corr, pval = pearsonr(masked_reg_corr[i,use_inds], fwhm[use_inds])
    fwhm_correlations[i] = corr
    fwhm_pvals[i] = pval
    corr, pval = pearsonr(masked_reg_corr[i,use_inds], airmass[use_inds])
    airmass_correlations[i] = corr
    airmass_pvals[i] = pval

    # ancillary_dict = {'Airmass':airmass[use_inds],'FWHM':fwhm[use_inds]}
    # reg_flux, intercept, coeffs, ancillary_dict_return = regress(masked_reg_corr[i,use_inds], ancillary_dict, pval_threshold=1e-3, verbose=True)
    

# fig, ax = plt.subplots(2,2,figsize=(8,7))
# ax[0,0].scatter(refdf['x'][1:],airmass_correlations)
# ax[1,0].scatter(refdf['x'][1:],fwhm_correlations)

# ax[0,0].set_ylabel('Airmass correlation')
# ax[1,0].set_ylabel('FWHM correlation')
# ax[1,0].set_xlabel('X position')

# ax[0,1].scatter(refdf['y'][1:],airmass_correlations, color='tab:orange')
# ax[1,1].set_xlabel('Y position')
# ax[1,1].scatter(refdf['y'][1:],fwhm_correlations, color='tab:orange')
# breakpoint()


# for i in range(len(ref_dists)):
#     plt.scatter(ref_dists[i], stddevs[i], color=colors[detector_half[i]])

# plt.xlabel('Ref. Dist. from Targ.', fontsize=16)
# plt.ylabel('$\sigma$ (ppt)', fontsize=16)
# plt.tick_params(labelsize=14)
# plt.tight_layout()

# plt.figure()
# for i in range(len(ref_dists)):
#     plt.scatter(bp_rps[i], stddevs[i], color=colors[detector_half[i]])

# plt.xlabel('B$_p$-R$_p$', fontsize=16)
# plt.ylabel('$\sigma$ (ppt)', fontsize=16)
# plt.tick_params(labelsize=14)
# plt.tight_layout()

# # Note: this plotting the stddev of the corrected ref stars across their whole time series, NOT the stddev of the medians.
# plt.figure()
# ref_source_minus_sky = np.nanmedian(masked_reg, axis=1)
# ref_stddevs = np.nanstd(masked_reg_corr,axis=1)
# for i in range(len(ref_source_minus_sky)):
#     plt.scatter(ref_source_minus_sky[i], ref_stddevs[i], color=colors[detector_half[i]])
# plt.xlabel('Median flux (ADU)', fontsize=16)
# plt.ylabel('$\sigma$ (ppt)', fontsize=16)
# plt.tick_params(labelsize=14)
# plt.tight_layout()

# plt.figure()
# color_inds = np.zeros(len(weights),dtype='int')
# color_inds[np.where(weights==0)[0]] = 1
# # plt.scatter(G_mags, stddevs, color=np.array(colors)[color_inds])
# # plt.axvline(refdf['G'][0],color='tab:red')

# plt.xlabel('G mag', fontsize=16)
# plt.ylabel('$\sigma$ (ppt)', fontsize=16)
# plt.tick_params(labelsize=14)
# plt.tight_layout()
# breakpoint()

# plt.figure()
# for i in range(len(binned_fluxes)):
#     plt.hist(binned_fluxes[i])
#     plt.xlabel('Median flux', fontsize=16)
#     plt.ylabel('N$_{nights}$', fontsize=16)
# #plt.title(f'Ref {complist[i]}')
# #pdb.set_trace()

GAIN = 5.9
effective_exp_time = np.median(exp_times) # Use the median exposure time to estimate the expected noise level across the data set
effective_sky = np.median(skies*GAIN) # Do the same thing for sky background. 'skies' are already a rate in ADU/s (see make_global_lists), just need to multiply by GAIN
fig, ax = noise_component_plot(ap_rad=12, exp_time=effective_exp_time, sky_rate=effective_sky, airmass=1.4)
#avg_source_fluxes = np.nanmedian(GAIN*masked_reg*10**(cs/(-2.5)),axis=1)
#source_stddevs = np.nanstd(masked_reg_corr, axis=1)*1e6
#ax.plot(avg_source_fluxes, source_stddevs, 'k.')
for ii in range(len(masked_reg_corr)):
    for jj in range(len(bjd_save)):
        use_bjds = np.array(bjd_save[jj])
        inds = np.where((x > np.min(use_bjds)) & (x < np.max(use_bjds)))[0]
        avg_flux = np.nanmedian(GAIN*masked_reg[ii, inds]*10**(cs[inds]/(-2.5)))
        stddev = np.nanstd(masked_reg_corr[ii, inds]*1e6)
        ax.plot(avg_flux, stddev, 'k.')

################################################################################################
###### TO DO: fix this plotting loop to be readable (for large N it becomes quite unruly) ######
# plot the data night-by-night
# initialize the pplot
# try:
#     N = len(lcfolderlist) - len(exclude_dates) #N can be set to lower value for testing/inspection on individual nights. N should be = len(lcfolderlist)-len(exclude_dates) 
# except TypeError:
#     N = len(lcfolderlist)
N = len(bjd_save)

fig, ax = plt.subplots(2, N, sharey='row', sharex=True, figsize=(30, 4))
target_binned_fluxes = np.zeros(N)
colors = []
cmap = plt.get_cmap('viridis')
colors.extend(cmap((255*np.arange(len(x))/len(x)).astype('int')))
colors = np.array(colors)
for ii in range(N):
   # get the indices corresponding to a given night
   use_bjds = np.array(bjd_save[ii])
   inds = np.where((x > np.min(use_bjds)) & (x < np.max(use_bjds)))[0]
   flare_inds = np.where((flare_x>min(use_bjds))&(flare_x<max(use_bjds)))[0]
   if len(inds) == 0:  # if the entire night was masked due to bad weather, don't plot anything
       continue
   else:
       # identify and plot the night of data
       bjd_plot = x[inds]
       flux_plot = y[inds]
       err_plot = err[inds]
       color_inds = colors[inds]

       #markers, caps, bars = ax[0, ii].errorbar((bjd_plot-np.min(bjd_plot))*24., flux_plot, yerr=err_plot, marker='.', ls='', color=cmap(color_ind), alpha=0.2)
       ax[0, ii].scatter((bjd_plot-np.min(bjd_plot))*24., flux_plot, marker='.', color=color_inds, alpha=0.2)
       # ax[0,ii].scatter((flare_x[flare_inds] - np.min(bjd_plot))*24., flare_y[flare_inds], marker='.', color='r', alpha=0.2)

       # Generate the date label for the top of the plot 
       ut_date = Time(bjd_plot[-1],format='jd',scale='tdb').iso.split(' ')[0]
       date_label = ut_date.split('-')[1].lstrip('0')+'/'+ut_date.split('-')[2].lstrip('0')+'/'+ut_date[2:4]
       ax[0, ii].set_title(date_label, rotation=45, fontsize=8)
       #[bar.set_alpha(0.05) for bar in bars]
       target_binned_fluxes[ii] = np.median(flux_plot)

target_binned_fluxes = target_binned_fluxes[target_binned_fluxes != 0] 
target_binned_fluxes /= np.median(target_binned_fluxes) # Calculate the median normalized target fluxes on each night after the zero-points have been applied

# # Add the median target flux stddevs to some plots from earlier
# plt.figure(2) 
# plt.plot(0, np.std(target_binned_fluxes)*1e3, 'r*', ms=10)

# plt.figure(3)
# plt.plot(refdf['bp_rp'][0], np.std(target_binned_fluxes)*1e3, 'r*', ms=10)

# plt.figure(5)
# plt.plot(refdf['G'][0], np.std(target_binned_fluxes)*1e3, 'r*', ms=10)

#plt.figure(8)
# format the plot
fig.text(0.5, 0.01, 'hours since start of night', ha='center')
ax[0, 0].set_ylabel('corrected flux')
ax[0, 0].set_ylim(np.nanpercentile(y, 1), np.nanpercentile(y, 99))  # don't let outliers wreck the y-axis scale
fig.tight_layout()
plt.subplots_adjust(wspace=0, hspace=0)
################################################################################################

# convert relative flux to ppt.
mu = np.nanmedian(y)
y = (y / mu - 1) * 1e3
err = (err/mu) * 1e3


# ################# PERIOD FINDING TREATMENT ############# 

# do the Lomb Scargle
freq = np.linspace(1./max_period, 1./min_period, ls_resolution)
ls = LombScargle(x, y, dy=err)
power = ls.power(freq)
peak = 1./freq[np.argmax(power)]

#ls = LombScargle(x, y, dy=err)
#freq, power = ls.autopower(minimum_frequency=1./max_period, maximum_frequency=1./min_period)
#peak = 1./freq[np.argmax(power)]

# make the window function
ls_window = LombScargle(x, np.ones_like(x), fit_mean=False, center_data=False)
wf, wp = ls_window.autopower(minimum_frequency=1./max_period, maximum_frequency=1./min_period)

# report the periodogram peak
print("LS Period:", peak)

# make a pretty periodogram
fig_ls, ax_ls = plt.subplots(2,1, sharex=True)

ax_ls[0].plot(1./freq, power, color='orange', label='data', zorder=10)
ax_ls[1].plot(1./wf, wp, color='k', label='window')
ax_ls[0].set_xlim((1./freq).min(), (1./freq).max())
ax_ls[0].axvline(peak, color='k', linestyle='dashed')
ax_ls[1].axvline(peak, color='orange', linestyle='dashed')
ax_ls[0].set_ylabel("power")
ax_ls[1].set_ylabel("power")
ax_ls[1].set_xlabel("period [d]")
ax_ls[0].semilogx()


# this variable is called period1 because I've adapted this from another code I wrote that fits multiple periods
# to blended TESS light curves. I've removed the second period for simplicity here since Tierras doesn't have the same
# pixel scale issues, but that functionality can be reimplemented if needed
period1 = peak # if you want to override the LS period, you can set this to your desired value instead of "peak"

# fit Hartman model and iteratively clip outliers from fit 
model, map_soln, extras = sigma_clip(x,y,err,period1,N_sinusoids) 

# print the maximum a posteriori (MAP) results
for var in map_soln.keys():
    print(var, map_soln[var])

t_lc_pred = np.linspace(x.min(), x.max(), 100000)  # times at which we're going to plot

# get the light curve associated with the maximum a posteriori model
# QUESTION: why is map_soln['mean_lc'] subtracted off the models here (and not added to the spota calculation)? The full spot model should be mean_lc + 1e3*spota(t)
# with model:
#     gp_pred = (pmx.eval_in_model(extras["gp_lc_pred"], map_soln) + map_soln["mean_lc"])
#     lc = (pmx.eval_in_model(extras["model_lc"](t_lc_pred), map_soln) - map_soln["mean_lc"])
#     lc_obs = (pmx.eval_in_model(extras["model_lc"](x), map_soln) - map_soln["mean_lc"])
#     spota = 1e3 * pmx.eval_in_model(extras["spota"](t_lc_pred), map_soln)

# Added by PT, I think this is what the model should actually be
with model:
    gp_pred = (pmx.eval_in_model(extras["gp_lc_pred"], map_soln) + map_soln["mean_lc"])
    lc = pmx.eval_in_model(extras["model_lc"](t_lc_pred), map_soln) 
    lc_obs = pmx.eval_in_model(extras["model_lc"](x), map_soln) 
    spota = 1e3 * pmx.eval_in_model(extras["spota"](t_lc_pred), map_soln) + map_soln['mean_lc']

# arrays to store residuals
all_res = []
all_res_bin = []

# overplot the fit on each night of data
for ii in range(N):
    bjds = bjd_save[ii]

    # get the indices corresponding to a given night
    use_bjds = np.array(bjd_save[ii])
    inds = np.where((x > np.min(use_bjds)) & (x < np.max(use_bjds)))[0]
    flare_inds = np.where((flare_x>min(use_bjds))&(flare_x<max(use_bjds)))[0]
    if len(inds) == 0:  # if the entire night was masked due to bad weather, don't plot anything
        continue
    else:
        # identify and plot the night of data
        bjd_plot = x[inds]
        flux_plot = y[inds]
        err_plot = err[inds]
        lc_plot = lc_obs[inds] 
        color_inds = colors[inds]

    xlim = ax[0, ii].get_xlim()  # remember the x-axis limits so we don't mess them up

    # plot the model fit
    ax[0, ii].plot((t_lc_pred - np.min(use_bjds))*24., (lc/1e3 + 1)*mu, color="r", lw=1, zorder=10)
    # add bins
    tbin = 20  # bin size in minutes
    xs_b, binned, e_binned = ep_bin((bjd_plot - np.min(bjd_plot))*24, (flux_plot/1e3+1)*mu, tbin/60.)
    _, binned_res, e_binned_res = ep_bin((bjd_plot - np.min(bjd_plot))*24, flux_plot-lc_plot, tbin/60.)
    marker, caps, bars = ax[0, ii].errorbar(xs_b, binned, yerr=e_binned, mfc='w', mec='k', mew=1, ecolor='k', fmt='.', alpha=0.7, zorder=5)
    [bar.set_alpha(0.3) for bar in bars]

    # plot the residuals
    #markers, caps, bars = ax[1, ii].errorbar((bjd_plot - np.min(bjd_plot))*24., flux_plot - lc_plot, yerr=err_plot, fmt='k.', alpha=0.2)
    ax[1,ii].scatter((bjd_plot - np.min(bjd_plot))*24., flux_plot - lc_plot, marker='.', color=color_inds, alpha=0.2)
    # ax[1,ii].scatter((flare_x[flare_inds] - np.min(bjd_plot))*24., flare_y[flare_inds] - lc_obs[flare_inds], marker='.', color='r', alpha=0.2)
    #[bar.set_alpha(0.05) for bar in bars]
    markers, caps, bars = ax[1, ii].errorbar(xs_b, binned_res, yerr=e_binned_res, mfc='w', mec='k', mew=1, ecolor='k', fmt='.', alpha=0.7, zorder=5)
    [bar.set_alpha(0.3) for bar in bars]
    ax[1, ii].axhline(0, linestyle='dashed', color='k')

    ax[0, ii].set_xlim(xlim)  # revert to original axis limits
    all_res.extend(flux_plot-lc_plot)  # keep track of residuals
    all_res_bin.extend(binned_res)  # keep track of binned residuals
    
    #resax[1,ii].plot((bjd_plot-np.min(bjd_plot))*24, flux_plot-lc_plot, color='k', marker='.', ls='')
    #breakpoint()

# report the time it took to run the code
print("Elapsed time:", np.round((time()-start)/60.,2), "min")

all_res = np.array(all_res)
all_res_bin = np.array(all_res_bin)

ax[1, 0].set_ylabel("O-C [ppt]")
ax[1, 0].set_ylim(np.nanpercentile(all_res, 1), np.nanpercentile(all_res, 99))  # don't let outliers wreck the y-axis scale

print("RMS model:", np.round(np.sqrt(np.nanmean(all_res**2))*1e3, 2), "ppm")
print("Binned RMS model:", np.round(np.sqrt(np.nanmean(all_res_bin**2))*1e3, 2), "ppm in", tbin, "minute bins")

#pdb.set_trace()

plt.show()

# subtract off first timestamp to make the x-axis less cluttered
min_t = np.min(extras["x"])
extras["x"] -= min_t
t_lc_pred -= min_t

# make plot of phased light curve
fig, ax1 = plt.subplots(1)
x_phased1 = (extras["x"] % period1) / period1
x_phased = (t_lc_pred % period1) / period1
y1 = extras['y']

# Color-code by x
colors_phase = []
inds_ = np.arange(len(extras['x']))
colors_phase.extend(cmap(((255*inds_)/len(inds_)).astype('int')))
colors_phase = np.array(colors_phase)

# bin the data into 100 bins that are evenly spaced in phase
bins = np.linspace(0, 1, 101)
sort1 = np.argsort(x_phased1)
x_phased1 = x_phased1[sort1]
sort = np.argsort(x_phased)
x_phased = x_phased[sort]
y1 = y1[sort1]
spota = spota[sort]
colors_phase = colors_phase[sort1]
binned1 = []
binned1_std = []
for ii in range(100):
    use_inds = np.where((x_phased1 < bins[ii + 1]) & (x_phased1 > bins[ii]))[0]
    binned1.append(np.nanmedian(y1[use_inds]))
    binned1_std.append(np.nanstd(y1[use_inds])/np.sqrt(len(use_inds)))

#pdb.set_trace()

# plot the phased data
#ax1.plot(x_phased1, y1, "k.", alpha=0.2)
ax1.scatter(x_phased1, y1, marker='.', color=colors_phase, alpha=0.2)
ax1.plot(x_phased, spota, 'r', lw=1, zorder=10)
ax1.errorbar(bins[:-1] + (bins[1] - bins[0]) / 2., binned1, binned1_std, marker='o', mfc='w', mec='k', mew=2, ecolor='k', alpha=0.7, ls='')

# I wanted this to adjust the y-axis to an appropriate scale automatically, but you may still need to play around with
# ylim_N depending on how noisy the data is.
ylim_N = 5
mean1 = np.nanmedian(spota)
max1 = np.max(spota)
diff1 = max1 - mean1
ax1.set_ylim(mean1 - ylim_N * diff1, mean1 + ylim_N * diff1)

# finish up the plot
ax1.set_ylabel("Tierras flux [ppt]")
ax1.set_xlabel("phase")
fig.tight_layout()
plt.show()
breakpoint()
