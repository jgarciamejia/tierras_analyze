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
from scipy.stats import sigmaclip

import pymc3 as pm 
import pymc3_ext as pmx 
from celerite2.theano import terms, GaussianProcess 

import test_load_data as ld
from test_mearth_style_for_tierras_pat import mearth_style, mearth_style_pat
from test_find_rotation_period import build_model, sigma_clip
from test_bin_lc import ep_bin 
from corrected_flux_plot import reference_flux_correction
from noise_calculation import noise_component_plot 

# Contributors (so far): JIrwin, EPass, JGarciaMejia, PTamburo.

# Deal with command line
ap = argparse.ArgumentParser()
ap.add_argument("-target", required=True, help="Name of observed target exactly as shown in raw FITS files.")       
ap.add_argument("-ffname", required=True, help="Name of folder in which to store reduced+flattened data. Convention is flatXXXX. XXXX=0000 means no flat was used.")
ap.add_argument("-N", required=False, type=int, default=1, help="Number of sinusoids to use in the Hartman et al 2018 spot model. Defaults to N=1")
ap.add_argument("-min_period", required=False, type=float, default=0.1, help="Minimum period limit for the Lomb Scargle search, units=days. Defaults to 0.1 days.")
ap.add_argument("-max_period", required=False, type=float, default=100, help="Maximum period limit for the Lomb Scargle search, units=days. Defaults to 100 days.")
ap.add_argument("-ls_resolution", required=False, type=int, default=100000, help="Resolution of the frequency array for the Lomb Scargle search. Defaults to 100000.")
ap.add_argument("-exclude_dates", nargs='*',type=str,help="Dates to exclude, if any. Write the dates separated by a space (e.g., 19950119 19901023)")
ap.add_argument("-exclude_comps", nargs='*',type=int,help="Comparison stars to exclude, if any. Write the comp/ref number assignments ONLY, separated by a space (e.g., 2 5 7 if you want to exclude references/comps R2,R5, and R7.) ")
ap.add_argument("-ap_radius", default='optimal',help='Aperture radius (in pixels) of data to be loaded. Write as an integer (e.g., 8 if you want to use the circular_fixed_ap_phot_8.csv files for all loaded dates of data). Defaults to the optimal radius. ')
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

basepath = '/data/tierras/'
lcpath = os.path.join(basepath,'lightcurves')
lcfolderlist = np.sort(glob.glob(lcpath+"/**/"+target))
lcdatelist = [lcfolderlist[ind].split("/")[4] for ind in range(len(lcfolderlist))] 

# start the timer (to see how long the code takes)
start = time()

#load the reference star df 
targpath = os.path.join(basepath, 'targets')
refdf_path = os.path.join(targpath, f'{target}/{target}_target_and_ref_stars.csv')
refdf = pd.read_csv(refdf_path)


# load the list of comparison stars to use. Alt method: use same strategy as in ld.calc_rel_flux
# compfname = os.path.join(lcfolderlist[0],ffname,"night_weights.csv")
# compfname_df = pd.read_csv(compfname)
# complist = compfname_df['Reference'].to_numpy()
# complist = np.array([int(s.split()[-1]) for s in complist])
complist = np.arange(len(refdf)-1)+1

# Load raw target and reference fluxes into global lists
full_bjd, bjd_save, full_flux, full_err, full_reg, full_reg_err, full_flux_div_expt, full_err_div_expt, full_relflux, full_exptime, full_sky = ld.make_global_lists(lcpath,target,ffname,exclude_dates,complist,ap_radius=args.ap_radius)


# Use the reference stars to calculate the zero-point offsets. 
# Measure the standard deviation of their median night-to-night fluxes after being corrected with the measured zero-points.
# Identify outliers and drop them. 
# Repeat the calcluation until no new outliers are found.
count = 0 
while True:
    # Update the mask with any comp stars identified as outliers
    mask = ~np.isin(complist,exclude_comps)

    # Drop flagged comp stars from the full regressor array
    full_reg_loop = copy.deepcopy(full_reg)[mask]
    full_reg_err_loop = copy.deepcopy(full_reg_err)[mask]

    # mask bad data and use comps to calculate frame-by-frame magnitude zero points
    x, y, err, masked_reg, masked_reg_err, cs, c_unc, exp_times, skies = mearth_style_pat(full_bjd, full_flux_div_expt, full_err_div_expt, full_reg_loop, full_reg_err_loop, full_exptime, full_sky) #TO DO: how to integrate weights into mearth_style?

    binned_fluxes = reference_flux_correction(x, y, masked_reg, masked_reg_err, cs, c_unc, complist[mask], plot=False) #Returns an n_comp_star x n_nights array of medians of corrected flux

    # ref_dists = (np.array((refdf['x'][0]-refdf['x'][1:])**2+(refdf['y'][0]-refdf['y'][1:])**2)**(0.5))[mask]
    # bp_rps = np.array(refdf['bp_rp'][1:][mask])
    # G_mags = np.array(refdf['G'][1:][mask])

    # detector_half = np.zeros(len(mask), dtype='int')
    # for i in range(len(ref_dists)):
    #     if refdf['y'][i+1] > 1023:
    #         detector_half[i] = 1
    # detector_half = detector_half[mask]

    # plt.figure()
    # colors = ['tab:blue', 'tab:orange']
    # stddevs = np.std(binned_fluxes, axis=1)*1e3
    # for i in range(len(ref_dists)):
    #     plt.scatter(ref_dists[i], stddevs[i], color=colors[detector_half[i]])
    # plt.xlabel('Ref. Dist. from Targ.', fontsize=16)
    # plt.ylabel('$\sigma$ (ppt)', fontsize=16)
    # plt.tick_params(labelsize=14)
    # plt.tight_layout()
    
    #Use the stddevs of the binned fluxes to determine which references to drop
    stddevs = np.std(binned_fluxes, axis=1)
    v, l, h = sigmaclip(stddevs, 3, 3)

    # If it's the first loop, instantiate the exclude_comps array by finding the indices of those that lie outside the range identified by sigmaclip
    if count == 0:
        exclude_comps = np.where((stddevs>h))[0] + 1 # Does sigma-clipping and also an absolute cut on the night-to-night stddevs of median fluxes: use the best 20%. 

    # If it's a subsequent loop, we append those indices to the already-existing list of bad comp stars
    else:
        exclude_comps = np.unique(np.sort(np.append(exclude_comps, np.where((stddevs>h))[0] + 1).astype('int')))

        # If the list of comp stars to be excluded matches the one from the previous loop, break
        # Alternatively, break if the number of iterations reaches the length of the number of stars in the complist. This protects against getting into an infinite loop.
        if (np.array_equal(exclude_comps, exclude_comps_old)) or (count > len(complist)):
            print('Converged!')
            break
    
    print(f'Excluding references {exclude_comps}')
    exclude_comps_old = copy.deepcopy(exclude_comps)
    count += 1

# complist = compfname_df['Reference'].to_numpy()
# complist = np.array([int(s.split()[-1]) for s in complist])
complist = np.arange(len(refdf)-1)+1

mask = ~np.isin(complist,exclude_comps)

# Generate the corrected flux figure
resfig, resax, binned_fluxes, masked_reg_corr = reference_flux_correction(x, y, masked_reg, masked_reg_err, cs, c_unc, complist[mask], plot=True) 

breakpoint()
ref_dists = (np.array((refdf['x'][0]-refdf['x'][1:])**2+(refdf['y'][0]-refdf['y'][1:])**2)**(0.5))[mask]
bp_rps = np.array(refdf['bp_rp'][1:][mask])
G_mags = np.array(refdf['G'][1:][mask])

detector_half = np.zeros(len(mask), dtype='int')
for i in range(len(ref_dists)):
    if refdf['y'][i+1] > 1023:
        detector_half[i] = 1
detector_half = detector_half[mask]

plt.figure()
colors = ['tab:blue', 'tab:orange']
stddevs = np.std(binned_fluxes, axis=1)*1e3
for i in range(len(ref_dists)):
    plt.scatter(ref_dists[i], stddevs[i], color=colors[detector_half[i]])

plt.xlabel('Ref. Dist. from Targ.', fontsize=16)
plt.ylabel('$\sigma$ (ppt)', fontsize=16)
plt.tick_params(labelsize=14)
plt.tight_layout()

plt.figure()
for i in range(len(ref_dists)):
    plt.scatter(bp_rps[i], stddevs[i], color=colors[detector_half[i]])

plt.xlabel('B$_p$-R$_p$', fontsize=16)
plt.ylabel('$\sigma$ (ppt)', fontsize=16)
plt.tick_params(labelsize=14)
plt.tight_layout()

# Note: this plotting the stddev of the corrected ref stars across their whole time series, NOT the stddev of the medians.
plt.figure()
ref_source_minus_sky = np.nanmedian(masked_reg, axis=1)
ref_stddevs = np.nanstd(masked_reg_corr,axis=1)
for i in range(len(ref_source_minus_sky)):
    plt.scatter(ref_source_minus_sky[i], ref_stddevs[i], color=colors[detector_half[i]])
plt.xlabel('Median flux (ADU)', fontsize=16)
plt.ylabel('$\sigma$ (ppt)', fontsize=16)
plt.tick_params(labelsize=14)
plt.tight_layout()

plt.figure()
for i in range(len(ref_dists)):
    plt.scatter(G_mags[i], stddevs[i], color=colors[detector_half[i]])

plt.xlabel('G mag', fontsize=16)
plt.ylabel('$\sigma$ (ppt)', fontsize=16)
plt.tick_params(labelsize=14)
plt.tight_layout()

plt.figure()
for i in range(len(binned_fluxes)):
    plt.hist(binned_fluxes[i])
    plt.xlabel('Median flux', fontsize=16)
    plt.ylabel('N$_{nights}$', fontsize=16)
#plt.title(f'Ref {complist[i]}')
#pdb.set_trace()

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

       # Generate the date label for the top of the plot 
       ut_date = Time(bjd_plot[-1],format='jd',scale='tdb').iso.split(' ')[0]
       date_label = ut_date.split('-')[1].lstrip('0')+'/'+ut_date.split('-')[2].lstrip('0')+'/'+ut_date[2:4]
       ax[0, ii].set_title(date_label, rotation=45, fontsize=8)
       #[bar.set_alpha(0.05) for bar in bars]
       target_binned_fluxes[ii] = np.median(flux_plot)

target_binned_fluxes = target_binned_fluxes[target_binned_fluxes != 0] 
target_binned_fluxes /= np.median(target_binned_fluxes) # Calculate the median normalized target fluxes on each night after the zero-points have been applied

# Add the median target flux stddevs to some plots from earlier
plt.figure(2) 
plt.plot(0, np.std(target_binned_fluxes)*1e3, 'r*', ms=10)

plt.figure(3)
plt.plot(refdf['bp_rp'][0], np.std(target_binned_fluxes)*1e3, 'r*', ms=10)

plt.figure(5)
plt.plot(refdf['G'][0], np.std(target_binned_fluxes)*1e3, 'r*', ms=10)

plt.figure(8)
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

t_lc_pred = np.linspace(x.min(), x.max(), 10000)  # times at which we're going to plot

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
for ii in range(100):
    use_inds = np.where((x_phased1 < bins[ii + 1]) & (x_phased1 > bins[ii]))[0]
    binned1.append(np.nanmedian(y1[use_inds]))

#pdb.set_trace()

# plot the phased data
#ax1.plot(x_phased1, y1, "k.", alpha=0.2)
ax1.scatter(x_phased1, y1, marker='.', color=colors_phase, alpha=0.2)
ax1.plot(x_phased, spota, 'r', lw=1, zorder=10)
ax1.plot(bins[:-1] + (bins[1] - bins[0]) / 2., binned1, marker='o', mfc='w', mec='k', mew=2,  alpha=0.7, ls='')

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
