import os
import sys
import pdb
import glob
import argparse 
import importlib
import numpy as np
import pandas as pd
from time import time
import matplotlib.pylab as plt
from astropy.timeseries import LombScargle

import pymc3 as pm 
import pymc3_ext as pmx 
from celerite2.theano import terms, GaussianProcess 

import test_load_data as ld
from test_mearth_style_for_tierras import mearth_style
from test_find_rotation_period import build_model, sigma_clip
from test_bin_lc import ep_bin 

# Contributors (so far): JIrwin, EPass, JGarciaMejia.

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
args = ap.parse_args()

target = args.target
ffname = args.ffname
N_sinusoids = args.N
min_period = args.min_period
max_period = args.max_period
ls_resolution = args.ls_resolution
exclude_dates = np.array(args.exclude_dates)
exclude_comps = np.array(args.exclude_comps)

basepath = '/data/tierras/'
lcpath = os.path.join(basepath,'lightcurves')
lcfolderlist = np.sort(glob.glob(lcpath+"/**/"+target))
lcdatelist = [lcfolderlist[ind].split("/")[4] for ind in range(len(lcfolderlist))] 

# start the timer (to see how long the code takes)
start = time()

# arrays to hold the full dataset
full_bjd = []
full_flux = []
full_err = []
full_reg = None

# array to hold individual nights
bjd_save = []

# load the list of comparison stars to use. Alt method: use same strategy as in ld.calc_rel_flux
compfname = os.path.join(lcfolderlist[0],ffname,"night_weights.csv")
compfname_df = pd.read_csv(compfname)
complist = compfname_df['Reference'].to_numpy()
complist = np.array([int(s.split()[-1]) for s in complist])

pdb.set_trace()

mask = ~np.isin(complist,exclude_comps)
complist = complist[mask]

# loop: load raw target and ref fluxes into global lists
for ii,lcfolder in enumerate(lcfolderlist):

    print("Processing", lcdatelist[ii])

    # if date excluded, skip
    if np.any(exclude_dates == lcdatelist[ii]):
        print ("{} :  Excluded".format(lcdatelist[ii]))
        continue
    
    # read the .csv file
    try:
        df, optimal_lc = ld.return_dataframe_onedate(lcpath,target,lcdatelist[ii],ffname)
    except TypeError:
        continue

    bjds = df['BJD TDB'].to_numpy()
    flux = df['Target Source-Sky ADU']
    err = df['Target Source-Sky Error ADU']
    expt = df['Exposure Time']

    # get the comparison fluxes.
    comps = {}
    for comp_num in complist:
        try:
            comps[comp_num] = df['Ref '+str(comp_num)+' Source-Sky ADU'] / expt  # divide by exposure time since it can vary between nights
        except:
            print("Error with comp", str(comp_num))
            continue

    # make a list of all the comps
    regressors = []
    for key in comps.keys():
        regressors.append(comps[key])
    regressors = np.array(regressors)

    # add this night of data to the full data set
    full_bjd.extend(bjds)
    full_flux.extend(flux/expt)
    full_err.extend(err/expt)
    bjd_save.append(bjds)

    if full_reg is None:
        full_reg = regressors
    else:
        full_reg = np.concatenate((full_reg, regressors), axis=1) 

# convert from lists to arrays
full_bjd = np.array(full_bjd)
full_flux = np.array(full_flux)
full_err = np.array(full_err)

# mask bad data and use comps to calculate frame-by-frame magnitude zero points
x, y, err = mearth_style(full_bjd, full_flux, full_err, full_reg) #TO DO: how to integrate weights into mearth_style?

#pdb.set_trace()

################################################################################################
###### TO DO: fix this plotting loop to be readable (for large N it becomes quite unruly) ######
# plot the data night-by-night
# initialize the pplot
try:
    N = len(lcfolderlist) - len(exclude_dates) #N can be set to lower value for testing/inspection on individual nights. N should be = len(lcfolderlist)-len(exclude_dates) 
except TypeError:
    N = len(lcfolderlist)

fig, ax = plt.subplots(2, N, sharey='row', sharex=True, figsize=(30, 4))

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
       markers, caps, bars = ax[0, ii].errorbar((bjd_plot-np.min(bjd_plot))*24., flux_plot, yerr=err_plot, fmt='k.', alpha=0.2)
       [bar.set_alpha(0.05) for bar in bars]

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
with model:
    gp_pred = (pmx.eval_in_model(extras["gp_lc_pred"], map_soln) + map_soln["mean_lc"])
    lc = (pmx.eval_in_model(extras["model_lc"](t_lc_pred), map_soln) - map_soln["mean_lc"])
    lc_obs = (pmx.eval_in_model(extras["model_lc"](x), map_soln) - map_soln["mean_lc"])
    spota = 1e3 * pmx.eval_in_model(extras["spota"](t_lc_pred), map_soln)

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

    xlim = ax[0, ii].get_xlim()  # remember the x-axis limits so we don't mess them up

    # plot the model fit
    ax[0, ii].plot((t_lc_pred - np.min(use_bjds))*24., (lc/1e3 + 1)*mu, color="C2", lw=1, zorder=10)

    # add bins
    tbin = 20  # bin size in minutes
    xs_b, binned, e_binned = ep_bin((bjd_plot - np.min(bjd_plot))*24, (flux_plot/1e3+1)*mu, tbin/60.)
    _, binned_res, e_binned_res = ep_bin((bjd_plot - np.min(bjd_plot))*24, flux_plot-lc_plot, tbin/60.)
    marker, caps, bars = ax[0, ii].errorbar(xs_b, binned, yerr=e_binned, color='purple', fmt='.', alpha=0.5, zorder=5)
    [bar.set_alpha(0.3) for bar in bars]

    # plot the residuals
    markers, caps, bars = ax[1, ii].errorbar((bjd_plot - np.min(bjd_plot))*24., flux_plot - lc_plot, yerr=err_plot, fmt='k.', alpha=0.2)
    [bar.set_alpha(0.05) for bar in bars]
    markers, caps, bars = ax[1, ii].errorbar(xs_b, binned_res, yerr=e_binned_res, color='purple', fmt='.', alpha=0.5, zorder=5)
    [bar.set_alpha(0.3) for bar in bars]
    ax[1, ii].axhline(0, linestyle='dashed', color='k')

    ax[0, ii].set_xlim(xlim)  # revert to original axis limits
    all_res.extend(flux_plot-lc_plot)  # keep track of residuals
    all_res_bin.extend(binned_res)  # keep track of binned residuals


# report the time it took to run the code
print("Elapsed time:", np.round((time()-start)/60.,2), "min")

all_res = np.array(all_res)
all_res_bin = np.array(all_res_bin)

ax[1, 0].set_ylabel("O-C [ppt]")
ax[1, 0].set_ylim(np.nanpercentile(all_res, 1), np.nanpercentile(all_res, 99))  # don't let outliers wreck the y-axis scale

print("RMS model:", np.round(np.sqrt(np.nanmedian(all_res**2))*1e3, 2), "ppm")
print("Binned RMS model:", np.round(np.sqrt(np.nanmedian(all_res_bin**2))*1e3, 2), "ppm in", tbin, "minute bins")

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

# bin the data into 100 bins that are evenly spaced in phase
bins = np.linspace(0, 1, 101)
sort1 = np.argsort(x_phased1)
x_phased1 = x_phased1[sort1]
sort = np.argsort(x_phased)
x_phased = x_phased[sort]
y1 = y1[sort1]
spota = spota[sort]
binned1 = []
for ii in range(100):
    use_inds = np.where((x_phased1 < bins[ii + 1]) & (x_phased1 > bins[ii]))[0]
    binned1.append(np.nanmedian(y1[use_inds]))

#pdb.set_trace()

# plot the phased data
ax1.plot(x_phased1, y1, "k.", alpha=0.2)
ax1.plot(x_phased, spota, 'r', lw=1, zorder=10)
ax1.plot(bins[:-1] + (bins[1] - bins[0]) / 2., binned1, 'go', alpha=0.5)

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
