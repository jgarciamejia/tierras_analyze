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
from test_bin_lc import ep_bin 

# Contributors (so far): JIrwin, EPass, JGarciaMejia.

# Deal with command line
ap = argparse.ArgumentParser()
ap.add_argument("-target", required=True, help="Name of observed target exactly as shown in raw FITS files.")       
ap.add_argument("-ffname", required=True, help="Name of folder in which to store reduced+flattened data. Convention is flatXXXX. XXXX=0000 means no flat was used.")
ap.add_argument("-exclude_dates", nargs='*',type=str,help="Dates to exclude, if any. Write the dates separated by a space (e.g., 19950119 19901023)")
ap.add_argument("-ref_as_target", required=True, type=int, help="Reference star/comp star/regressor to load as the target.")
args = ap.parse_args()

target = args.target
ffname = args.ffname
exclude_dates = np.array(args.exclude_dates)
ref_as_target = args.ref_as_target

basepath = '/data/tierras/'
lcpath = os.path.join(basepath,'lightcurves')
lcfolderlist = np.sort(glob.glob(lcpath+"/**/"+target))
lcdatelist = [lcfolderlist[ind].split("/")[4] for ind in range(len(lcfolderlist))] 

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
complist = complist[np.argwhere(complist != ref_as_target)].T[0] #exclude reference used as target from list of comps


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
    flux = df['Ref '+str(ref_as_target)+' Source-Sky ADU']
    err = df['Ref '+str(ref_as_target)+' Source-Sky Error ADU']
    #flux = df['Target Source-Sky ADU']
    #err = df['Target Source-Sky Error ADU']
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

# convert relative flux to ppt.
mu = np.nanmedian(y)
y = (y / mu - 1) * 1e3
err = (err/mu) * 1e3

################################################################################################
###### TO DO: fix this plotting loop to be readable (for large N it becomes quite unruly) ######
# plot the data night-by-night
# initialize the pplot
try:
    N = len(lcfolderlist) - len(exclude_dates) #N can be set to lower value for testing/inspection on individual nights. N should be = len(lcfolderlist)-len(exclude_dates) 
except TypeError:
    N = len(lcfolderlist)

medians_per_night = []
binned_medians_per_night = []

fig, ax = plt.subplots(1, N, sharey='row', sharex=True, figsize=(30, 4))

# loop through each night to 
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
		markers, caps, bars = ax[ii].errorbar((bjd_plot-np.min(bjd_plot))*24., flux_plot, yerr=err_plot, fmt='k.', alpha=0.2)
		[bar.set_alpha(0.05) for bar in bars]

		# add bins
		tbin = 20  # bin size in minutes
		xs_b, binned, e_binned = ep_bin((bjd_plot - np.min(bjd_plot))*24, flux_plot, tbin/60.)
		marker, caps, bars = ax[ii].errorbar(xs_b, binned, yerr=e_binned, color='purple', fmt='.', alpha=0.5, zorder=5)
		[bar.set_alpha(0.3) for bar in bars]

		# calculate medians 
		medians_per_night.append(np.nanmedian(flux_plot))
		binned_medians_per_night.append(np.nanmedian(binned))

# format the plot
fig.text(0.5, 0.01, 'hours since start of night', ha='center')
plt.subplots_adjust(wspace=0, hspace=0)
ax[0].set_ylabel('corrected flux [ppt]')
ax[0].set_xlabel("BJD")
ax[0].set_ylim(np.nanpercentile(y, 1), np.nanpercentile(y, 99))  # don't let outliers wreck the y-axis scale
fig.tight_layout()
plt.show()
################################################################################################

print("RMS data, native binning:", np.std(medians_per_night)*1e3, "ppm")
print("Binned RMS model:", np.std(binned_medians_per_night)*1e3, "ppm in", tbin, "minute bins")



