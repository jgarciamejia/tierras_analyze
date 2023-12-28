import os
import sys
import pdb
import glob
import argparse 
import importlib
import numpy as np
import pandas as pd
from time import time
import matplotlib.pyplot as plt
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
ap.add_argument("-exclude_comps", required=False, nargs='*',type=int,help="Comparison stars to exclude, if any. Write the comp/ref number assignments ONLY, separated by a space (e.g., 2 5 7 if you want to exclude references/comps R2,R5, and R7.) ")
ap.add_argument("-comp_weight_thold", required=False, type=float, default=0.01, help="Sets the weight threshold below which to ignore comparison stars. The weight of each comp star is read from the night_weights.csv folder for the FIRST night in the data set.")
args = ap.parse_args()

target = args.target
ffname = args.ffname
exclude_dates = np.array(args.exclude_dates)
exclude_comps = np.array(args.exclude_dates)
treshold = args.comp_weight_thold

# Define data path
basepath = '/data/tierras/'
lcpath = os.path.join(basepath,'lightcurves')
lcfolderlist = np.sort(glob.glob(lcpath+"/**/"+target))

# Load the list of comparison stars to use from the FIRST night in the list.
compfname = os.path.join(lcfolderlist[0],ffname,"night_weights.csv")
compfname_df = pd.read_csv(compfname)
complist = compfname_df['Reference'].to_numpy()
complist = np.array([int(s.split()[-1]) for s in complist])

# Make mask of comparison stars specified by the user
mask = ~np.isin(complist,exclude_comps)

# Make mask of comparison stars with weights below the user-defined threshold, or 0.01.
compweights = compfname_df['Weight'].to_numpy() 
mask2 = compweights > treshold
complist = complist[mask & mask2]

pdb.set_trace()

# Load raw target and ref fluxes into global lists
full_bjd, bjd_save, full_flux, full_err, full_reg = ld.make_global_lists(lcfolderlist)

# mask bad data and use comps to calculate frame-by-frame magnitude zero points
x, y, err = mearth_style(full_bjd, full_flux, full_err, full_reg) #TO DO: how to integrate weights into mearth_style?

pdb.set_trace()

# Plot/compare original data to corrected data 
try:
    nnights = len(lcfolderlist) - len(exclude_dates) #N can be set to lower value for testing/inspection on individual nights. N should be = len(lcfolderlist)-len(exclude_dates) 
except TypeError:
    nnights = len(lcfolderlist)

fig, ax = plt.subplots((1,nnights), sharey='row', sharex=True, figsize=(5*nnights, 5))

for nth_night in range(nnights):

    # get the indices corresponding to the nth_night
    use_bjds = np.array(bjd_save[nth_night])
    original_inds = np.where((full_bjd > np.min(use_bjds)) & (full_bjd < np.max(use_bjds)))[0]
    corr_inds = np.where((x > np.min(use_bjds)) & (x < np.max(use_bjds)))[0]

    # identify and plot the original data for the nth_night
    bjd = full_bjd[original_inds]
    flux = full_flux[original_inds]
    ax[nth_night].scatter(bjd,flux,color='black',s=2)

    # identify and plot the corrected data for the nth_night
    if len(corr_inds) == 0:  # if the entire nth_night was masked due to bad weather, don't plot anything
        continue
    else:
        corrected_bjd = x[inds]
        corrected_flux = y[inds]
        err = err[inds]
        ax[nth_night].errorbar(corrected_bjd, corrected_flux, yerr=err, color='purple', fmt='.')

ax[0].set_ylabel('Flux')
fig.text(0.5, 0.01, 'BJD TDB', ha='center')
ax[0].set_ylim(np.nanpercentile(y, 1), np.nanpercentile(y, 99))  # don't let outliers wreck the y-axis scale
fig.tight_layout()
plt.subplots_adjust(wspace=0, hspace=0)


# # convert relative flux to ppt.
# mu = np.nanmedian(y)
# y = (y / mu - 1) * 1e3
# err = (err/mu) * 1e3

