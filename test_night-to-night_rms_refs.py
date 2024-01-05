import os
import re
import sys
import pdb
import glob
import palpy
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
import test_locatemoon_givenmjd as moon

# Define site characteristics. Currently hard-coded for Tierras.
latitude = 31.680889 * palpy.DD2R #radians
longitude = -110.878750 * palpy.DD2R #radians
height = 2345.0 # meters

# Deal with command line
ap = argparse.ArgumentParser()
ap.add_argument("-target", required=True, help="Name of observed target exactly as shown in raw FITS files.")       
ap.add_argument("-ffname", required=True, help="Name of folder in which to store reduced+flattened data. Convention is flatXXXX. XXXX=0000 means no flat was used.")
ap.add_argument("-exclude_dates", nargs='*',type=str,help="Dates to exclude, if any. Write the dates separated by a space (e.g., 19950119 19901023)")
#ap.add_argument("-refs_as_target", required=True, type=int, help="Reference stars/comp stars/regressors to iterate through as the targets.")
ap.add_argument("-aperture_radius", default='optimal',help='Aperture radius (in pixels) of data to be loaded. Write as an integer (e.g., 8 if you want to use the circular_fixed_ap_phot_8.csv files for all loaded dates of data). Defaults to the optimal radius. ')
ap.add_argument("-target_gaiaid", type=int, help="Gaia DR2 or DR3 ID for target. Numbers only - do not include the DRX suffix.")
args = ap.parse_args()

target = args.target
ffname = args.ffname
exclude_dates = np.array(args.exclude_dates)
ref_as_target = args.ref_as_target
ap_radius = args.aperture_radius
gaia_id = args.target_gaiaid

# Define path to data  
basepath = '/data/tierras/'
lcpath = os.path.join(basepath,'lightcurves')
lcfolderlist = np.sort(glob.glob(lcpath+"/**/"+target))
lcdatelist = np.array([lcfolderlist[ind].split("/")[4] for ind in range(len(lcfolderlist))]) 
date_mask = ~np.isin(lcdatelist,exclude_dates)
lcdatelist = lcdatelist[date_mask]

# Load the list of comparison stars to use.
compfname = os.path.join(lcfolderlist[0],ffname,"night_weights.csv")
compfname_df = pd.read_csv(compfname)
complist = compfname_df['Reference'].to_numpy()
complist = np.array([int(s.split()[-1]) for s in complist])
mask= ~np.isin(complist,ref_as_target) #exclude reference used as target from list of comps
complist = complist[mask]

# Load raw target and reference fluxes into global lists
full_bjd, bjd_save, full_flux, full_err, full_reg, full_flux_div_expt, full_err_div_expt = ld.make_global_lists_refastarget(ref_as_target,lcpath,target,ffname,exclude_dates,complist,ap_radius=ap_radius)

# mask bad data and use comps to calculate frame-by-frame magnitude zero points
x, y, err = mearth_style(full_bjd, full_flux_div_expt, full_err_div_expt, full_reg) #TO DO: how to integrate weights into mearth_style?

# convert relative flux to ppt.
mu = np.nanmedian(y)
y = (y / mu - 1) * 1e3
err = (err/mu) * 1e3

################################################################################################
###### TO DO: fix this plotting loop to be readable (for large N it becomes quite unruly) ######
# plot the data night-by-night
# initialize the pplot

medians_per_night = []
binned_medians_per_night = []

fully_masked_inds = []
fully_masked_dates = []

# if the entire night was masked due to bad weather, remove from lcdatelist so a plot is not made for it 
for nth_night in range(len(lcdatelist)):
	# get the indices corresponding to a given night
	use_bjds = np.array(bjd_save[nth_night])
	inds = np.where((x > np.min(use_bjds)) & (x < np.max(use_bjds)))[0]
	#print (lcdatelist[nth_night],len(inds))

	if len(inds) == 0:  # if the entire night was masked due to bad weather, don't plot anything
		#print ('made it')
		fully_masked_inds.append(nth_night)
		fully_masked_dates.append(lcdatelist[nth_night])

date_mask2 = ~np.isin(lcdatelist,fully_masked_dates)
lcdatelist = lcdatelist[date_mask2]

# remove fully masked nights from bjd_save to not screw up plot indexing below
for ind in sorted(fully_masked_inds,reverse=True):
	bjd_save.pop(ind)

# plot data of each night not excluded by user and where data survived clean up:
fig, ax = plt.subplots(1, len(lcdatelist), sharey='row', sharex=True, figsize=(30, 4))

for nth_night in range(len(lcdatelist)):
	# get the indices corresponding to a given night. have to keep repeat loop to correctly index bjd_save
	use_bjds = np.array(bjd_save[nth_night])
	inds = np.where((x > np.min(use_bjds)) & (x < np.max(use_bjds)))[0]

	# identify and plot the night of data
	bjd_plot = x[inds]
	flux_plot = y[inds]
	median_bjd,median_flux = np.nanmedian(bjd_plot),np.nanmedian(flux_plot)
	err_plot = err[inds]
	markers, caps, bars = ax[nth_night].errorbar((bjd_plot-np.min(bjd_plot))*24., flux_plot, yerr=err_plot, fmt='k.', alpha=0.2)
	[bar.set_alpha(0.05) for bar in bars]
	#ax[nth_night].scatter((median_bjd-np.min(bjd_plot))*24, median_flux, s=70,color='white', edgecolor='black', marker='*', alpha=1.0, zorder=10)

	# add bins
	tbin = 20  # bin size in minutes
	xs_b, binned, e_binned = ep_bin((bjd_plot - np.min(bjd_plot))*24, flux_plot, tbin/60.)
	marker, caps, bars = ax[nth_night].errorbar(xs_b, binned, yerr=e_binned, color='purple', fmt='.', alpha=0.5, zorder=5)
	[bar.set_alpha(0.3) for bar in bars]

	# calculate medians 
	medians_per_night.append(np.nanmedian(flux_plot))
	binned_medians_per_night.append(np.nanmedian(binned))

	# add title and moon info
	date = lcdatelist[nth_night]
	moonsep, moonillum = moon.get_moon_sep_and_illum(gaia_id, date, latitude, longitude, height)
	moonsep, moonillum = np.round(moonsep,1), np.round(moonillum,1)
	text_string1 = '{:.0f} deg'.format(moonsep)
	text_string2 = '{:.0f} %'.format(moonillum)
	ax[nth_night].text(0.95,0.95,text_string1, transform = ax[nth_night].transAxes,fontsize=8, verticalalignment='top',horizontalalignment='right')
	ax[nth_night].text(0.95,0.90,text_string2, transform = ax[nth_night].transAxes,fontsize=8, verticalalignment='top',horizontalalignment='right')
	ax[nth_night].set_title(lcdatelist[nth_night])

# format the plot
fig.text(0.5, 0.01, 'hours since start of night', ha='center')
plt.subplots_adjust(wspace=0, hspace=0)
ax[0].set_ylabel('corrected flux')
ax[0].set_xlabel("BJD")
ax[0].set_ylim(np.nanpercentile(y, 1), np.nanpercentile(y, 99))  # don't let outliers wreck the y-axis scale
fig.tight_layout()
plt.show()
################################################################################################

print("RMS data, native binning:", np.std(medians_per_night)*1e3, "ppm")
print("Binned RMS model:", np.std(binned_medians_per_night)*1e3, "ppm in", tbin, "minute bins")



