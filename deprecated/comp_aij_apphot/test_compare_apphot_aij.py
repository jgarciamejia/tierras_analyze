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


# Deal with command line
ap = argparse.ArgumentParser()
ap.add_argument("-target", required=True, help="Name of observed target exactly as shown in raw FITS files.")       
ap.add_argument("-ffname", required=True, help="Name of folder in which to store reduced+flattened data. Convention is flatXXXX. XXXX=0000 means no flat was used.")
ap.add_argument("-exclude_dates", nargs='*',type=str,help="Dates to exclude, if any. Write the dates separated by a space (e.g., 19950119 19901023)")
args = ap.parse_args()

target = args.target
ffname = args.ffname
exclude_dates = np.array(args.exclude_dates)

# Define data path
basepath = '/data/tierras/'
lcpath = os.path.join(basepath,'lightcurves')
lcfolderlist = np.sort(glob.glob(lcpath+"/**/"+target))

# Load ap_phot.py 'optimal aperture' target and reference flux data into global lists
apphot_complist = [2,3,5,6,7,8,10,12,13,16,17,20] # for 2M3495, manually selected to match AIJ reduction
full_bjd, bjd_save, full_flux, full_err, full_reg, full_relflux = ld.make_raw_global_lists(lcpath,target,ffname,exclude_dates,apphot_complist,ap_radius=None)

# Load ap_phot.py 'AIJ equivalent aperture/annulus' target and reference flux data into global lists
full_bjd2, bjd_save2, full_flux2, full_err2, full_reg2, full_relflux2 = ld.make_raw_global_lists(lcpath,target,ffname,exclude_dates,apphot_complist,ap_radius=15)

# Load AIJ target and reference flux data into global lists. Rather nasty data upload. 
aij_df = pd.read_csv('2M3495_aij_reduced_concat_data.csv')
#aij_corr_df = pd.read_csv('2M3495_aij_corrected_data.csv')
aij_bjd,aij_flux = aij_df['BJD_TDB_MOBS'],aij_df['Source-Sky_T1']


# get list and number of dates for plotting
lcdatelist = [lcfolderlist[ind].split("/")[4] for ind in range(len(lcfolderlist))]
try:
    nnights = len(lcfolderlist) - len(exclude_dates)
except TypeError:
    nnights = len(lcfolderlist)

# Plot/compare: AIJ vs ap_phot target counts
fig, ax = plt.subplots(1,nnights, sharey='row', figsize=(8*nnights, 8))
for nth_night in range(nnights):

    # get the indices corresponding to the nth_night in ap_phot 'optimal aperture'
    use_bjds = np.array(bjd_save[nth_night])
    apphot_inds = np.where((full_bjd > np.min(use_bjds)) & (full_bjd < np.max(use_bjds)))[0]

    # identify and plot the raw target counts from ap_phot'optimal aperture'
    night_bjd = full_bjd[apphot_inds]
    night_flux = full_flux[apphot_inds]
    ax[nth_night].scatter(night_bjd-full_bjd[0],night_flux,color='black',s=5,label='Raw Counts (ap_phot.py), Optimal Aperture')

    # get the indices corresponding to the nth_night in ap_phot '15 pix aperture'
    apphot_inds2 = np.where((full_bjd2 > np.min(use_bjds)) & (full_bjd2 < np.max(use_bjds)))[0]

    # identify and plot the raw target counts from ap_phot '15 pix aperture'
    night_bjd2 = full_bjd2[apphot_inds2]
    night_flux2 = full_flux2[apphot_inds2]
    ax[nth_night].scatter(night_bjd2-full_bjd[0],night_flux2,color='green',s=8,marker='s',label='Raw Counts (ap_phot.py), 15/20/30 radius/an_in/an_out')

    # get the indices corresponding to the nth_night in AIJ
    aij_inds = np.where((aij_bjd > np.min(use_bjds)) & (aij_bjd < np.max(use_bjds)))[0]

    # identify and plot the raw target counts from AIJ
    aij_night_bjd = aij_bjd[aij_inds]
    aij_night_flux = aij_flux[aij_inds]
    ax[nth_night].scatter(aij_night_bjd-full_bjd[0],aij_night_flux,color='red',s=5,alpha=0.7,label='Raw Counts (AIJ)')
    
    # title for each night
    ax[nth_night].set_title(lcdatelist[nth_night])

ax[0].set_ylabel('Flux of Source-Sky (ADU)')
fig.text(0.5, 0.01, 'BJD TDB - {}'.format(str(np.round(full_bjd[0],2))), ha='center')
ax[0].set_ylim(np.nanpercentile(aij_flux, 1), np.nanpercentile(aij_flux, 99))  # don't let outliers wreck the y-axis scale
#fig.tight_layout()
fig.suptitle(target)
plt.legend()
plt.subplots_adjust(wspace=0, hspace=0)
plt.show()



