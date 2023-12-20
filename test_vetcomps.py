import os
import pdb
import re
import sys
import glob
import argparse 

import numpy as np 
import pandas as pd
import importlib
import matplotlib.pyplot as plt

import astropy.coordinates as coord
import astropy.units as u

# JGM code
import test_load_data as ld 
import test_plot_cum_phot as pcp
import test_plot_all_comps as pac
import test_bin_lc as bl

importlib.reload(ld)
importlib.reload(pcp)
importlib.reload(pac)
importlib.reload(bl)

####### User-defined Params #######
# Tests to run
do_plot_cum_phot = True # TO DO: move to the argparse
plot_comp_counts = True # TO DO: move to the argparse
###################################

# Deal with command line 
ap = argparse.ArgumentParser()

ap.add_argument("-target", required=True, help="Name of observed target exactly as shown in raw FITS files.")
ap.add_argument("-ffname", required=True, help="Name of folder in which to store reduced+flattened data. Convention is flatXXXX. XXXX=0000 means no flat was used.")

ap.add_argument("-exclude_dates", nargs='*',type=str,help="Dates to exclude, if any. Write the dates separated by a space (e.g., 19950119 19901023)")
ap.add_argument("-exclude_comps", nargs='*',type=int,help="Comparison stars to exclude, if any. Write the comp/ref number assignments ONLY, separated by a space (e.g., 2 5 7 if you want to exclude references/comps R2,R5, and R7.) ")

ap.add_argument("-normalization", default='none', help="Normalization options for a cumulative photometry plot. options are 'none', 'nightly', and 'global'.")
ap.add_argument("-binsize", type=float, default=10, help="Bin size for cumulative photometry plot.")
args = ap.parse_args()

targetname = args.target
ffname = args.ffname

normalize = args.normalization
binsize = args.binsize

exclude_comps = np.array(args.exclude_comps)
exclude_dates = np.array(args.exclude_dates)

basepath = '/data/tierras/'
lcpath = os.path.join(basepath,'lightcurves')
lcfolderlist = np.sort(glob.glob(lcpath+"/**/"+targetname))
lcdatelist = [lcfolderlist[ind].split("/")[4] for ind in range(len(lcfolderlist))] 


######## Plot cumulative photometry #######
if do_plot_cum_phot:
	dfs = pcp.plot_cum_phot(lcpath,targetname,ffname,exclude_dates,exclude_comps,normalize,binsize)

######## Plot comparison star behavior per night #######
if plot_comp_counts:      
	print ('Reference stars sorted, from brightest to dimmest:')
	for date in lcdatelist:
		if np.any(exclude_dates == date):
			print ("{} :  Excluded".format(date))
			continue
		else:
			comp_kws = pac.plot_all_comps_onedate(lcpath,targetname,date,ffname,exclude_comps,False,True)
			medians, stars, sortedmeds,this_sorted_comp_nums = pac.rank_comps(lcpath,targetname,date,ffname,exclude_comps)
			print ('{} : {}'.format(date,this_sorted_comp_nums))










