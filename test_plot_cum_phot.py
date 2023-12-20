"""
Script to plot cumulative photometry as a function of Julian Date.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import glob
import re

import test_load_data as ld 
import test_bin_lc as bl
from jgmmedsig import *

def plot_cum_phot(lcpath,targetname,ffname,exclude_dates,exclude_comps,normalize='none',binsize=10,show_plot=True,save_fig=False):

	# Initialize array to hold global rel fluxes
	glob_bjds = np.array([])
	glob_rel_flux = np.array([])

	# Initialize Figure 
	fig, ax = plt.subplots(figsize=(15,5))

	#Load data frame per date
	lcfolderlist = np.sort(glob.glob(lcpath+"/**/"+targetname))
	lcdatelist = [lcfolderlist[ind].split("/")[4] for ind in range(len(lcfolderlist))]
	dfs = [None] * len(lcdatelist)

	for i,date in enumerate(lcdatelist):		
		if np.any(exclude_dates == date):
			print ('Date {} Excluded'.format(date))
			continue
		else:
			df,bjds,rel_flux,airmasses,widths = ld.return_data_onedate(lcpath,targetname,date,ffname,exclude_comps)
			texp = df['Exposure Time'][0]
			dfs[i] = df

			# Bin and normalize (if desired) the light curve
			if normalize == 'none':
				xbin, ybin, _ = bl.bin_lc_binsize(bjds, rel_flux, binsize)
				ax.scatter(bjds, rel_flux, s=20, color='seagreen')
				ax.scatter(xbin, ybin, s=60, color='darkgreen', alpha = 0.9)
			elif normalize == 'nightly': 
				rel_flux /= np.median(rel_flux)
				xbin, ybin, _ = bl.bin_lc_binsize(bjds, rel_flux, binsize)
				ax.scatter(bjds, rel_flux, s=20, color='seagreen')
				ax.scatter(xbin, ybin, s=60, color='darkgreen', alpha = 0.9)
			elif normalize == 'global':
				glob_rel_flux = np.append(glob_rel_flux, rel_flux)
				glob_bjds = np.append(glob_bjds, bjds)

	if normalize == 'global':
			glob_rel_flux /= np.median(glob_rel_flux)
			ax.scatter(glob_bjds, glob_rel_flux, s=20, color='seagreen')
			glob_xbin, glob_ybin, _ = bl.bin_lc_binsize(glob_bjds, glob_rel_flux, binsize)
			ax.scatter(glob_xbin, glob_ybin, s=60, color='darkgreen', alpha = 0.9)

	# Config plot labels
	ax.set_xlabel("Time (BJD)")
	ax.set_ylabel("Normalized flux")
	
	if normalize == 'none':
		ax.set_title('No nightly or global norm., texp = {} min, Bin size = {} min'.format(texp/60,binsize))
	elif normalize == 'nightly':
		ax.set_title('Nightly norm., texp = {} min, Bin size = {} min'.format(texp/60,binsize))
	elif normalize == 'global':
		ax.set_title('Global norm., texp = {} min, Bin size = {} min'.format(texp/60,binsize))
	
	if show_plot:
		plt.show()
	if save_fig:
		fig_name = os.path.join(lcpath,lcdatelist[0],targetname,ffname)
		fig.save_fig(fig_name)

	return dfs
