"""
Functions to plot and compare star fluxes 
"""
import glob
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import test_load_data as ld

def plot_all_comps_onedate(mainpath,targetname,obsdate,ffname,exclude_comps,plot_separate=True,save_plots=False): #JGM MODIFIED NOV 10/2023
	
	# Get data frame and comp star numbers
	df,filename = ld.return_dataframe_onedate(mainpath,targetname,obsdate,ffname)
	comp_nums = ld.get_AIJ_star_numbers(df,'Source-Sky ADU')
	all_stars = np.arange(1,np.max(comp_nums[comp_nums != None])) #number of refs sans target

	# Get target star comp counts
	bjds = df['BJD TDB'].to_numpy()
	target_counts = df['Target Source-Sky ADU']

	# Initialize figure
	fig, ax = plt.subplots(figsize=(15,10))
	ax.scatter(bjds,target_counts,s=6,color='black',label='Target')

	# Filter excluded stars from all star list
	for star in all_stars:
		if np.any(exclude_comps == star):
			all_stars = np.delete(all_stars, np.argwhere(all_stars == star))

	# Plot counts per reference star
	medians = []
	marker = itertools.cycle((',','+',',','o','.','o','*'))
	for star in all_stars:
		if star in comp_nums:
			nthcomp_counts = df['Ref '+str(star)+' Source-Sky ADU'].to_numpy()
			median_counts = np.nanmedian(nthcomp_counts)
			medians.append(median_counts)
			ax.scatter(bjds,nthcomp_counts,s=4,marker=next(marker),label='R'+str(star)+'<{:.1f} ADU>'.format(median_counts)) 
		# Plot in separate panels
		if plot_separate:
			ax.set_xlabel('BJD')
			ax.set_ylabel('Comp Counts')
			ax.set_yscale('log')
			plt.legend(loc='upper right')
			fig.tight_layout()
			plt.show()

	# Plot counts per comp star for all comps in one panel
	medians = np.array(medians)
	sorted_inds = np.argsort(medians)[::-1]
	sorted_medians = medians[sorted_inds] 
	sorted_comp_nums = all_stars[sorted_inds]
	nrefs = len(all_stars)
	if not plot_separate:
		ax.set_xlabel('BJD')
		ax.set_ylabel('Comp Counts')
		ax.set_yscale('log')
		ax.set_title(obsdate + str(' : {} Refs Used (bright to dim):{}'.format(nrefs,sorted_comp_nums)))
		plt.legend(loc='upper right')
		fig.tight_layout()
		if save_plots:
                       plt.savefig('{}_{}_raw_targ_and_ref_counts.pdf'.format(targetname,obsdate)) #TODO: decide where to save these to
		else:
                       plt.show() 
	return None

def rank_comps(mainpath,targetname,obsdate,ffname,exclude_comps):#JGM MODIFIED NOV 11/2023
	
	# Get data frame and comp star numbers
	df,filename = ld.return_dataframe_onedate(mainpath,targetname,obsdate,ffname)
	comp_nums = ld.get_AIJ_star_numbers(df,'Source-Sky ADU')
	all_stars = np.arange(1,np.max(comp_nums[comp_nums != None]))

	medians = np.array([])
	for star in all_stars:
		if np.any(exclude_comps == star):
			all_stars = np.delete(all_stars, np.argwhere(all_stars == star))

	for star in all_stars:
		nthcomp_counts = df['Ref '+str(star)+' Source-Sky ADU'].to_numpy()
		medians = np.append(medians,np.median(nthcomp_counts))

	sorted_inds = np.argsort(medians)[::-1]
	sorted_medians = medians[sorted_inds] 
	sorted_comp_nums = all_stars[sorted_inds] # brightest to dimmest

	return medians,all_stars,sorted_medians, sorted_comp_nums
