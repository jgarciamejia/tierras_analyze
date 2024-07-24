import argparse 
import os 
import numpy as np 
import pandas as pd 
from scipy.stats import sigmaclip, pearsonr 
from glob import glob 
from ap_phot import tierras_binner_inds, set_tierras_permissions, tierras_binner, t_or_f
import copy 
from astroquery.simbad import Simbad
import matplotlib.pyplot as plt
plt.ion() 
from matplotlib import gridspec
from pathlib import Path 
from scipy.optimize import curve_fit
import pickle
import time 
import pyarrow.parquet as pq 
from astropy.stats import sigma_clip
import gc
import warnings
warnings.filterwarnings('ignore')

def ref_selection(target_ind, sources, delta_target_rp=5, target_distance_limit=4000, max_refs=50):

	# now select suitable reference stars
	target_rp = sources['phot_rp_mean_mag'][target_ind]
	target_x = sources['X pix'][target_ind]
	target_y = sources['Y pix'][target_ind]	

	# figure out which references are within the specified delta_target_rp of the target's Rp mag
	candidate_rp_inds = sources['phot_rp_mean_mag'] < target_rp + delta_target_rp
	candidate_distance_inds = np.sqrt((target_x-sources['X pix'])**2+(target_y-sources['Y pix'])**2) < target_distance_limit

	# remove the target from the candidate reference list 
	ref_inds = np.where(candidate_rp_inds & candidate_distance_inds)[0]
	ref_inds = np.delete(ref_inds, np.where(ref_inds == target_ind)[0][0])

	# if more than max_refs have been found, cut to max_refs
	# print(f'Found {len(inds)} suitable reference stars!')
	if len(ref_inds) > max_refs:
		# print(f'Cutting to brightest {max_refs} reference stars.')
		ref_inds = ref_inds[0:max_refs]

	# refs = sources.iloc[ref_inds]
	return ref_inds

def mearth_style_pat_weighted_flux(flux, flux_err, non_linear_flag, airmasses, exptimes, source_ids=None, max_iters=20):

	""" Use the comparison stars to derive a frame-by-frame zero-point magnitude. Also filter and mask bad cadences """
	""" it's called "mearth_style" because it's inspired by the mearth pipeline """
	""" this version works with fluxes """

	# calculate relative scintillation noise 
	sigma_s = 0.09*130**(-2/3)*airmasses**(7/4)*(2*exptimes)**(-1/2)*np.exp(-2306/8000)

	n_sources = flux.shape[1]
	n_ims = flux.shape[0]

	# mask any cadences where the flux is negative for any of the sources 
	mask = np.any(flux < 0,axis=1)
	flux[mask] = np.nan 
	flux_err[mask] = np.nan

	mask_save = np.zeros(n_ims, dtype='bool')
	mask_save[np.where(mask)] = True

	# regressor_inds = np.arange(1,flux.shape[1]) # get the indices of the stars to use as the zero point calibrators; these represent the indices of the calibrators *in the data_dict arrays*

	# grab target and source fluxes and apply initial mask 
	# target_flux = flux[:,0]
	# target_flux_err = flux_err[:,0]
	# regressors = flux[:,regressor_inds]
	# regressors_err = flux_err[:,regressor_inds]
	#nl_flags = non_linear_flag[:,regressor_inds]

	tot_regressor = np.sum(flux, axis=1)  # the total regressor flux at each time point = sum of comp star fluxes in each exposure

	# identify cadences with "low" flux by looking at normalized summed reference star fluxes
	zp0s = tot_regressor/np.nanmedian(tot_regressor) 	
	mask = np.zeros_like(zp0s, dtype='bool')  # initialize another bad data mask
	mask[np.where(zp0s < 0.25)[0]] = 1  # if regressor flux is decremented by 75% or more, this cadence is bad
	
	flux[mask] = np.nan
	# target_flux[mask] = np.nan 
	# target_flux_err[mask] = np.nan 

	# regressors[mask] = np.nan
	# regressors_err[mask] = np.nan
	mask_save[np.where(mask)] = True

	# repeat the cs estimate now that we've masked out the bad cadences
	# phot_regressor = np.nanpercentile(regressors, 90, axis=1)  # estimate photometric flux level for each star
	norms = np.nanmedian(flux, axis=0)
	# regressors_norm = regressors / norms
	regressors_err_norm = flux_err / norms

	weights_init = 1/(np.nanmedian(regressors_err_norm,axis=0)**2)

	#de-weight references with ANY non-linear frames 
	nl_source_inds = np.sum(non_linear_flag, axis=0) > 0	
	weights_init[nl_source_inds] = 0
	weights_init /= np.nansum(weights_init)

	# do a 'crude' weighting loop to figure out which regressors, if any, should be totally discarded	
	delta_weights = np.zeros(flux.shape[1])+999 # initialize
	threshold = 1e-3 # delta_weights must converge to this value for the loop to stop
	weights_old = weights_init
	full_ref_inds = np.arange(flux.shape[1])
	count = 0 

	t1 = time.time()
	while len(np.where(delta_weights>threshold)[0]) > 0:
		stddevs_measured = np.zeros(flux.shape[1])		
		stddevs_expected = np.zeros_like(stddevs_measured) 
		f_corr_save = np.zeros((n_ims, n_sources))
		# alc_save = np.zeros((n_ims, n_sources))
		# sigma_save = np.zeros_like(f_corr_save)
		# loop over each regressor
		for jj in range(flux.shape[1]):
			# skip any stars with 0 weights
			if weights_old[jj] == 0:
				continue

			F_t = copy.deepcopy(flux[:,jj])
			N_t = copy.deepcopy(flux_err[:,jj])
			
			# make its zeropoint correction using the flux of all the *other* regressors
			use_inds = np.delete(full_ref_inds, jj)

			# re-normalize the weights to sum to one
			weights_wo_jj = weights_old[use_inds]
			weights_wo_jj /= np.nansum(weights_wo_jj)

			# create a zeropoint correction using those weights 
			F_e = np.matmul(flux[:,use_inds],weights_wo_jj)
			N_e = np.sqrt(np.matmul(flux_err[:,use_inds]**2, weights_wo_jj**2))
			
			# calculate the relative flux 
			F_rel_flux = F_t / F_e
			sigma_rel_flux = np.sqrt((N_t/F_e)**2 + (F_t*N_e/(F_e**2))**2)

			# renormalize 
			norm = np.nanmedian(F_rel_flux)
			F_corr = F_rel_flux/norm 
			sigma_rel_flux= sigma_rel_flux/norm 
			
			# calculate total error on F_rel flux from sigma_rel_flux and sigma_scint			
			
			sigma_scint = 1.5*sigma_s*np.sqrt(1 + 1/(len(use_inds)))
			sigma_tot = np.sqrt(sigma_rel_flux**2 + sigma_scint**2)	
			
			# sigma clip 
			sc_mask = sigma_clip(F_corr,sigma=4).mask
			F_corr[sc_mask] = np.nan 
			sigma_tot[sc_mask] = np.nan

			# record the standard deviation of the corrected flux
			stddevs_measured[jj] = np.nanstd(F_corr)
			stddevs_expected[jj] = np.nanmedian(sigma_tot)
			# alc_save[:,jj] = F_e
			f_corr_save[:, jj]  = F_corr
			# sigma_save[:, jj] = sigma_tot
			
		# update the weights using the measured standard deviations	
		weights_new = 1/stddevs_measured**2
		weights_new /= np.sum(weights_new[~np.isinf(weights_new)])			
		weights_new[np.isinf(weights_new)] = 0
		weights_new /= np.sum(weights_new)
		delta_weights = abs(weights_new-weights_old)
		weights_old = weights_new
		count += 1 
		if count == max_iters:
			break	
	
	weights = weights_new
	# determine if any references should be totally thrown out based on the ratio of their measured/expected noise
	noise_ratios = stddevs_measured/stddevs_expected

	# the noise ratio threshold will depend on how many bad/variable reference stars were used in the ALC
	# sigmaclip the noise ratios and set the upper limit to the n-sigma upper bound 

	# v, l, h = sigmaclip(noise_ratios[~np.isnan(noise_ratios)], 2, 2)
	# weights[np.where(noise_ratios[~np.isnan(noise_ratios)]>h)[0]] = 0

	sc_mask = sigma_clip(noise_ratios, sigma_lower=np.inf, sigma_upper=1).mask 

	# breakpoint()
	weights[sc_mask] = 0 
	# weights[np.where(noise_ratios>5)[0]] = 0
	weights /= sum(weights)
	# weights[np.where(abs(-0.5-allen_slopes) > 0.3)] = 0 
	# weights /= sum(weights)	

	# breakpoint() 

	if len(np.where(weights == 0)[0]) > 0:
		# now repeat the weighting loop with the bad refs removed 
		delta_weights = np.zeros(flux.shape[1])+999 # initialize
		threshold = 1e-6 # delta_weights must converge to this value for the loop to stop
		weights_old = weights
		full_ref_inds = np.arange(flux.shape[1])
		count = 0
		while len(np.where(delta_weights>threshold)[0]) > 0:
			stddevs_measured = np.zeros(flux.shape[1])
			stddevs_expected = np.zeros_like(stddevs_measured)

			for jj in range(flux.shape[1]):
				if weights_old[jj] == 0:
					continue

				F_t = copy.deepcopy(flux[:,jj])
				N_t = copy.deepcopy(flux_err[:,jj])

				use_inds = np.delete(full_ref_inds, jj)
				weights_wo_jj = weights_old[use_inds]
				weights_wo_jj /= np.nansum(weights_wo_jj)
				
				# create a zeropoint correction using those weights 
				F_e = np.matmul(flux[:,use_inds], weights_wo_jj)
				N_e = np.sqrt(np.matmul(flux_err[:,use_inds]**2, weights_wo_jj**2))

				# calculate the relative flux 
				F_rel_flux = F_t / F_e
				sigma_rel_flux = np.sqrt((N_t/F_e)**2 + (F_t*N_e/(F_e**2))**2)

				# renormalize
				norm = np.nanmedian(F_rel_flux)
				F_corr = F_rel_flux/norm 
				sigma_rel_flux = sigma_rel_flux/norm 	
	
				# calculate total error on F_rel flux from sigma_rel_flux and sigma_scint				
				sigma_scint = 1.5*sigma_s*np.sqrt(1 + 1/(len(use_inds)))
				sigma_tot = np.sqrt(sigma_rel_flux**2 + sigma_scint**2)

				# sigma clip 
				sc_mask = sigma_clip(F_corr,sigma=4).mask
				F_corr[sc_mask] = np.nan 
				sigma_tot[sc_mask] = np.nan

				# record the standard deviation of the corrected flux
				stddevs_measured[jj] = np.nanstd(F_corr)
				stddevs_expected[jj] = np.nanmedian(sigma_tot)

			weights_new = 1/(stddevs_measured**2)
			weights_new /= np.sum(weights_new[~np.isinf(weights_new)])
			weights_new[np.isinf(weights_new)] = 0
			delta_weights = abs(weights_new-weights_old)
			weights_old = weights_new
			count += 1
			if count == max_iters:
				break		
		
	weights = weights_new

	# calculate the zero-point correction

	# F_e = np.matmul(regressors, weights)
	# N_e = np.sqrt(np.matmul(regressors_err**2, weights**2))	
	# flux_corr = target_flux / F_e
	# err_corr = np.sqrt((target_flux_err/F_e)**2 + (target_flux*N_e/(F_e**2))**2)

	# # renormalize
	# norm = np.nanmedian(flux_corr)
	# flux_corr /= norm 
	# err_corr /= norm 

	# alc = F_e
	# alc_err = N_e 
	# flux_err_corr = err_corr

	# weights = np.insert(weights, 0, 0)
	return weights, mask_save

def allen_deviation(times, flux, flux_err):

	max_ppb = int(np.floor(len(times)/4))
	bins = []
	i = 0
	ppb = 1
	while ppb < max_ppb:
		bins.append(ppb)
		i += 1
		ppb = 2**i

	inds_list = [[] for j in range(len(bins))]
	for i in range(len(bins)):
		n_bins = int(np.floor(len(times)/bins[i]))
		for j in range(n_bins):
			inds_list[i].append(np.arange(j*bins[i],(j+1)*bins[i]))

	std = np.zeros(len(bins))
	theo = np.zeros(len(bins))

	for i in range(len(inds_list)):
		inds = inds_list[i] 
		binned_flux = []
		theo[i] = np.nanmean(flux_err)*1/np.sqrt(bins[i])
		for j in range(len(inds)):
			binned_flux.extend([np.nanmean(flux[inds[j]])])
		std[i] = np.nanstd(binned_flux)

	# plt.plot(bins, std, marker='.')
	# plt.plot(bins, theo, marker='.')
	# plt.xscale('log')
	# plt.yscale('log')	
	# coeffs, var = curve_fit(lambda t,a,b: a*t**b,  bins,  std, p0=(1e-2,-0.5)) 
	# plt.plot(bins, coeffs[0]*bins**coeffs[1])
	# breakpoint()
	# plt.close()	
	# slope = coeffs[1]
	return bins, std, theo

def plot_target_summary(data_dict, bin_mins=10):
	plt.ioff()

	target = data_dict['Target']
	date = data_dict['Date']
	ffname = data_dict['ffname']

	nan_inds = ~np.isnan(data_dict['Flux'])

	times_ = data_dict['BJD'][nan_inds]
	x_offset =  int(np.floor(times_[0]))
	times = times_ - x_offset


	flux = data_dict['Flux'][nan_inds]
	flux_err = data_dict['Flux Error'][nan_inds]
	alc = data_dict['ALC'][nan_inds]
	alc_err = data_dict['ALC Error'][nan_inds]
	corr_flux = data_dict['Corrected Flux'][nan_inds]
	corr_flux_err = data_dict['Corrected Flux Error'][nan_inds]
	airmass = data_dict['Airmass'][nan_inds]
	ha = data_dict['Hour Angle'][nan_inds]
	sky = data_dict['Sky'][nan_inds]
	x = data_dict['X'][nan_inds]
	y = data_dict['Y'][nan_inds]
	fwhm_x = data_dict['FWHM X'][nan_inds]
	fwhm_y = data_dict['FWHM Y'][nan_inds]
	humidity = data_dict['Dome Humidity'][nan_inds]
	exptimes = data_dict['Exposure Time'][nan_inds]

	fig = plt.figure(figsize=(8,14))	
	gs = gridspec.GridSpec(8,1,height_ratios=[0.75,1,0.75,0.75,0.75,0.75,0.75,1])
	ax1 = plt.subplot(gs[0])
	ax2 = plt.subplot(gs[1],sharex=ax1)
	ax3 = plt.subplot(gs[2],sharex=ax1)
	ax_ha = ax3.twinx()
	ax4 = plt.subplot(gs[3],sharex=ax1)
	ax5 = plt.subplot(gs[4],sharex=ax1)
	ax6 = plt.subplot(gs[5],sharex=ax1)
	ax7 = plt.subplot(gs[6],sharex=ax1)
	ax8 = plt.subplot(gs[7])

	# plot 1: normalized raw target flux and its alc 
	label_size = 11

	v, l, h = sigmaclip(flux)
	use_inds = np.where((flux > l) & (flux < h))[0]
	ax1.plot(times[use_inds], flux[use_inds]/np.nanmedian(flux[use_inds]), marker='.', color='k',ls='', label='Norm. targ. flux')
	ax1.plot(times[use_inds], alc[use_inds]/np.nanmedian(alc[use_inds]), marker='.', color='r',ls='', label='Norm. ALC flux')
	ax1.tick_params(labelsize=label_size, labelbottom=False)
	ax1.legend(loc='center left', bbox_to_anchor=(1,0.5),fontsize=10)
	ax1.set_ylabel('Norm. Flux',fontsize=label_size)

	# plot 2: corrected target flux and binned data
	ax2.plot(times[use_inds], corr_flux[use_inds], marker='.',color='#b0b0b0',ls='',label='Corr. targ. flux')
	bx, by, bye = tierras_binner(times[use_inds], corr_flux[use_inds], bin_mins=bin_mins)
	ax2.errorbar(bx,by,bye,marker='o',mfc='none',mec='k',mew=1.5,ecolor='k',ms=7,ls='',zorder=3,label=f'{bin_mins:d}-min bins')
	ax2.legend(loc='center left', bbox_to_anchor=(1,0.5),fontsize=10)
	ax2.tick_params(labelsize=label_size, labelbottom=False)
	ax2.set_ylabel('Corr. Flux',fontsize=label_size)
	
	# plot 3: airmass and hour angle
	ax3.plot(times, airmass, color='tab:blue',lw=2)
	ax3.tick_params(labelsize=label_size, labelbottom=False, labelcolor='tab:blue', axis='y')
	ax3.set_ylabel('Airmass',fontsize=label_size, color='tab:blue')
	ax_ha.plot(times, ha, color='tab:orange')
	ax_ha.tick_params(axis='y', labelcolor='tab:orange', labelsize=label_size)
	ax_ha.set_ylabel('Hour Angle', fontsize=label_size, color='tab:orange',rotation=270, labelpad=15)
	ax_ha.set_xticklabels([])

	# plot 4: sky 
	sky[sky == 0] = np.nan
	ax4.plot(times, sky/exptimes,color='tab:cyan',lw=2)
	ax4.tick_params(labelsize=label_size)
	ax4.set_ylabel('Sky\n(ADU/s)',fontsize=label_size)
	ax4.tick_params(labelbottom=False)

	# plot 5: centroids 
	ax5.plot(times, x-np.nanmedian(x),color='tab:green',lw=2,label='X-med(X)')
	ax5.plot(times, y-np.nanmedian(y), color='tab:red',lw=2,label='Y-med(Y)')
	ax5.tick_params(labelsize=label_size)
	ax5.set_ylabel('Pos.\n(Pix.)',fontsize=label_size)
	ax5.legend(loc='center left', bbox_to_anchor=(1,0.5),fontsize=10)
	v1,l1,h1 = sigmaclip(x-np.nanmedian(x),5,5)
	v2,l2,h2 = sigmaclip(y-np.nanmedian(y),5,5)
	ax5.set_ylim(np.min([l1,l2]),np.max([h1,h2]))
	ax5.tick_params(labelbottom=False)

	# plot 6: fwhm 
	fwhm_x[fwhm_x == 0] = np.nan 
	fwhm_y[fwhm_y == 0] = np.nan
	ax6.plot(times,fwhm_x,color='tab:pink',lw=2,label='X')
	ax6.plot(times, fwhm_y, color='tab:purple', lw=2,label='Y')
	ax6.legend(loc='center left', bbox_to_anchor=(1,0.5),fontsize=10)
	ax6.tick_params(labelsize=label_size, labelbottom=False)
	ax6.set_ylabel('FWHM\n(")',fontsize=label_size)

	# plot 7: humidity 
	ax7.plot(times, humidity,lw=2, color='tab:brown')
	ax7.tick_params(labelsize=label_size, labelbottom=True)
	ax7.set_ylabel('Dome\nHumid. (%)',fontsize=label_size)
	ax7.set_xlabel(f'Time - {x_offset}'+' (BJD$_{TDB}$)',fontsize=label_size)
	ax7.set_xticklabels([f'{i:.2f}' for i in ax7.get_xticks()])

	# plot 8: allen deviation
	bins, std, theo  = allen_deviation(times[use_inds], corr_flux[use_inds], corr_flux_err[use_inds])

	ax8.plot(bins, std*1e6, lw=2,label='Measured', marker='.')
	ax8.plot(bins, theo*1e6,lw=2,label='Theoretical', marker='.')

	ax8.set_xlabel('Bin size (min)',fontsize=label_size)
	ax8.set_ylabel('$\sigma$ (ppm)',fontsize=label_size)
	ax8.set_yscale('log')
	ax8.legend(loc='center left', bbox_to_anchor=(1,0.5),fontsize=10)
	ax8.tick_params(labelsize=label_size)
	ax8.set_yscale('log')
	ax8.set_xscale('log')
	ax8.grid(alpha=0.5)

	# tidy up
	fig.align_labels()
	plt.tight_layout()
	plt.subplots_adjust(hspace=0.7)

	summary_plot_output_path = f'/data/tierras/lightcurves/{date}/{target}/{ffname}/{date}_{target}_summary.png'
	plt.savefig(summary_plot_output_path,dpi=300)
	set_tierras_permissions(summary_plot_output_path)

	plt.close('all')
	plt.ion()
	return

def plot_target_lightcurve(data_dict,bin_mins=10):
	plt.ioff()
	ffname = data_dict['ffname']
	target = data_dict['Target']
	date = date = data_dict['Date']

	times_ = data_dict['BJD']
	x_offset = int(np.floor(times_[0]))
	times = times_ - x_offset

	corr_targ_flux = data_dict['Corrected Flux']
	corr_targ_flux_err = data_dict['Corrected Flux Error']

	norm = np.nanmedian(data_dict['Flux'])
	targ_flux = data_dict['Flux'] / norm 
	targ_flux_err =  data_dict['Flux Error'] / norm 

	v, l, h = sigmaclip(targ_flux)
	use_inds = np.where((targ_flux > l) & (targ_flux < h))[0]

	norm = np.nanmedian(data_dict['ALC'])
	alc_flux = data_dict['ALC'] / norm 
	alc_flux_err = data_dict['ALC Error'] / norm 

	#print(f"bp_rp: {targ_and_refs['bp_rp'][i+1]}")
	fig, ax = plt.subplots(2,1,figsize=(10,6.5),sharex=True)
	
	bx, by, bye = tierras_binner(times[use_inds],corr_targ_flux[use_inds],bin_mins=bin_mins)

	fig.suptitle(f'{target} on {date}',fontsize=16)
	ax[0].errorbar(times[use_inds], targ_flux[use_inds], targ_flux_err[use_inds], color='k',ecolor='k',marker='.',ls='',zorder=3,label='Target')
	ax[0].errorbar(times[use_inds], alc_flux[use_inds], alc_flux_err[use_inds], color='r', ecolor='r', marker='.',ls='',zorder=3,label='ALC')
	ax[0].set_ylabel('Normalized Flux', fontsize=16)
	ax[0].tick_params(labelsize=14)
	ax[0].grid(True, alpha=0.8)
	ax[0].legend(fontsize=12)

	ax[1].errorbar(times[use_inds], corr_targ_flux[use_inds],corr_targ_flux_err[use_inds], marker='.', color='#b0b0b0', ls='', label='Corr. targ. flux')
	ax[1].errorbar(bx, by, bye, marker='o', color='k', ecolor='k', ls='', ms=7, mfc='none', zorder=4, mew=1.4, label=f'{bin_mins}-min binned flux')
	ax[1].grid(alpha=0.8)
	ax[1].tick_params(labelsize=14)
	ax[1].legend(fontsize=12)
	ax[1].set_xlabel(f'Time - {x_offset}'+' (BJD$_{TDB}$)', fontsize=14)

	plt.tight_layout()
	output_path = f'/data/tierras/lightcurves/{date}/{target}/{ffname}/{date}_{target}_lc.png'
	plt.savefig(output_path,dpi=300)
	set_tierras_permissions(output_path)

	plt.close()
	plt.ion()
	return

def plot_ref_lightcurves(data_dict, sources, bin_mins=10):
	plt.ioff()

	ffname = data_dict['ffname']
	target = data_dict['Target']
	date = data_dict['Date']

	output_path = Path(f'/data/tierras/lightcurves/{date}/{target}/{ffname}/reference_lightcurves/')
	if not os.path.exists(output_path):
		os.mkdir(output_path)		
		set_tierras_permissions(output_path)

	#Clear out existing files
	existing_files = glob(str(output_path/'*.png'))
	for file in existing_files:
		os.remove(file)

	# load in the data for the reference stars 
	times_ = data_dict['BJD']
	x_offset = int(np.floor(times_[0]))
	times = times_ - x_offset

	corr_flux = data_dict['Corrected Flux'][:,1:]
	corr_flux_err = data_dict['Corrected Flux Error'][:,1:]
	weights = data_dict['Weights'][:,0]
	n_refs = corr_flux.shape[1]

	for i in range(n_refs):

		print(f'Doing Ref {i+1} of {n_refs}')

		flux = corr_flux[:,i]
		flux_err = corr_flux_err[:,i]
	
		nan_inds = ~np.isnan(flux)
		times_plot = times[nan_inds]
		flux = flux[nan_inds]
		flux_err = flux_err[nan_inds]

		v, l, h = sigmaclip(flux)
		sc_inds = np.where((flux > l) & (flux < h))[0]
		times_plot = times_plot[sc_inds]
		flux = flux[sc_inds]
		flux_err = flux_err[sc_inds]

		#print(f"bp_rp: {targ_and_refs['bp_rp'][i+1]}")
		fig, ax = plt.subplots(1,1,figsize=(10,4),sharex=True)

		bx, by, bye = tierras_binner(times_plot, flux,bin_mins=bin_mins)

		fig.suptitle(f'Reference {i+1} (S{sources.iloc[i+1].name}), Weight={weights[i]:.2g}',fontsize=16)

		ax.errorbar(times_plot, flux, flux_err,  marker='.', ls='', color='#b0b0b0')
		ax.errorbar(bx,by,bye,marker='o',color='none',ecolor='k',mec='k',mew=1.4,ms=7,ls='',label=f'{bin_mins}-min binned photometry',zorder=3)
		ax.set_ylim(l, h)
		ax.tick_params(labelsize=14)
		ax.set_ylabel('Normalized Flux',fontsize=16)
		ax.grid(True, alpha=0.8)
		ax.legend()	
		plt.tight_layout()
		plt.savefig(output_path/f'ref_{i+1}.png',dpi=300)
		set_tierras_permissions(output_path/f'ref_{i+1}.png')

		plt.close()
	plt.ion()
	return

def plot_raw_fluxes(data_dict, sources):

	ffname = data_dict['ffname']
	target = data_dict['Target']
	date = data_dict['Date']

	times_ = np.array(data_dict['BJD'])
	x_offset = int(np.floor(times_[0]))
	times = times_ - x_offset

	targ_flux = data_dict['Flux'][:,0]

	n_refs = data_dict['Flux'].shape[1]-1

	plt.figure(figsize=(10,12))
	plt.plot(times, targ_flux, '.', color='k', label='Targ.')
	plt.text(times[0]-(times[-1]-times[0])/75,targ_flux[0],'Targ.',color='k',ha='right',va='center',fontsize=12)

	#xvals[0] = times[0]
	#markers = ['v','s','p','*','+','x','D','|','X']
	markers = ['.']
	#colors = plt.get_cmap('viridis_r')
	#offset = 0.125
	colors = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:grey','tab:olive','tab:cyan']
	for i in range(n_refs):
		ref_flux = np.array(data_dict['Flux'][:,i+1])
		#ref_flux /= np.median(ref_flux)
		plt.plot(times, ref_flux, marker=markers[i%len(markers)],ls='',label=f'{i+1}',color=colors[i%len(colors)])
		if i % 2 == 0:
			plt.text(times[0]-(times[-1]-times[0])/25,ref_flux[0],f'{i+1}',color=colors[i%len(colors)],ha='right',va='center',fontsize=12)
		else:
			plt.text(times[0]-(times[-1]-times[0])/100,ref_flux[0],f'{i+1}',color=colors[i%len(colors)],ha='right',va='center',fontsize=12)
	#breakpoint()
	plt.yscale('log')
	#plt.legend(loc='center left', bbox_to_anchor=(1,0.5),ncol=2)
	plt.xlim(times[0]-(times[-1]-times[0])/15,times[-1]+(times[-1]-times[0])/100)

	plt.ylabel('Flux (ADU)',fontsize=14)
	plt.xlabel(f'Time - {x_offset}'+' (BJD$_{TDB}$)',fontsize=14)
	plt.tick_params(labelsize=14)
	plt.tight_layout()

	breakpoint()
	output_path = f'/data/tierras/lightcurves/{date}/{target}/{ffname}/{date}_{target}_raw_flux.png'
	plt.savefig(output_path,dpi=300)
	set_tierras_permissions(output_path)
	plt.close()

	return 

def make_lc_path(lc_path):
	lc_path = Path(lc_path)
	if not os.path.exists(lc_path.parent.parent):
		os.mkdir(lc_path.parent.parent)
		set_tierras_permissions(lc_path.parent.parent)
	if not os.path.exists(lc_path.parent):
		os.mkdir(lc_path.parent)
		set_tierras_permissions(lc_path.parent)
	if not os.path.exists(lc_path):
		os.mkdir(lc_path)
		set_tierras_permissions(lc_path)

def identify_target_gaia_id(target, sources=None, x_pix=None, y_pix=None):
	
	if 'Gaia DR3' in target:
		gaia_id = int(target.split('Gaia DR3 ')[1])
		return gaia_id

	if (x_pix is not None) and (y_pix) is not None:
		x = np.array(sources['X pix'])
		y = np.array(sources['Y pix'])
		dists = ((x-x_pix)**2+(y-y_pix)**2)**0.5
		return sources['source_id'][np.argmin(dists)] 
	
	objids = Simbad.query_objectids(target)
	for i in range(len(objids)):
		if 'Gaia DR3' in str(objids[i]).split('\n')[-1]:
			gaia_id = int(str(objids[i]).split('DR3 ')[1])
			break		
	return gaia_id

def main(raw_args=None):
	ap = argparse.ArgumentParser()
	ap.add_argument("-date", required=True, help="Date of observation in YYYYMMDD format.")
	ap.add_argument("-field", required=True, help="Name of observed target field exactly as shown in raw FITS files.")
	ap.add_argument("-target", required=False, default=None, help="Specifier for the target in the field to be analyzed. Can be a Gaia DR3 source id (e.g.: 'Gaia DR3 3758629475341196672'), a 2MASS ID (e.g.: '2MASS J10582800-1046304'), or a string of coordinates enclosed by parentheses (e.g. (10:58:28.0 -10:46:58.3) or (164.616667 -10.775138)'. If no argument is passed, the program defaults to using the target field name as the target.")
	ap.add_argument("-ffname", required=False, default='flat0000', help="Name of folder in which to store reduced+flattened data. Convention is flatXXXX. XXXX=0000 means no flat was used.")
	ap.add_argument("-overwrite", required=False, default=False, help="Whether or not to overwrite existing lc files.")
	ap.add_argument("-email", required=False, default=False, help="Whether or not to send email with summary plots.")
	ap.add_argument("-plot", required=False, default=False, help="Whether or not to generate a summary plot to the target's /data/tierras/targets directory")
	ap.add_argument("-target_x_pix", required=False, default=None, help="x pixel position of the Tierras target in the field", type=float)
	ap.add_argument("-target_y_pix", required=False, default=None, help="y pixel position of the Tierras target in the field", type=float)
	ap.add_argument("-SAME", required=False, default=False, help="whether or not to run SAME analysis")

	args = ap.parse_args(raw_args)

	#Access observation info
	date = args.date
	field = args.field
	ffname = args.ffname
	email = t_or_f(args.email)
	plot = t_or_f(args.plot)
	targ_x_pix = args.target_x_pix
	targ_y_pix = args.target_y_pix
	same = t_or_f(args.SAME)

	if args.target is None: 
		target = field 
	else:
		target = args.target


	lc_path = Path(f'/data/tierras/lightcurves/{date}/{field}/{ffname}')
	if not os.path.exists(lc_path):
		make_lc_path(lc_path)	

	if args.target is None: 
		target = field 
	else:
		target = args.target

	date_path = f'/data/tierras/photometry/{date}/{field}/{ffname}'
	phot_files = glob(date_path+'/**.parquet')
	n_ims = len(pq.read_table(phot_files[0]))
	n_dfs = len(phot_files)

	source_df = pd.read_csv(date_path+f'/{date}_{field}_sources.csv')
	source_ids = np.array(list(source_df['source_id']))
	n_sources = len(source_ids)

	times = np.zeros(n_ims, dtype='float64')
	airmasses = np.zeros(n_ims, dtype='float16')
	exposure_times = np.zeros(n_ims, dtype='float16')
	filenames = np.zeros(n_ims, dtype='str')
	ha = np.zeros(n_ims, dtype='float16')
	humidity = np.zeros(n_ims, dtype='float16')
	fwhm_x = np.zeros(n_ims, dtype='float16')
	fwhm_y = np.zeros(n_ims, dtype='float16')
	flux = np.zeros((n_dfs, n_ims, n_sources), dtype='float32')
	flux_err = np.zeros_like(flux)
	non_linear_flags = np.zeros_like(flux, dtype='bool')
	saturated_flags = np.zeros_like(flux, dtype='bool')
	x = np.zeros((n_ims, n_sources), dtype='float32')
	y = np.zeros_like(x)
	sky = np.zeros_like(x)

	times_list = []
	start = 0

	t1 = time.time()

	ancillary_cols = ['Filename', 'BJD TDB', 'Airmass', 'Exposure Time', 'HA', 'Dome Humid', 'FWHM X', 'FWHM Y']

	print(f'Reading in photometry from {date_path}.')

	ancillary_file = glob(date_path+'/**ancillary**.parquet')
	phot_files = glob(date_path+'/**phot**.parquet')
	if len(phot_files) == 0:
		raise RuntimeError(f'No photometry files for {field} on {date}! Run ap_phot.py first.')
	phot_files = sorted(phot_files, key=lambda x:float(x.split('_')[-1].split('.')[0])) # sort on aperture size so everything is in order

	n_dfs = len(phot_files)
	breakpoint()
	ancillary_tab = pq.read_table(ancillary_file, columns=ancillary_cols)

	source_inds = np.arange(len(source_ids))
	for j in range(n_dfs):
		# only read in the necessary columns from the data files to save a lot of time in very crowded fields
		use_cols = []
		for k in range(len(source_inds)):
			use_cols.append(f'S{source_inds[k]} Source-Sky')
			use_cols.append(f'S{source_inds[k]} Source-Sky Err')
			use_cols.append(f'S{source_inds[k]} NL Flag')
			use_cols.append(f'S{source_inds[k]} Sat Flag')
			use_cols.append(f'S{source_inds[k]} Sky')
			use_cols.append(f'S{source_inds[k]} X')
			use_cols.append(f'S{source_inds[k]} Y')

		data_tab = pq.read_table(phot_files[j], columns=use_cols, memory_map=True)
		stop = start+len(data_tab)

		times[start:stop] = np.array(ancillary_tab['BJD TDB'])
		if j == 0:
			times_list.append(times[start:stop])
		airmasses[start:stop] = np.array(ancillary_tab['Airmass'])
		exposure_times[start:stop] = np.array(ancillary_tab['Exposure Time'])
		filenames[start:stop] = np.array(ancillary_tab['Filename'])
		ha[start:stop] = np.array(ancillary_tab['HA'])
		humidity[start:stop] = np.array(ancillary_tab['Dome Humid'])
		fwhm_x[start:stop] = np.array(ancillary_tab['FWHM X'])
		fwhm_y[start:stop] = np.array(ancillary_tab['FWHM Y'])
		
		for k in range(n_sources):
			flux[j,start:stop,k] = np.array(data_tab[f'S{source_inds[k]} Source-Sky'])
			flux_err[j,start:stop,k] = np.array(data_tab[f'S{source_inds[k]} Source-Sky Err'])
			non_linear_flags[j,start:stop,k] = np.array(data_tab[f'S{source_inds[k]} NL Flag'])
			saturated_flags[j,start:stop,k] = np.array(data_tab[f'S{source_inds[k]} Sat Flag'])
			if j == 0:
				x[start:stop,k] = np.array(data_tab[f'S{source_inds[k]} X'])
				y[start:stop,k] = np.array(data_tab[f'S{source_inds[k]} Y'])
				sky[start:stop,k] = np.array(data_tab[f'S{source_inds[k]} Sky'])

	start = stop
	print(f'Read-in: {time.time()-t1}')

	x_offset = int(np.floor(times[0]))
	times -= x_offset

	x_deviations = np.median(x - np.nanmedian(x, axis=0), axis=1)
	y_deviations = np.median(y - np.nanmedian(y, axis=0), axis=1)
	median_sky = np.median(sky, axis=1)/exposure_times
	median_flux = np.nanmedian(flux[0],axis=1)/np.nanmedian(np.nanmedian(flux[0],axis=1))

	# optionally restrict to SAME
	if same: 
		same_mask = np.zeros(len(fwhm_x), dtype='int')

		fwhm_inds = np.where(fwhm_x > 3)[0]
		pos_inds = np.where((abs(x_deviations) > 2.5) | (abs(y_deviations) > 2.5))[0]
		airmass_inds = np.where(airmasses > 2.0)[0]
		sky_inds = np.where(median_sky > 5)[0]
		# humidity_inds = np.where(humidity > 50)[0]
		flux_inds = np.where(median_flux < 0.5)[0]
		same_mask[fwhm_inds] = 1
		same_mask[pos_inds] = 1
		same_mask[airmass_inds] = 1
		same_mask[sky_inds] = 1
		# same_mask[humidity_inds] = 1
		same_mask[flux_inds] = 1
		breakpoint()	
		print(f'Restrictions cut to {len(np.where(same_mask == 0)[0])} SAME exposures out of {len(x_deviations)} total exposures.')
		mask = same_mask==1

		# times[mask] = np.nan
		airmasses[mask] = np.nan
		exposure_times[mask] = np.nan
		filenames[mask] = np.nan
		ha[mask] = np.nan
		humidity[mask] = np.nan
		fwhm_x[mask] = np.nan 
		fwhm_y[mask] = np.nan 
		flux[:,mask,:] = np.nan 
		flux_err[:,mask,:] = np.nan 
		x[mask,:] = np.nan 
		y[mask,:] = np.nan 
		sky[mask,:] = np.nan

	# determine maximum time range covered by all of the nights, we'll need this for plotting 
	time_deltas = [i[-1]-i[0] for i in times_list]
	x_range = np.nanmax(time_deltas)

	try:
		gaia_id_file = f'/data/tierras/fields/{field}/{field}_gaia_dr3_id.txt'
		with open(gaia_id_file, 'r') as f:
			tierras_target_id = f.readline()
		tierras_target_id = int(tierras_target_id.split(' ')[-1])
		# tierras_target_id = identify_target_gaia_id(field, source_df, x_pix=targ_x_pix, y_pix=targ_y_pix)
	except:
		raise RuntimeError('Could not identify Gaia DR3 source_id of Tierras target.')
	
	ppb = 1
	n_bins = int(np.ceil(n_ims/ppb))
	bin_inds = []
	for i in range(n_bins):
		if i != n_bins - 1:
			bin_inds.append(np.arange(i*ppb, (i+1)*ppb))
		else:
			bin_inds.append(np.arange(i*ppb, n_ims))
	bin_inds = np.array(bin_inds)

	binned_times = np.zeros(len(bin_inds))
	binned_airmass = np.zeros_like(binned_times)
	binned_exposure_time = np.zeros_like(binned_times)
	binned_flux = np.zeros((n_dfs, len(bin_inds), n_sources))
	binned_flux_err = np.zeros_like(binned_flux)
	binned_nl_flags = np.zeros_like(binned_flux, dtype='int')
	for i in range(n_bins):
		binned_times[i] = np.mean(times[bin_inds[i]])
		binned_airmass[i] = np.mean(airmasses[bin_inds[i]])
		binned_exposure_time[i] = np.nansum(exposure_times[bin_inds[i]])
		for j in range(n_dfs):
			binned_flux[j,i,:] = np.nanmean(flux[j,bin_inds[i]], axis=0)
			binned_flux_err[j,i,:] = np.nanmean(flux_err[j,bin_inds[i]],axis=0)/np.sqrt(len(bin_inds[i]))
			binned_nl_flags[j,i,np.sum(non_linear_flags[j,bin_inds[i]], axis=0)>1] = 1

	# start_ind = np.where(common_source_ids == 4147120983934854400)[0][0]
	
	avg_mearth_times = np.zeros(n_sources)

	# choose a set of references and generate weights
	max_refs = 1000
	if len(source_ids) > max_refs:
		ref_gaia_ids = source_ids[0:max_refs]
		ref_inds = np.arange(max_refs)
	else:
		ref_gaia_ids = source_ids
		ref_inds = np.arange(len(source_ids))

	print(f'Weighting {len(ref_inds)} reference stars...')
	weights_arr = np.zeros((len(ref_inds), n_dfs))
	for i in range(n_dfs):
		print(f'{i+1} of {n_dfs}')
		flux_arr = binned_flux[i][:,ref_inds]
		flux_err_arr = binned_flux_err[i][:,ref_inds]
		nl_flag_arr = binned_nl_flags[i][:,ref_inds]
		weights, mask = mearth_style_pat_weighted_flux(flux_arr, flux_err_arr, nl_flag_arr, binned_airmass, binned_exposure_time, source_ids=ref_gaia_ids)
		weights_arr[:, i] = weights

	# save a csv with the weights 
	weights_dict = {'Ref ID':source_ids[ref_inds]}
	for i in range(n_dfs):
		ap_size = phot_files[i].split('/')[-1].split('_')[-1].split('.')[0]
		weights_dict[ap_size] = weights_arr[:,i]
	weights_df = pd.DataFrame(weights_dict)
	weights_df.to_csv(f'/data/tierras/lightcurves/{date}/{field}/{ffname}/{date}_{field}_weights.csv', index=0)
	set_tierras_permissions(f'/data/tierras/lightcurves/{date}/{field}/{ffname}/{date}_{field}_weights.csv')
	
	ppb = 5 # set up bins for evaluating stddevs of lcs
	n_bins = int(np.ceil(n_ims/ppb))
	bin_inds = []
	for i in range(n_bins):
		if i != n_bins - 1:
			bin_inds.append(np.arange(i*ppb, (i+1)*ppb))
		else:
			bin_inds.append(np.arange(i*ppb, n_ims))
	bin_inds = np.array(bin_inds)

	for tt in range(len(source_ids)):
		tloop = time.time()
		
		if source_ids[tt] == tierras_target_id:
			target = field
			target_gaia_id = tierras_target_id 
			plot = True
		else:
			target = 'Gaia DR3 '+str(source_ids[tt])
			target_gaia_id = identify_target_gaia_id(target)
			plot = False 
		
		print(f'Doing {target}, source {tt+1} of {len(source_ids)}.')
		# ref_gaia_ids = ref_selection(target_gaia_id, source_dfs[0], common_source_ids, max_refs=100)
		# inds = np.where(target_gaia_id == common_source_ids)[0]
		# for i in range(len(ref_gaia_ids)):
		# 	inds = np.append(inds, np.where(ref_gaia_ids[i] == common_source_ids)[0])

		med_stddevs = np.zeros(n_dfs)
		best_med_stddev = 9999999.
		mearth_style_times = np.zeros(n_dfs)
		best_corr_flux = None
		for i in range(n_dfs):			
			# tmearth = time.time()
			# flux_arr = binned_flux[i][:,inds]
			# flux_err_arr = binned_flux_err[i][:,inds]
			# nl_flag_arr = binned_nl_flags[i][:,inds]
			# weights, mask = mearth_style_pat_weighted_flux(flux_arr, flux_err_arr, nl_flag_arr, binned_airmass, binned_exposure_time)
			
			# if the target is one of the reference stars, set its ALC weight to zero and re-weight all the other stars
			if target_gaia_id in ref_gaia_ids:
				if i == 0:
					weight_ind = np.where(ref_gaia_ids == target_gaia_id)[0][0]
				weights = copy.deepcopy(weights_arr[:,i])
				weights[weight_ind] = 0
				weights /= np.nansum(weights)
			else:
				weights = copy.deepcopy(weights_arr[:,i])

			# mearth_style_times[i] = time.time()-tmearth
				
			mask = np.zeros(len(flux[i][:,tt]), dtype='int') 
			mask[np.where(saturated_flags[i][:,tt])] = True # mask out any saturated exposures for this source
			if sum(mask) == len(mask):
				print(f'All exposures saturated for {target_gaia_id}, skipping!')
				break

			# use the weights calculated in mearth_style to create the ALC 
			alc = np.matmul(flux[i][:,ref_inds], weights)
			alc_err = np.sqrt(np.matmul(flux_err[i][:,ref_inds]**2, weights**2))

			# correct the target flux by the ALC and incorporate the ALC error into the corrected flux error
			target_flux = copy.deepcopy(flux[i][:,tt])
			target_flux_err = copy.deepcopy(flux_err[i][:,tt])
			target_flux[mask] = np.nan
			target_flux_err[mask] = np.nan
			corr_flux = target_flux / alc
			rel_flux_err = np.sqrt((target_flux_err/alc)**2 + (target_flux*alc_err/(alc**2))**2)
			
			# normalize
			norm = np.nanmedian(corr_flux)
			corr_flux /= norm 
			rel_flux_err /= norm

			# inflate error by scintillation estimate (see Stefansson et al. 2017)
			n_refs = len(np.where(weights != 0)[0])
			sigma_s = 0.09*130**(-2/3)*airmasses**(7/4)*(2*exposure_times)**(-1/2)*np.exp(-2306/8000)
			sigma_scint = 1.5*sigma_s*np.sqrt(1 + 1/(n_refs))
			corr_flux_err = np.sqrt(rel_flux_err**2 + sigma_scint**2)

			# sigma clip
			corr_flux_sc = sigma_clip(corr_flux).data
			norm = np.nanmedian(corr_flux_sc)
			corr_flux_sc /= norm 

			# Evaluate the median standard deviation over 5-minute intervals 
			stddevs = np.zeros(len(bin_inds))
			for k in range(len(bin_inds)):
				stddevs[k] = np.nanstd(corr_flux_sc[bin_inds[k]])
			
			med_stddevs[i] = np.nanmedian(stddevs)	

			# if this light curve is better than the previous ones, store it for later
			if med_stddevs[i] < best_med_stddev: 
				best_med_stddev = med_stddevs[i]
				best_phot_file = i 
				best_corr_flux = corr_flux
				best_corr_flux_err = corr_flux_err
				best_alc = alc
				best_alc_err = alc_err

		if best_corr_flux is not None:
			
			v, l, h = sigmaclip(best_corr_flux[~np.isnan(best_corr_flux)])
			use_inds = np.where((best_corr_flux >= l) & (best_corr_flux <= h))
			
			bins, std, theo = allen_deviation(times[use_inds], best_corr_flux[use_inds], best_corr_flux_err[use_inds])

			med_std_over_theo = np.nanmedian(std/theo)
			if med_std_over_theo > 10:
				print('Possible variable!')
				plot = False

			# write out the best light curve 
			output_dict = {'BJD TDB':times+x_offset, 'Flux':best_corr_flux, 'Flux Error':best_corr_flux_err}
			output_path = Path(f'/data/tierras/lightcurves/{date}/{field}/{ffname}/')
			if not os.path.exists(output_path.parent.parent):
				os.mkdir(output_path.parent.parent)
				set_tierras_permissions(output_path.parent.parent)
			if not os.path.exists(output_path.parent):
				os.mkdir(output_path.parent)
				set_tierras_permissions(output_path.parent)
			if not os.path.exists(output_path):
				os.mkdir(output_path)
				set_tierras_permissions(output_path)

			best_phot_style = phot_files[best_phot_file].split(f'{field}_')[1].split('.csv')[0]

			if os.path.exists(output_path/f'{date}_{target}_lc.csv'):
				os.remove(output_path/f'{date}_{target}_lc.csv')
			
			filename = open(output_path/f'{date}_{target}_lc.csv', 'a')
			filename.write(f'# this light curve was made using {best_phot_style}\n' )
			output_df = pd.DataFrame(output_dict)
			output_df.to_csv(filename, index=0, na_rep=np.nan)
			filename.close()
			set_tierras_permissions(output_path/f'{date}_{target}_lc.csv')
			
			if plot: 
				data_dict = {'Target':target, 'Date':date, 'ffname':ffname, 'BJD':times+x_offset, 'Flux':flux[best_phot_file][:,tt], 
				 'Flux Error':flux_err[best_phot_file][:,tt], 'ALC':best_alc, 'ALC Error':best_alc_err, 'Corrected Flux':best_corr_flux,
				 'Corrected Flux Error':best_corr_flux_err, 'Airmass':airmasses, 'Hour Angle':ha, 'Sky':sky[:,tt], 'X':x[:,tt], 'Y':y[:,tt], 
				 'FWHM X':fwhm_x, 'FWHM Y':fwhm_y, 'Dome Humidity':humidity, 'Exposure Time':exposure_times}
				
				plot_target_summary(data_dict)
			gc.collect() # do garbage collection to prevent memory leaks 
			print(f'tloop: {time.time()-tloop:.1f}')

			
	
	if email: 
		# Send summary plots 
		subject = f'[Tierras]_Data_Analysis_Report:{date}_{field}'
		summary_path = f'/data/tierras/lightcurves/{date}/{field}/{ffname}/{date}_{field}_summary.png'
		lc_path = f'/data/tierras/lightcurves/{date}/{field}/{ffname}/{date}_{field}_lc.png' 
		append = '{} {}'.format(summary_path,lc_path)
		emails = 'juliana.garcia-mejia@cfa.harvard.edu patrick.tamburo@cfa.harvard.edu'
		os.system('echo | mutt {} -s {} -a {}'.format(emails,subject,append))

if __name__ == '__main__':
	main()