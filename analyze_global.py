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
from matplotlib.ticker import ScalarFormatter
from pathlib import Path 
from scipy.optimize import curve_fit
import pickle
from analyze_night import mearth_style_pat_weighted_flux, allen_deviation
import time 
import warnings 
import gc 
import csv
import pyarrow as pa 
import pyarrow.parquet as pq 
from astropy.stats import sigma_clip

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

def ref_selection(target_gaia_id, sources, common_source_ids, delta_target_rp=5, target_distance_limit=4000, max_refs=50):

	target_ind = np.where(sources['source_id'] == target_gaia_id)[0][0]

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

	ref_gaia_ids = sources['source_id'][ref_inds]
	# refs = sources.iloc[ref_inds]
	return np.array(ref_gaia_ids)

def main(raw_args=None):
	warnings.filterwarnings('ignore')

	ap = argparse.ArgumentParser()
	ap.add_argument("-field", required=True, help="Name of observed target field exactly as shown in raw FITS files.")
	ap.add_argument("-target", required=False, default=None, help="Specifier for the target in the field to be analyzed. Can be a Gaia DR3 source id (e.g.: 'Gaia DR3 3758629475341196672'), a 2MASS ID (e.g.: '2MASS J10582800-1046304'), or a string of coordinates enclosed by parentheses (e.g. (10:58:28.0 -10:46:58.3) or (164.616667 -10.775138)'. If no argument is passed, the program defaults to using the target field name as the target.")
	ap.add_argument("-ffname", required=False, default='flat0000', help="Name of folder in which to store reduced+flattened data. Convention is flatXXXX. XXXX=0000 means no flat was used.")
	ap.add_argument("-email", required=False, default=False, help="Whether or not to send email with summary plots.")
	ap.add_argument("-plot", required=False, default=False, help="Whether or not to generate a summary plot to the target's /data/tierras/targets directory")
	ap.add_argument("-target_x_pix", required=False, default=None, help="x pixel position of the Tierras target in the field", type=float)
	ap.add_argument("-target_y_pix", required=False, default=None, help="y pixel position of the Tierras target in the field", type=float)
	ap.add_argument("-SAME", required=False, default=False, help="whether or not to run SAME analysis")
	ap.add_argument("-use_nights", required=False, default=None, help="Nights to be included in the analysis. Format as chronological comma-separated list, e.g.: 20240523, 20240524, 20240527")
	ap.add_argument("-minimum_night_duration", required=False, default=1, help="Minimum cumulative exposure time on a given night (in hours) that a night has to have in order to be retained in the analysis (AFTER SAME RESTRICTIONS HAVE BEEN APPLIED).", type=float)

	args = ap.parse_args(raw_args)

	#Access observation info
	field = args.field
	ffname = args.ffname
	email = t_or_f(args.email)
	plot = t_or_f(args.plot)
	targ_x_pix = args.target_x_pix
	targ_y_pix = args.target_y_pix
	same = t_or_f(args.SAME)
	minimum_night_duration = args.minimum_night_duration

	if args.target is None: 
		target = field 
	else:
		target = args.target

	# identify dates on which this field was observed 
	date_list = glob(f'/data/tierras/photometry/**/{field}/{ffname}')	
	date_list = np.array(sorted(date_list, key=lambda x:int(x.split('/')[4]))) # sort on date so that everything is in order
	
	# if a set of use_nights has been specified, cut the date list to match 
	if args.use_nights is not None:
		use_nights = args.use_nights.replace(' ','').split(',')
		dates_to_keep = []
		for i in range(len(use_nights)):
			for j in range(len(date_list)):
				if use_nights[i] in date_list[j]:
					dates_to_keep.append(j)
					break
		date_list = date_list[dates_to_keep]

	# date_list = date_list[0:7]
	dates = np.array([i.split('/')[4] for i in date_list])

	# skip some nights of processing for TIC362144730
	# TODO: automate this process 
	if field == 'TIC362144730':
		bad_dates = ['20240519', '20240520', '20240531', '20240607', '20240609', '20240610', '20240706', '20240707', '20240708']
		dates_to_remove = []
		for i in range(len(bad_dates)):
			dates_to_remove.append(np.where(dates == bad_dates[i])[0][0])
		dates = np.delete(dates, dates_to_remove)
		date_list = np.delete(date_list, dates_to_remove)		

	if field == 'LHS2919':
		bad_dates = ['20240530', '20240531', '20240601', '20240602']
		dates_to_remove = []
		for i in range(len(bad_dates)):
			dates_to_remove.append(np.where(dates == bad_dates[i])[0][0])
		dates = np.delete(dates, dates_to_remove)
		date_list = np.delete(date_list, dates_to_remove)		

	# read in the source df's from each night 
	source_dfs = []
	source_ids = []
	for i in range(len(date_list)):
		source_file = glob(date_list[i]+'/**sources.csv')[0]
		source_dfs.append(pd.read_csv(source_file))	
		source_ids.append(list(source_dfs[i]['source_id']))

	# determine the Gaia ID's of sources that were observed on every night
	# initialize using the first night
	common_source_ids = np.array(source_ids[0])

	# remove sources if they don't have photometry on all other nights
	inds_to_remove = []
	for i in range(len(common_source_ids)):
		for j in source_ids[1:]:
			if common_source_ids[i] not in j:
				inds_to_remove.append(i)
	if len(inds_to_remove) > 0:
		inds_to_remove = np.array(inds_to_remove)
		common_source_ids = np.delete(common_source_ids, inds_to_remove)

	
	# get the index mapping between the source dfs and the photometry source names
	source_inds = []
	for i in range(len(source_dfs)):
		source_inds.append([])
		for j in range(len(common_source_ids)):
			source_inds[i].extend([np.where(source_dfs[i]['source_id'] == common_source_ids[j])[0][0]])
	
	#TODO: what happens when there are different sources on different nights (i.e., if we shifted the field??)
	# figure out how big the arrays need to be to hold the data
	# NOTE this assumes that every directory has the exact same photometry files...	
	n_sources = len(common_source_ids)
	n_ims = 0 
	for i in range(len(date_list)):
		phot_files = glob(date_list[i]+'/**phot**.parquet')
		n_dfs = len(phot_files)
		if len(phot_files) != 0:
			n_ims += len(pq.read_table(phot_files[0]))

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

	for i in range(len(date_list)):
		print(f'Reading in photometry from {date_list[i]} (date {i+1} of {len(date_list)}).')
		ancillary_file = glob(date_list[i]+'/**ancillary**.parquet')
		phot_files = glob(date_list[i]+'/**phot**.parquet')
		if len(phot_files) == 0:
			continue 
		phot_files = sorted(phot_files, key=lambda x:float(x.split('_')[-1].split('.')[0])) # sort on aperture size so everything is in order

		n_dfs = len(phot_files)

		ancillary_tab = pq.read_table(ancillary_file, columns=ancillary_cols)

		for j in range(n_dfs):
			# only read in the necessary columns from the data files to save a lot of time in very crowded fields
			use_cols = []
			for k in range(len(source_inds[i])):
				use_cols.append(f'S{source_inds[i][k]} Source-Sky')
				use_cols.append(f'S{source_inds[i][k]} Source-Sky Err')
				use_cols.append(f'S{source_inds[i][k]} NL Flag')
				use_cols.append(f'S{source_inds[i][k]} Sat Flag')
				use_cols.append(f'S{source_inds[i][k]} Sky')
				use_cols.append(f'S{source_inds[i][k]} X')
				use_cols.append(f'S{source_inds[i][k]} Y')
				# use_cols.append(f'S{source_inds[i][k]} X FWHM')
				# use_cols.append(f'S{source_inds[i][k]} Y FWHM')

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
				flux[j,start:stop,k] = np.array(data_tab[f'S{source_inds[i][k]} Source-Sky'])
				flux_err[j,start:stop,k] = np.array(data_tab[f'S{source_inds[i][k]} Source-Sky Err'])
				non_linear_flags[j,start:stop,k] = np.array(data_tab[f'S{source_inds[i][k]} NL Flag'])
				saturated_flags[j,start:stop,k] = np.array(data_tab[f'S{source_inds[i][k]} Sat Flag'])
				if j == 0:
					x[start:stop,k] = np.array(data_tab[f'S{source_inds[i][k]} X'])
					y[start:stop,k] = np.array(data_tab[f'S{source_inds[i][k]} Y'])
					sky[start:stop,k] = np.array(data_tab[f'S{source_inds[i][k]} Sky'])

		start = stop
	print(f'Read-in: {time.time()-t1}')

	x_offset = int(np.floor(times[0]))
	times -= x_offset

	x_deviations = np.median(x - np.nanmedian(x, axis=0), axis=1)
	y_deviations = np.median(y - np.nanmedian(y, axis=0), axis=1)
	median_sky = np.median(sky, axis=1)/exposure_times
	median_flux = np.nanmedian(flux[0],axis=1)/np.nanmedian(np.nanmedian(flux[0],axis=1))
	
	# optionally restrict to SAME 
	same_mask = np.zeros(len(fwhm_x), dtype='int')
	if same: 

		fwhm_inds = np.where(fwhm_x > 3.0)[0]
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

	# determine if nights should be dropped based on minimum time duration criterion 
	dates_to_remove = []
	night_inds_list = []
	for i in range(len(times_list)):
		night_inds = np.where((times >= times_list[i][0]) & (times <= times_list[i][-1]))[0]
		night_inds_list.append(night_inds)
		tot_exposure_time = np.nansum(exposure_times[night_inds])/3600
		if tot_exposure_time <= minimum_night_duration:
			same_mask[night_inds] = 1

			print(f'{dates[i]} dropped! Less than {minimum_night_duration} hours of exposures.')
			dates_to_remove.append(i)
	
	dates = np.delete(dates, dates_to_remove)
	date_list = np.delete(date_list, dates_to_remove)		
	times_list = np.delete(times_list, dates_to_remove)
	night_inds_list = np.delete(night_inds_list, dates_to_remove)
	mask = same_mask==1
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
	print(f'Quality restrictions cut to {len(np.where(same_mask == 0)[0])} exposures out of {len(x_deviations)} total exposures. Data are on {len(dates)} nights.')

	breakpoint()
	# determine maximum time range covered by all of the nights, we'll need this for plotting 
	time_deltas = [i[-1]-i[0] for i in times_list]
	x_range = np.nanmax(time_deltas)

	try:
		gaia_id_file = f'/data/tierras/fields/{field}/{field}_gaia_dr3_id.txt'
		with open(gaia_id_file, 'r') as f:
			tierras_target_id = f.readline()
		tierras_target_id = int(tierras_target_id.split(' ')[-1])
		# tierras_target_id = identify_target_gaia_id(field, source_dfs[0], x_pix=targ_x_pix, y_pix=targ_y_pix)
	except:
		raise RuntimeError('Could not identify Gaia DR3 source_id of Tierras target.')

	# # NOTE: Uncomment these lines if you want to do the weighting on nightly medians only. 
	# binned_times = np.zeros(len(night_inds_list))
	# binned_airmass = np.zeros_like(binned_times)
	# binned_exposure_time = np.zeros_like(binned_times)
	# binned_flux = np.zeros((n_dfs, len(night_inds_list), n_sources))
	# binned_flux_err = np.zeros_like(binned_flux)
	# binned_nl_flags = np.zeros_like(binned_flux, dtype='int')

	# for i in range(len(night_inds_list)):
	# 	binned_times[i] = np.mean(times[night_inds_list[i]])
	# 	binned_airmass[i] = np.nanmean(airmasses[night_inds_list[i]])
	# 	binned_exposure_time[i] = np.nansum(exposure_times[night_inds_list[i]])
	# 	for j in range(n_dfs):
	# 		binned_flux[j,i,:] = np.nanmedian(flux[j,night_inds_list[i]], axis=0)
	# 		binned_flux_err[j,i,:] = np.nanmean(flux_err[j,night_inds_list[i]],axis=0)/np.sqrt(len(night_inds_list[i]))
	# 		binned_nl_flags[j,i,np.sum(non_linear_flags[j,night_inds_list[i]], axis=0)>1] = 1

	ppb = 5 # TODO: how do we get 5-minute bins in the general case where exposure time is changing 
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

	
	avg_mearth_times = np.zeros(n_sources)

	# choose a set of references and generate weights
	max_refs = 1000
	# max_refs = n_sources
	if len(common_source_ids) > max_refs:
		ref_gaia_ids = common_source_ids[0:max_refs]
		ref_inds = np.arange(max_refs)
	else:
		ref_gaia_ids = common_source_ids
		ref_inds = np.arange(len(common_source_ids))

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
	weights_dict = {'Ref ID':common_source_ids[ref_inds]}
	for i in range(n_dfs):
		ap_size = phot_files[i].split('/')[-1].split('_')[-1].split('.')[0]
		weights_dict[ap_size] = weights_arr[:,i]
	weights_df = pd.DataFrame(weights_dict)
	if not os.path.exists(f'/data/tierras/fields/{field}/sources'):
		os.mkdir(f'/data/tierras/fields/{field}/sources')
		set_tierras_permissions(f'/data/tierras/fields/{field}/sources')
	weights_df.to_csv(f'/data/tierras/fields/{field}/sources/weights.csv', index=0)
	set_tierras_permissions(f'/data/tierras/fields/{field}/sources/weights.csv')
	
	for tt in range(len(common_source_ids)):
		tloop = time.time()
		
		if common_source_ids[tt] == tierras_target_id:
			target = field
			target_gaia_id = tierras_target_id
			plot = True	
		else:
			target = 'Gaia DR3 '+str(common_source_ids[tt])
			target_gaia_id = identify_target_gaia_id(target)
			plot = False 
		
		print(f'Doing {target}, source {tt+1} of {len(common_source_ids)}.')
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


			# do a plot if this is the tierras target 

			if plot:
				plt.ioff()

				use_inds = ~np.isnan(best_corr_flux)

				# raw_flux = flux[best_phot_file,:,tt]

				v, l, h = sigmaclip(best_corr_flux[~np.isnan(best_corr_flux)])
				use_inds = np.where((best_corr_flux>l)&(best_corr_flux<h))[0]

				fig = plt.figure(figsize=(24/6*len(date_list), 20))	
				gs = gridspec.GridSpec(8,len(date_list),height_ratios=[2,2,1,1,1,1,1,2], figure=fig)

				ax1 = fig.add_subplot(gs[0,:]) # linear time light curve 					

				# times -= x_offset

				ax1.plot(times[use_inds], best_corr_flux[use_inds], marker='.', color='#b0b0b0', ls='')
				ax1.set_ylabel('Corr. Flux', fontsize=14)
				ax1.set_xlabel(f'BJD - {x_offset:.0f}', fontsize=14)
				ax1.set_xlim(times[use_inds[0]]-1/24, times[use_inds[-1]]+1/24)

				for i in range(len(date_list)):
					use_inds_night = np.where((times >= times_list[i][0]) & (times <= times_list[i][-1]))[0]
					
					if len(date_list) == 1:
						ax2 = fig.add_subplot(gs[1]) 
						ax3 = fig.add_subplot(gs[2])
						ax4 = fig.add_subplot(gs[3])
						ax5 = fig.add_subplot(gs[4])
						ax6 = fig.add_subplot(gs[5])
						ax7 = fig.add_subplot(gs[6])
						ax8 = fig.add_subplot(gs[7])
					else:
						ax2 = fig.add_subplot(gs[1,i])
						ax3 = fig.add_subplot(gs[2,i])
						ax4 = fig.add_subplot(gs[3,i])
						ax5 = fig.add_subplot(gs[4,i])
						ax6 = fig.add_subplot(gs[5,i])
						ax7 = fig.add_subplot(gs[6,i])
						ax8 = fig.add_subplot(gs[7,i])


					
					try:
						allen_inds = np.where((times>=times_list[i][0])&(times<=times_list[i][-1]))[0]
					except:
						continue

					ax2.set_title(date_list[i].split('/')[4], fontsize=14)
					# ax[0].plot(times[use_inds], raw_flux[use_inds]/np.nanmedian(raw_flux[use_inds]), 'k.', label='Target')
					# ax[0].plot(times[use_inds], best_alc[use_inds]/np.nanmedian(best_alc[use_inds]), 'r.', label='ALC')

					
					ax2.errorbar(times[use_inds], best_corr_flux[use_inds], best_corr_flux_err[use_inds], marker='.', color='#b0b0b0', ls='')

					# plot nightly median 
					night_times = times[use_inds_night]
					night_flux = best_corr_flux[use_inds_night]
					nan_inds = ~np.isnan(night_flux)
					night_times = night_times[nan_inds]
					night_flux = night_flux[nan_inds]
					v, l, h = sigmaclip(night_flux)
					night_inds = np.where((night_flux > l) & (night_flux < h))[0]

					ax1.errorbar(np.nanmedian(night_times[night_inds]), np.nanmean(night_flux[night_inds]), 1.2533*np.nanstd(night_flux[night_inds])/np.sqrt(len(night_inds)), marker='o', color='k', ecolor='k', zorder=4, ls='')

					ax2.errorbar(np.nanmedian(night_times[night_inds]), np.nanmean(night_flux[night_inds]), 1.2533*np.nanstd(night_flux[night_inds])/np.sqrt(len(night_inds)), marker='o', color='k', ecolor='k', zorder=4, ls='')

					ax3.plot(times, airmasses, marker='.', ls='')	
					# ax_ha = ax[2].twinx()
					# ax_ha.plot(times, ha, color='tab:orange')

					ax4.plot(times, sky[:,0]/exposure_times, color='tab:cyan', marker='.', ls='')

					ax5.plot(times, x[:,0]-np.nanmedian(x[:,0]), color='tab:green',label='X-med(X)', marker='.', ls='')
					ax5.plot(times, y[:,0]-np.nanmedian(y[:,0]), color='tab:red',label='Y-med(Y)', marker='.', ls='')
					ax5.set_ylim(-15,15)

					ax6.plot(times, fwhm_x, color='tab:pink', label='X', marker='.', ls='')
					ax6.plot(times, fwhm_y, color='tab:purple',label='Y', marker='.', ls='')	
					
					ax7.plot(times, humidity, color='tab:brown', marker='.', ls='')
					
					bins, std, theo = allen_deviation(times[allen_inds], best_corr_flux[allen_inds], best_corr_flux_err[allen_inds])

					ax8.plot(bins, std*1e6, lw=2,label='Measured', marker='.')
					ax8.plot(bins, theo*1e6,lw=2,label='Theoretical', marker='.')
					ax8.set_yscale('log')
					ax8.set_xscale('log')
				
					if i == 0:
						ax2.set_ylabel('Corr. Flux', fontsize=14)
						ax3.set_ylabel('Airmass', fontsize=14)
						ax4.set_ylabel('Sky\n(ADU/s)', fontsize=14)
						ax5.set_ylabel('Pos.\n(pix.)', fontsize=14)
						ax6.set_ylabel('FWHM\n(")', fontsize=14)
						ax7.set_ylabel('Dome Humid.\n(%)',fontsize=14)
						ax8.set_ylabel('$\sigma$ (ppm)', fontsize=14)

					ax2.set_xlim(times_list[i][0], times_list[i][0]+x_range)
					ax3.set_xlim(times_list[i][0], times_list[i][0]+x_range)
					ax4.set_xlim(times_list[i][0], times_list[i][0]+x_range)
					ax5.set_xlim(times_list[i][0], times_list[i][0]+x_range)
					ax6.set_xlim(times_list[i][0], times_list[i][0]+x_range)
					ax7.set_xlim(times_list[i][0], times_list[i][0]+x_range)

					ax2.grid(alpha=0.7)
					ax3.grid(alpha=0.7)
					ax4.grid(alpha=0.7)
					ax5.grid(alpha=0.7)
					ax6.grid(alpha=0.7)
					ax7.grid(alpha=0.7)
					ax8.grid(alpha=0.7)

					ax2.tick_params(labelsize=12, labelbottom=False)
					ax3.tick_params(labelsize=12, labelbottom=False)
					ax4.tick_params(labelsize=12, labelbottom=False)
					ax5.tick_params(labelsize=12, labelbottom=False)
					ax6.tick_params(labelsize=12, labelbottom=False)
					ax7.tick_params(labelsize=12)
					ax8.tick_params(labelsize=12)

					if i > 0 and len(date_list) != 1:
						ax2.spines['left'].set_visible(False)
						ax3.spines['left'].set_visible(False)
						ax4.spines['left'].set_visible(False)
						ax5.spines['left'].set_visible(False)
						ax6.spines['left'].set_visible(False)
						ax7.spines['left'].set_visible(False)
						ax8.spines['left'].set_visible(False)

						ax2.tick_params(labelleft=False)
						ax3.tick_params(labelleft=False)
						ax4.tick_params(labelleft=False)
						ax5.tick_params(labelleft=False)
						ax6.tick_params(labelleft=False)
						ax7.tick_params(labelleft=False)
						ax8.tick_params(labelleft=False)

						ax2.yaxis.tick_left()
						ax3.yaxis.tick_left()
						ax4.yaxis.tick_left()
						ax5.yaxis.tick_left()
						ax6.yaxis.tick_left()
						ax7.yaxis.tick_left()
						ax8.yaxis.tick_left()

					if i < len(date_list) - 1:
						ax2.spines['right'].set_visible(False)
						ax3.spines['right'].set_visible(False)
						ax4.spines['right'].set_visible(False)
						ax5.spines['right'].set_visible(False)
						ax6.spines['right'].set_visible(False)
						ax7.spines['right'].set_visible(False)
						ax8.spines['right'].set_visible(False)
					
						ax2.yaxis.tick_right()
						ax3.yaxis.tick_right()
						ax4.yaxis.tick_right()
						ax5.yaxis.tick_right()
						ax6.yaxis.tick_right()
						ax7.yaxis.tick_right()
						ax8.yaxis.tick_right()
					
					if i == 0:
						ax2.yaxis.tick_left()
						ax3.yaxis.tick_left()
						ax4.yaxis.tick_left()
						ax5.yaxis.tick_left()
						ax6.yaxis.tick_left()
						ax7.yaxis.tick_left()
						ax8.yaxis.tick_left()
					
					if i == len(date_list) - 1:
						ax2.yaxis.tick_right()
						ax3.yaxis.tick_right()
						ax4.yaxis.tick_right()
						ax5.yaxis.tick_right()
						ax6.yaxis.tick_right()
						ax7.yaxis.tick_right()
						ax8.yaxis.tick_right()

						ax2.tick_params(labelright=True)
						ax3.tick_params(labelright=True)
						ax4.tick_params(labelright=True)
						ax5.tick_params(labelright=True)
						ax6.tick_params(labelright=True)
						ax7.tick_params(labelright=True)
						ax8.tick_params(labelright=True)

						ax5.legend()
						ax6.legend()
					
					ax7.set_xlabel(f'BJD - {x_offset:.0f}', fontsize=14)
					ax7.tick_params(labelbottom=True)

					ax8.set_xlabel('Bin size', fontsize=14)	
					ax8.xaxis.set_major_formatter(ScalarFormatter())
				
				# plt.suptitle(target, fontsize=14)
				plt.subplots_adjust(hspace=0.1,wspace=0.05,left=0.05,right=0.92,bottom=0.05,top=0.92)
				plt.tight_layout()
				
				

				output_path = Path(f'/data/tierras/fields/{field}/sources/plots/')
				if not os.path.exists(output_path):
					os.mkdir(output_path)
					set_tierras_permissions(output_path)
				plt.savefig(output_path/f'{med_std_over_theo:.2f}_{target}_global_summary.png', dpi=200)
				set_tierras_permissions(output_path/f'{med_std_over_theo:.2f}_{target}_global_summary.png')
				plt.close()
				plt.ion()

				if email: 
					# Send summary plots 
					subject = f'[Tierras]_Data_Analysis_Report:{date}_{field}'
					summary_path = f'/data/tierras/lightcurves/{date}/{field}/{ffname}/{date}_{field}_summary.png'
					lc_path = f'/data/tierras/lightcurves/{date}/{field}/{ffname}/{date}_{field}_lc.png' 
					append = '{} {}'.format(summary_path,lc_path)
					emails = 'juliana.garcia-mejia@cfa.harvard.edu patrick.tamburo@cfa.harvard.edu'
					os.system('echo | mutt {} -s {} -a {}'.format(emails,subject,append))

			# write out the best light curve 
			output_dict = {'BJD TDB':times+x_offset, 'Flux':best_corr_flux, 'Flux Error':best_corr_flux_err, 'Airmass':airmasses}
			output_path = Path(f'/data/tierras/fields/{field}/sources/lightcurves/')
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

			if os.path.exists(output_path/f'{target}_global_lc.csv'):
				os.remove(output_path/f'{target}_global_lc.csv')
			
			filename = open(output_path/f'{target}_global_lc.csv', 'a')
			filename.write(f'# this light curve was made using {best_phot_style}\n' )
			output_df = pd.DataFrame(output_dict)
			output_df.to_csv(filename, index=0, na_rep=np.nan)
			filename.close()
			set_tierras_permissions(output_path/f'{target}_global_lc.csv')

			gc.collect() # do garbage collection to prevent memory leaks 
			print(f'tloop: {time.time()-tloop:.1f}')
			avg_mearth_times[tt] = np.mean(mearth_style_times)
			# print(f'avg mearth_style time: {np.mean(avg_mearth_times[0:tt+1]):.2f}')
		
if __name__ == '__main__':
	main()