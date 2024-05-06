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

def identify_target_gaia_id(target):
	if 'Gaia DR3' in target:
		gaia_id = int(target.split('Gaia DR3 ')[1])
		return gaia_id

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
	ap.add_argument("-overwrite", required=False, default=False, help="Whether or not to overwrite existing lc files.")
	ap.add_argument("-email", required=False, default=False, help="Whether or not to send email with summary plots.")
	ap.add_argument("-plot", required=False, default=False, help="Whether or not to generate a summary plot to the target's /data/tierras/targets directory")

	args = ap.parse_args(raw_args)

	#Access observation info
	field = args.field
	ffname = args.ffname
	overwrite = t_or_f(args.overwrite)
	email = t_or_f(args.email)
	plot = t_or_f(args.plot)

	if args.target is None: 
		target = field 
	else:
		target = args.target

	# identify dates on which this field was observed 
	date_list = glob(f'/data/tierras/photometry/**/{field}/{ffname}')	
	# date_list = np.array(date_list)[[0,-1]]
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
		n_ims += len(pq.read_table(phot_files[0]))

	times = np.zeros(n_ims, dtype='float64')
	airmasses = np.zeros(n_ims, dtype='float16')
	exposure_times = np.zeros(n_ims, dtype='float16')
	filenames = np.zeros(n_ims, dtype='str')
	ha = np.zeros(n_ims, dtype='float16')
	humidity = np.zeros(n_ims, dtype='float16')
	flux = np.zeros((n_dfs, n_ims, n_sources), dtype='float32')
	flux_err = np.zeros_like(flux)
	non_linear_flags = np.zeros_like(flux, dtype='int')
	saturated_flags = np.zeros_like(flux, dtype='int')
	sky = np.zeros_like(flux)
	x = np.zeros_like(flux)
	y = np.zeros_like(flux)
	fwhm_x = np.zeros_like(flux)
	fwhm_y = np.zeros_like(flux)

	times_list = []
	start = 0

	print('Reading in data...')
	t1 = time.time()

	ancillary_cols = ['Filename', 'BJD TDB', 'Airmass', 'Exposure Time', 'HA', 'Dome Humid']

	for i in range(len(date_list)):
		ancillary_file = glob(date_list[i]+'/**ancillary**.parquet')
		phot_files = glob(date_list[i]+'/**phot**.parquet')
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
				use_cols.append(f'S{source_inds[i][k]} X FWHM')
				use_cols.append(f'S{source_inds[i][k]} Y FWHM')

			
			data_tab = pq.read_table(phot_files[j], columns=use_cols)
			stop = start+len(data_tab)

			times[start:stop] = np.array(ancillary_tab['BJD TDB'])
			if j == 0:
				times_list.append(times[start:stop])
			airmasses[start:stop] = np.array(ancillary_tab['Airmass'])
			exposure_times[start:stop] = np.array(ancillary_tab['Exposure Time'])
			filenames[start:stop] = np.array(ancillary_tab['Filename'])
			ha[start:stop] = np.array(ancillary_tab['HA'])
			humidity[start:stop] = np.array(ancillary_tab['Dome Humid'])
			
			for k in range(n_sources):
				flux[j,start:stop,k] = np.array(data_tab[f'S{source_inds[i][k]} Source-Sky'])
				flux_err[j,start:stop,k] = np.array(data_tab[f'S{source_inds[i][k]} Source-Sky Err'])
				non_linear_flags[j,start:stop,k] = np.array(data_tab[f'S{source_inds[i][k]} NL Flag'])
				saturated_flags[j,start:stop,k] = np.array(data_tab[f'S{source_inds[i][k]} Sat Flag'])
				sky[j,start:stop,k] = np.array(data_tab[f'S{source_inds[i][k]} Sky'])
				x[j,start:stop,k] = np.array(data_tab[f'S{source_inds[i][k]} X'])
				y[j,start:stop,k] = np.array(data_tab[f'S{source_inds[i][k]} Y'])
				fwhm_x[j,start:stop,k] = np.array(data_tab[f'S{source_inds[i][k]} X FWHM'])
				fwhm_y[j,start:stop,k] = np.array(data_tab[f'S{source_inds[i][k]} Y FWHM'])
		start = stop
	print(f'Read-in: {time.time()-t1}')
	
	tierras_target_id = identify_target_gaia_id(field)

	for tt in range(len(common_source_ids)):
		tloop = time.time()
		if common_source_ids[tt] == tierras_target_id:
			target = field
			plot = True
		else:
			target = 'Gaia DR3 '+str(common_source_ids[tt])
			plot = False
		print(f'Doing {target}, source {tt+1} of {len(common_source_ids)}.')

		target_gaia_id = identify_target_gaia_id(target)

		ref_gaia_ids = ref_selection(target_gaia_id, source_dfs[0], common_source_ids, max_refs=25)
		inds = np.where(target_gaia_id == common_source_ids)[0]
		for i in range(len(ref_gaia_ids)):
			inds = np.append(inds, np.where(ref_gaia_ids[i] == common_source_ids)[0])

		med_stddevs = np.zeros(n_dfs)
		best_med_stddev = 9999999.
		for i in range(n_dfs):
			# flux[i, :, inds[0]][saturated_flags[i,:,inds[0]]==1] = np.nan
			corr_flux, corr_flux_err, alc, alc_err, weights = mearth_style_pat_weighted_flux(flux[i][:,inds], flux_err[i][:,inds], non_linear_flags[i][:,inds], airmasses, exposure_times)

			#Trim any NaNs
			use_inds = ~np.isnan(corr_flux)
			times_sc = times[use_inds]
			corr_flux_sc = corr_flux[use_inds]
			corr_flux_err_sc = corr_flux_err[use_inds] 
			
			#Sigmaclip
			v,l,h = sigmaclip(corr_flux_sc)
			use_inds = np.where((corr_flux_sc>l)&(corr_flux_sc<h))[0]
			times_sc = times_sc[use_inds]
			corr_flux_sc = corr_flux_sc[use_inds]
			corr_flux_err_sc = corr_flux_err_sc[use_inds]
			
			norm = np.mean(corr_flux_sc)
			corr_flux_sc /= norm 
			corr_flux_err_sc /= norm

			# Evaluate the median standard deviation over 5-minute intervals 
			bin_inds = tierras_binner_inds(times_sc, bin_mins=5)
			stddevs = np.zeros(len(bin_inds))
			for k in range(len(bin_inds)):
				stddevs[k] = np.nanstd(corr_flux_sc[bin_inds[k]])
			med_stddevs[i] = np.nanmedian(stddevs)	
			if med_stddevs[i] < best_med_stddev: 
				#print(f'New best light curve found! {phot_files[j].split("/")[-1]}: {med_stddevs[j]*1e6:.1f} in 5-minute bins')
				best_med_stddev = med_stddevs[i]
				best_phot_file = i 
				best_corr_flux = corr_flux
				best_corr_flux_err = corr_flux_err
				best_alc = alc
				best_alc_err = alc_err

		# write out the best light curve 
		output_dict = {'BJD TDB':times, 'Flux':best_corr_flux, 'Flux Error':best_corr_flux_err}
		output_path = f'/data/tierras/targets/{target}'
		if not os.path.exists(output_path):
			os.mkdir(output_path)

		best_phot_style = phot_files[best_phot_file].split(f'{field}_')[1].split('.csv')[0]

		if os.path.exists(output_path+f'/{target}_global_lc.csv'):
			os.remove(output_path+f'/{target}_global_lc.csv')
		f = open(output_path+f'/{target}_global_lc.csv', 'a')
		f.write(f'# this light curve was made using {best_phot_style}\n' )
		output_df = pd.DataFrame(output_dict)
		output_df.to_csv(f, index=0, na_rep=np.nan)

		
		bins, std, theo = allen_deviation(times, best_corr_flux, best_corr_flux_err)
		if np.nanmedian(std/theo) > 10:
			print('Possible variable!')
			plot = True
		
		# do a plot if this is the tierras target 

		if (target == field) or plot:
			plt.ioff()

			use_inds = ~np.isnan(best_corr_flux)

			raw_flux = flux[best_phot_file,:,0]

			v, l, h = sigmaclip(best_corr_flux[~np.isnan(best_corr_flux)])
			use_inds = np.where((best_corr_flux>l)&(best_corr_flux<h))[0]

			fig, axes = plt.subplots(8, len(date_list), figsize=(24/6*len(date_list),20), sharey='row', gridspec_kw={'height_ratios':[1,2,1,1,1,1,1,2]})

			x_offset = int(np.floor(times[0]))
			times -= x_offset
			for i in range(len(date_list)):
				if len(date_list) == 1:
					ax = axes 
				else:
					ax = axes[:,i]

				ax[0].set_title(date_list[i].split('/')[4], fontsize=14)
				ax[0].plot(times[use_inds], raw_flux[use_inds]/np.nanmedian(raw_flux[use_inds]), 'k.', label='Target')
				ax[0].plot(times[use_inds], best_alc[use_inds]/np.nanmedian(best_alc[use_inds]), 'r.', label='ALC')

				ax[1].errorbar(times[use_inds], best_corr_flux[use_inds], best_corr_flux_err[use_inds], marker='.', color='#b0b0b0', ls='')

				ax[2].plot(times, airmasses, marker='.', ls='')	
				# ax_ha = ax[2].twinx()
				# ax_ha.plot(times, ha, color='tab:orange')

				ax[3].plot(times, sky[best_phot_file][:,0]/exposure_times, color='tab:cyan', marker='.', ls='')

				ax[4].plot(times, x[best_phot_file][:,0]-np.nanmedian(x[best_phot_file][:,0]), color='tab:green',label='X-med(X)', marker='.', ls='')
				ax[4].plot(times, y[best_phot_file][:,0]-np.nanmedian(y[best_phot_file][:,0]), color='tab:red',label='Y-med(Y)', marker='.', ls='')
				ax[4].set_ylim(-10,10)

				ax[5].plot(times, fwhm_x[best_phot_file][:,0], color='tab:pink', label='X', marker='.', ls='')
				ax[5].plot(times, fwhm_y[best_phot_file][:,0], color='tab:purple',label='Y', marker='.', ls='')	
				
				ax[6].plot(times, humidity, color='tab:brown', marker='.', ls='')

				allen_inds = np.where((times>=times_list[i][0])&(times<=times_list[i][-1]))[0]
				bins, std, theo = allen_deviation(times[allen_inds], best_corr_flux[allen_inds], best_corr_flux_err[allen_inds])

				ax[7].plot(bins, std*1e6, lw=2,label='Measured', marker='.')
				ax[7].plot(bins, theo*1e6,lw=2,label='Theoretical', marker='.')
				ax[7].set_yscale('log')
				ax[7].set_xscale('log')
			
				if i == 0:
					ax[0].set_ylabel('Norm. Flux', fontsize=14)
					ax[1].set_ylabel('Corr. Flux', fontsize=14)
					ax[2].set_ylabel('Airmass', fontsize=14)
					ax[3].set_ylabel('Sky\n(ADU/s)', fontsize=14)
					ax[4].set_ylabel('Pos.\n(pix.)', fontsize=14)
					ax[5].set_ylabel('FWHM\n(")', fontsize=14)
					ax[6].set_ylabel('Dome Humid.\n(%)',fontsize=14)
					ax[7].set_ylabel('$\sigma$ (ppm)', fontsize=14)
				
				for a in range(len(ax)):
					if a != len(ax) - 1:
						ax[a].set_xlim(times_list[i][0], times_list[i][-1])
					ax[a].grid(alpha=0.7)
					ax[a].tick_params(labelsize=12)
					if a != len(ax)-1:
						ax[a].tick_params(labelbottom=False)
				
					if i > 0 and len(date_list) != 1:
						ax[a].spines['left'].set_visible(False)
						ax[a].tick_params(labelleft=False)
						ax[a].yaxis.tick_left()
					if i < len(date_list) - 1:
						ax[a].spines['right'].set_visible(False)
						ax[a].yaxis.tick_right()
					if i == 0:
						ax[a].yaxis.tick_left()
					if i == len(date_list) - 1:
						ax[a].yaxis.tick_right()
						ax[a].tick_params(labelright=True)
						# if a == 2:
						# 	ax[a].set_ylabel('Hour Angle', fontsize=14, color='tab:orange')
						# ax[0].legend(loc='center left', bbox_to_anchor=(1.1,0.5))
						ax[4].legend()
						ax[5].legend()
					if a == 6:
						ax[a].set_xlabel('BJD$_{TDB}$-'+f'{x_offset:d}', fontsize=14)
						ax[a].tick_params(labelbottom=True)
					if a == 7:
						ax[a].set_xlabel('Bin size', fontsize=14)	
						ax[a].xaxis.set_major_formatter(ScalarFormatter())

			# plt.suptitle(target, fontsize=14)
			plt.subplots_adjust(hspace=0.1,wspace=0.05,left=0.05,right=0.92,bottom=0.05,top=0.92)
			plt.tight_layout()

			plt.savefig(output_path+f'/{target}_global_summary.png', dpi=300)
			plt.ion()
			if email: 
				# Send summary plots 
				subject = f'[Tierras]_Data_Analysis_Report:{date}_{field}'
				summary_path = f'/data/tierras/lightcurves/{date}/{field}/{ffname}/{date}_{field}_summary.png'
				lc_path = f'/data/tierras/lightcurves/{date}/{field}/{ffname}/{date}_{field}_lc.png' 
				append = '{} {}'.format(summary_path,lc_path)
				emails = 'juliana.garcia-mejia@cfa.harvard.edu patrick.tamburo@cfa.harvard.edu'
				os.system('echo | mutt {} -s {} -a {}'.format(emails,subject,append))

		gc.collect() # do garbage collection to prevent memory leaks 
		print(f'tloop: {time.time()-tloop:.1f}')
if __name__ == '__main__':
	main()