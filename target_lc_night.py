import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 
import argparse
from astroquery.simbad import Simbad
from glob import glob 
from scipy.stats import sigmaclip 
import copy 
from ap_phot import regression

'''
	Script for creating a Tierras light curve for a target *on a single night*.
'''

def target_and_reference_selection(source_df, target_ind=None, target_gaia_id=None, target_2mass_id=None, ref_star_delta_rp=5, same_chip=False, distance_limit=4580):

	source_df['source_id'] = source_df['source_id'].astype(str) # cast source_id column to be str, not int64
	
	# start by identifying the target in the source_df
	# if the target_ind has been specified by the user then you're all set 
	# If not, if a gaia or 2mass id has been specified, use that to identify the target
	if target_ind is None:
		if (target_gaia_id is not None): 
			target_ind = np.where(source_df['source_id'] == target_gaia_id)[0][0]
		elif (target_2mass_id is not None) and (target_gaia_id is None):
			target_ind = np.where(source_df['2MASS'] == target_2mass_id)[0][0]
		else:	
			# if no gaia/2mass identifier passed, try searching for it on simbad
			result_table = Simbad.query_objectids(target).to_pandas().stack().str.decode('utf-8').unstack()

			if result_table is None:
				raise RuntimeError(f'No object matching {target} found on Simbad!\nTry specifying the Gaia DR3 or 2MASS identifier in the call to target_and_reference_selection.')
			
			target_gaia_id_ind = np.where(result_table['ID'].str.contains('Gaia DR3'))[0]
			target_twomass_id_ind = np.where(result_table['ID'].str.contains('2MASS'))[0]

			if (len(target_gaia_id_ind) == 0) and (len(target_twomass_id_ind) == 0):
				raise RuntimeError(f'No Gaia DR3 or 2MASS identifiers found for {target} on Simbad!')
			
			if len(target_gaia_id_ind) == 1:
				target_gaia_id = result_table['ID'][target_gaia_id_ind[0]].split('Gaia DR3 ')[1]	
				target_ind = np.where(source_df['source_id'] == target_gaia_id)[0][0]	
			else:
				target_2mass_id = result_table['ID'][target_twomass_id_ind[0]].split('2MASS J')[1] 
				target_ind = np.where(source_df['2MASS'] == target_2mass_id)[0][0] 


	# now identify suitable reference stars 
	ref_df = source_df.drop(index=target_ind)

	target_rp = source_df['phot_rp_mean_mag'][target_ind]
	target_x = source_df['X pix'][target_ind]
	target_y = source_df['Y pix'][target_ind]
	target_chip = source_df['Chip'][target_ind]

	print(f'Identified target at ({target_x:.1f},{target_y:.1f}), Rp mag = {target_rp:.1f}')

	variable_mask = ref_df['phot_variable_flag'] != 'VARIABLE' # toss any identified as variable in Gaia
	rp_mag_mask = ref_df['phot_rp_mean_mag'] < target_rp + ref_star_delta_rp # toss any fainter than the target in Rp band by ref_star_delta_rp mags
	distance_mask = np.sqrt((ref_df['X pix']-target_x)**2 + (ref_df['Y pix']-target_y)**2) < distance_limit

	# TODO: drop references based on flux contamination?
	
	use_refs = variable_mask & rp_mag_mask & distance_mask

	if same_chip:
		chip_mask = ref_df['Chip'] == target_chip
		use_refs = use_refs & chip_mask

	if sum(use_refs) < 2:
		raise RuntimeError('Found fewer than 2 reference stars! Try applying less restrictive reference star criteria.')

	print(f'Found {sum(use_refs)} suitable reference stars out of {len(ref_df)} possible sources in the field with photometry.')
	refs = ref_df[use_refs]
	targ_and_refs = pd.concat([pd.DataFrame(source_df.loc[target_ind]).T,refs], ignore_index=True)
	
	# get the indices of each source in the full source df so they can be referenced when making light curves
	source_df_ind = np.zeros(len(targ_and_refs), dtype='int')
	for ii in range(len(targ_and_refs)):
		gaia_id = targ_and_refs['source_id'][ii]
		source_df_ind[ii] = np.where(source_df['source_id'] == gaia_id)[0][0] + 1

	targ_and_refs.insert(0, 'source_index', source_df_ind)

	return targ_and_refs

def mearth_style_pat_weighted_flux(data_dict):
	""" Use the comparison stars to derive a frame-by-frame zero-point magnitude. Also filter and mask bad cadences """
	""" it's called "mearth_style" because it's inspired by the mearth pipeline """
	""" this version works with fluxes """

	flux = data_dict['Flux']
	flux_err = data_dict['Flux Error']
	airmasses = data_dict['Airmass']
	exptimes = data_dict['Exposure Time']

	D = 130 # cm 
	H = 2306 # m 
	sigma_s = 0.09*D**(-2/3)*airmasses**(7/4)*(2*exptimes)**(-1/2)*np.exp(-H/8000)


	n_sources = flux.shape[1]
	n_ims = flux.shape[0]

	# mask any cadences where the flux is negative for any of the sources 
	mask = np.any(flux < 0,axis=1)
	flux[mask] = np.nan 
	flux_err[mask] = np.nan

	flux_corr_save = np.zeros_like(flux)
	flux_err_corr_save = np.zeros_like(flux)
	alc_save = np.zeros_like(flux)
	alc_err_save = np.zeros_like(flux)
	mask_save = np.zeros_like(flux)
	weights_save = np.zeros((flux.shape[1]-1, flux.shape[1]))

	# loop over each star, calculate its zero-point correction using the other stars
	for i in range(n_sources):
		# target_source_id = cluster_ids[i] # this represents the ID of the "target" *in the photometry files
		regressor_inds = [j for j in np.arange(n_sources) if i != j] # get the indices of the stars to use as the zero point calibrators; these represent the indices of the calibrators *in the data_dict arrays*

		
		# grab target and source fluxes and apply initial mask 
		target_flux = flux[:,i]
		target_flux_err = flux_err[:,i]
		regressors = flux[:,regressor_inds]
		regressors_err = flux_err[:,regressor_inds]

		tot_regressor = np.sum(regressors, axis=1)  # the total regressor flux at each time point = sum of comp star fluxes in each exposure

		# identify cadences with "low" flux by looking at normalized summed reference star fluxes
		zp0s = tot_regressor/np.nanmedian(tot_regressor) 	
		mask = np.ones_like(zp0s, dtype='bool')  # initialize another bad data mask
		mask[np.where(zp0s < 0.8)[0]] = 0  # if regressor flux is decremented by 20% or more, this cadence is bad
		target_flux[~mask] = np.nan 
		target_flux_err[~mask] = np.nan 
		regressors[~mask] = np.nan
		regressors_err[~mask] = np.nan
		

		# repeat the cs estimate now that we've masked out the bad cadences
		# phot_regressor = np.nanpercentile(regressors, 90, axis=1)  # estimate photometric flux level for each star
		norms = np.nanmedian(regressors, axis=0)
		regressors_norm = regressors / norms
		regressors_err_norm = regressors_err / norms

		# # mask out any exposures where any reference star is significantly discrepant 
		# TODO: I think this is too aggressive, we don't really care if a reference star that is given very low weight in the ALC has is super discrepant in one frame
		# mask = np.zeros_like(target_flux, dtype='bool')
		# for j in range(regressors_norm.shape[1]):
		# 	v, l, h = sigmaclip(regressors_norm[:,j][~np.isnan(regressors_norm[:,j])])
		# 	mask[np.where((regressors_norm[:,j] < l) | (regressors_norm[:,j] > h))[0]] = 1

		# target_flux[mask] = np.nan 
		# target_flux_err[mask] = np.nan 
		# regressors[mask] = np.nan 
		# regressors_err[mask] = np.nan 
		# regressors_norm[mask] = np.nan 
		# regressors_err_norm[mask] = np.nan
				
		# now calculate the weights for each regressor
		# give all stars equal weights at first
		# weights_init = np.ones(regressors.shape[1])/regressors.shape[1]
		weights_init = 1/(np.nanmedian(regressors_err_norm,axis=0)**2)
		weights_init /= np.nansum(weights_init)
			
		# plot the normalized regressor fluxes and the initial zero-point correction
		# for j in range(len(regressors)):
		# 	plt.plot(regressors_norm[j])
		# plt.errorbar(np.arange(len(zp)), zp, zp_err, color='k', zorder=4)

		# do a 'crude' weighting loop to figure out which regressors, if any, should be totally discarded	
		delta_weights = np.zeros(regressors.shape[1])+999 # initialize
		threshold = 1e-4 # delta_weights must converge to this value for the loop to stop
		weights_old = weights_init
		full_ref_inds = np.arange(regressors.shape[1])
		while len(np.where(delta_weights>threshold)[0]) > 0:
			stddevs_measured = np.zeros(regressors.shape[1])		
			stddevs_expected = np.zeros_like(stddevs_measured) 

			# loop over each regressor
			for jj in range(regressors.shape[1]):
				F_t = regressors[:,jj]
				N_t = regressors_err[:,jj]

				# make its zeropoint correction using the flux of all the *other* regressors
				use_inds = np.delete(full_ref_inds, jj)

				# re-normalize the weights to sum to one
				weights_wo_jj = weights_old[use_inds]
				weights_wo_jj /= np.nansum(weights_wo_jj)

				# create a zeropoint correction using those weights 
				F_e = np.matmul(regressors[:,use_inds],weights_wo_jj)
				N_e = np.sqrt(np.matmul(regressors_err[:,use_inds]**2, weights_wo_jj**2))
				
				# calculate the relative flux 
				F_rel_flux = F_t / F_e
				sigma_rel_flux = np.sqrt((N_t/F_e)**2 + (F_t*N_e/(F_e**2))**2)

				# calculate total error on F_rel flux from sigma_rel_flux and sigma_scint			
	
				sigma_scint = 1.5*sigma_s*np.sqrt(1 + 1/(len(use_inds)))
				sigma_tot = np.sqrt(sigma_rel_flux**2 + sigma_scint**2)

				# renormalize
				norm = np.nanmedian(F_rel_flux)
				F_corr = F_rel_flux/norm 
				sigma_tot_corr = sigma_tot/norm 	

				# record the standard deviation of the corrected flux
				stddevs_measured[jj] = np.nanstd(F_corr)
				stddevs_expected[jj] = np.nanmean(sigma_tot_corr)

			# update the weights using the measured standard deviations
			weights_new = 1/stddevs_measured**2
			weights_new /= np.nansum(weights_new)
			delta_weights = abs(weights_new-weights_old)
			weights_old = weights_new

		weights = weights_new
		
		# determine if any references should be totally thrown out based on the ratio of their measured/expected noise
		noise_ratios = stddevs_measured/stddevs_expected
		# the noise ratio threshold will depend on how many bad/variable reference stars were used in the ALC
		# sigmaclip the noise ratios and set the upper limit to the n-sigma upper bound 
		# v, l, h = sigmaclip(noise_ratios, 2, 2)
		# weights[np.where(noise_ratios>h)[0]] = 0

		weights[np.where(noise_ratios>5)[0]] = 0
		weights /= sum(weights)

		if len(np.where(weights == 0)[0]) > 0:
			# now repeat the weighting loop with the bad refs removed 
			delta_weights = np.zeros(regressors.shape[1])+999 # initialize
			threshold = 1e-6 # delta_weights must converge to this value for the loop to stop
			weights_old = weights
			full_ref_inds = np.arange(regressors.shape[1])
			count = 0
			while len(np.where(delta_weights>threshold)[0]) > 0:
				stddevs_measured = np.zeros(regressors.shape[1])
				stddevs_expected = np.zeros_like(stddevs_measured)

				for jj in range(regressors.shape[1]):
					if weights_old[jj] == 0:
						continue

					F_t = regressors[:,jj]
					N_t = regressors[:,jj]

					use_inds = np.delete(full_ref_inds, jj)
					weights_wo_jj = weights_old[use_inds]
					weights_wo_jj /= np.nansum(weights_wo_jj)
					
					# create a zeropoint correction using those weights 
					F_e = np.matmul(regressors[:,use_inds], weights_wo_jj)
					N_e = np.sqrt(np.matmul(regressors_err[:,use_inds]**2, weights_wo_jj**2))

					# calculate the relative flux 
					F_rel_flux = F_t / F_e
					sigma_rel_flux = np.sqrt((N_t/F_e)**2 + (F_t*N_e/(F_e**2))**2)

					# calculate total error on F_rel flux from sigma_rel_flux and sigma_scint				
					sigma_scint = 1.5*sigma_s*np.sqrt(1 + 1/(len(use_inds)))
					sigma_tot = np.sqrt(sigma_rel_flux**2 + sigma_scint**2)

					# renormalize
					norm = np.nanmedian(F_rel_flux)
					F_corr = F_rel_flux/norm 
					sigma_tot_corr = sigma_tot/norm 	
		
					# record the standard deviation of the corrected flux
					stddevs_measured[jj] = np.nanstd(F_corr)
					stddevs_expected[jj] = np.nanmean(sigma_tot_corr)

				weights_new = 1/(stddevs_measured**2)
				weights_new /= np.sum(weights_new[~np.isinf(weights_new)])
				weights_new[np.isinf(weights_new)] = 0
				delta_weights = abs(weights_new-weights_old)
				weights_old = weights_new
				count += 1

		weights = weights_new

		# calculate the zero-point correction

		F_e = np.matmul(regressors, weights)
		N_e = np.sqrt(np.matmul(regressors_err**2, weights**2))	
		
		flux_corr = target_flux / F_e
		err_corr = np.sqrt((target_flux_err/F_e)**2 + (target_flux*N_e/(F_e**2))**2)

		# renormalize
		norm = np.nanmedian(flux_corr)
		flux_corr /= norm 
		err_corr /= norm 

		mask_save[:,i] = ~np.isnan(flux_corr)
		flux_corr_save[:,i] = flux_corr
		flux_err_corr_save[:,i] = err_corr
		alc_save[:,i] = F_e
		alc_err_save[:,i] = N_e
		weights_save[:,i] = weights

	output_dict = copy.deepcopy(data_dict)
	output_dict['ZP Mask'] = mask_save
	output_dict['Corrected Flux'] = flux_corr_save
	output_dict['Corrected Flux Error'] = flux_err_corr_save
	output_dict['ALC'] = alc_save 
	output_dict['ALC Error'] = alc_err_save
	output_dict['Weights'] = weights_save		

	return output_dict

def load_data(phot_df, source_df):

	times = np.array(phot_df['BJD TDB'])
	flux = np.zeros((len(phot_df), len(source_df)))
	flux_err = np.zeros_like(flux)
	airmasses = np.array(phot_df['Airmass'])
	exptimes = np.array(phot_df['Exposure Time'])
	backgrounds = np.zeros_like(flux)

	for ii in range(len(source_df)):
		flux[:,ii] = phot_df[f'S{source_df["source_index"][ii]} Source-Sky ADU']
		flux_err[:,ii] = phot_df[f'S{source_df["source_index"][ii]} Source-Sky Error ADU']
		backgrounds[:,ii] = phot_df[f'S{source_df["source_index"][ii]} Sky ADU']
	data_dict = {'BJD TDB':times, 'Flux':flux, 'Flux Error':flux_err, 'Airmass':airmasses, 'Exposure Time':exptimes, 'Background':backgrounds}

	return data_dict 

def allen_deviation(bins, times, flux, flux_err, exptime):
	'''Bins up a Tierras light curve following example from testextract.py 

	bins: array of bin sizes in minutes
	times: array of times in days
	flux: array of normalized fluxes 
	flux_err: array of normalized flux uncertainties (for calculating the theoretical curve)
	exptime: exposure time in seconds
	
	'''
	# calculate the theoretical curve: 
	# take the mean of the error array as the representation of the stddev at 0th bin size (i.e., data taken at the actual exposure time)
	# it is supposed to decrease by 1/sqrt(N), where N is the number of points in the given bin size
	theo = np.nanmean(flux_err)*1/np.sqrt(bins/(exptime/60))
	# now calculate the observed curve
	std = np.empty([len(bins)]) 
	binned_flux_list = []
	binned_time_list = []
	for ibinsize, thisbinsize in enumerate(bins):

		nbin = (times[-1] - times[0]) * 1440.0 / thisbinsize #figure out number of bins at this bin size

		bins_ = times[0] + thisbinsize * np.arange(nbin+1) / 1440.0
		
		binned_time = []
		binned_flux = []
		for i in range(len(bins_)-1): 
			inds = np.where((times>=bins_[i])&(times<=bins_[i+1]))
			if len(inds) > 0:
				binned_time.append(np.mean(times[inds]))
				binned_flux.append(np.mean(flux[inds]))
			else:
				binned_time.append(np.nan)
				binned_flux.append(np.nan)
				
		std[ibinsize] = np.nanstd(binned_flux)
		binned_time_list.append(binned_time)
		binned_flux_list.append(binned_flux)
	return std, theo, binned_time_list, binned_flux_list 

def main(raw_args=None):
	parser = argparse.ArgumentParser()
	parser.add_argument("-date", required=True, help="Date of observation in YYYYMMDD format.")
	parser.add_argument("-target", required=True, help="Name of observed target exactly as shown in raw FITS files.")
	parser.add_argument("-ffname", required=False, default='flat0000', help="Name of folder in which to store reduced+flattened data. Convention is flatXXXX. XXXX=0000 means no flat was used.")
	args = parser.parse_args(raw_args)

	global date, target, ffname, phot_path
	date = args.date
	target = args.target 
	ffname = args.ffname
	phot_path = f'/data/tierras/photometry/{date}/{target}/{ffname}'
	lc_files = glob(phot_path+'/**ap_phot**.csv')

	allen_dev_upper = 20
	allen_dev_res = 0.5

	# load in sources
	sources = pd.read_csv(f'{phot_path}/{date}_{target}_sources.csv')

	# identify target and reference stars
	targ_and_refs = target_and_reference_selection(sources, ref_star_delta_rp=4)

	for ii in range(10, len(lc_files)):
		print(ii)
		phot_df = pd.read_csv(lc_files[ii])
		data_dict = load_data(phot_df, targ_and_refs)
		corr_dict = mearth_style_pat_weighted_flux(data_dict)

		exptimes = np.unique(data_dict['Exposure Time']) 
		if len(exptimes) > 1:
			exp_time_counts = np.zeros(len(exptimes))
			for j in range(len(exptimes)):
				exp_time_counts[j] = len(np.where(phot_df['Exposure Time'] == exptimes[j])[0])
			exptime = exptimes[np.argmax(exp_time_counts)]
			print(f'WARNING: multiple exposure times used for {target} on {date}.\nUsing most common exptime for Allen deviation plot, which is {exptime}-s')
		else:
			exptime = exptimes[0]
		

		times = corr_dict['BJD TDB']
		norm_flux = corr_dict['Corrected Flux']
		norm_flux_errs = corr_dict['Corrected Flux Error']
		
		# calculate Allen deviation for the target
		bins = np.array([exptime])
		next_exptime = np.floor(exptime/(allen_dev_res*60)) * allen_dev_res*60 + allen_dev_res*60
		bins = np.append(bins, np.arange(next_exptime, allen_dev_upper*60+allen_dev_res*60, allen_dev_res*60)) / 60

		inds = ~np.isnan(norm_flux[:,0])
		regress_dict = {'Background':corr_dict['Background'][inds,0]}
		regressed_flux, intercept, coeffs, regress_dict_return = regression(norm_flux[inds,0], regress_dict)
		
		std, theo, binned_times, binned_fluxes = allen_deviation(bins, times[inds], regressed_flux, norm_flux_errs[inds,0], exptimes[0])

		if ii == 10:
			plt.ion()
			plt.plot(bins, std)
			plt.plot(bins, theo)
			plt.xscale('log')
			plt.yscale('log')

			plt.figure()
			plt.plot(binned_times[0], binned_fluxes[0], marker='o', color='k', ls='')
			plt.plot(binned_times[9], binned_fluxes[9], marker='o', ls='')
			breakpoint()
	return 

if __name__ == '__main__':
	main()