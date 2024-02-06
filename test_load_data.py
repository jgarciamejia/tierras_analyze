import re
import os
import pdb
import glob
import numpy as np
import pandas as pd

def find_all_cols_w_keywords(df,*keywords):
	cols = []
	for col in df.columns:
		for keyword in keywords:
			if keyword in col:
				cols.append(col)
	return cols

def get_num_in_str(string):
	num = re.findall(r'[0-9]+', string)
	if len(num) > 1:
		print ('get_num_in_str error: more than one number IDd in string')
	elif len(num) == 1:
		return int(num[0]) 

def get_AIJ_star_numbers(df,column_kw):
	comp_keywords = find_all_cols_w_keywords(df,column_kw)
	comp_nums = [get_num_in_str(keyword) for keyword in comp_keywords]
	return np.array(comp_nums)

def calc_rel_flux(df,exclude_comps): #JGM MODIFIEF NOV 15/2023
	source_min_sky_T1 = df['Target Source-Sky ADU'].to_numpy()
	sum_source_min_sky_Cs = np.zeros(len(source_min_sky_T1))
	aij_comps = get_AIJ_star_numbers(df,'Source-Sky ADU')
	all_stars = np.arange(1,np.max(aij_comps[aij_comps != None])) #number of refs sans target

	used_compstars = []
	for star in all_stars:
		if not np.any(exclude_comps == star) and np.any(aij_comps == star):
		#if ref star not in exclude_comps and ref star in aij_comps:
			source_min_sky_comp = df['Ref '+str(star)+' Source-Sky ADU'].to_numpy()
			sum_source_min_sky_Cs += source_min_sky_comp
			used_compstars.append(star)
		else:
			continue

	return source_min_sky_T1 / sum_source_min_sky_Cs, used_compstars

def return_dataframe_onedate(mainpath,targetname,obsdate,ffname): #JGM MODIFIED NOV 9/2023
	datepath = os.path.join(mainpath,obsdate,targetname,ffname)
	optimal_lc_fname = os.path.join(datepath,'optimal_lc.txt') #returns only optimal aperture lc
	try:
		optimal_lc_csv = open(optimal_lc_fname).read().rstrip('\n')
		df = pd.read_csv(optimal_lc_csv)
	except FileNotFoundError:
		print ("No photometric extraction for {} on {}".format(targetname,obsdate))
		return None
	return df, optimal_lc_csv

def return_dataframe_onedate_forapradius(mainpath,targetname,obsdate,ffname,ap_radius='optimal'): #JGM MODIFIED JAN3/2024: return dataframe for a user-defined aperture.
        if ap_radius == 'optimal': # simply use optimal radius according to ap_phot. Needs work. 
                df,lc_fname = return_dataframe_onedate(mainpath,targetname,obsdate,ffname)
                print('Optimal ap radius: ',lc_fname.split('_')[-1].split('.csv')[0])
                return df,lc_fname
        else:
                datepath = os.path.join(mainpath,obsdate,targetname,ffname)
                lc_fname = os.path.join(datepath,'circular_fixed_ap_phot_{}.csv'.format(str(ap_radius)))
                print (ap_radius)
                try:
                        df = pd.read_csv(lc_fname)
                except FileNotFoundError:
                        print ("No photometric extraction for {} with aperture radius of {} pixels on {}".format(targetname,ap_radius,obsdate))
                        return None
                return df, lc_fname

def return_data_onedate(mainpath,targetname,obsdate,ffname,exclude_comps): #JGM MODIFIED NOV 9/2023
	df,filename = return_dataframe_onedate(mainpath,targetname,obsdate,ffname)
	bjds = df['BJD TDB'].to_numpy()
	widths = np.sqrt ((df['Target X FWHM Arcsec'].to_numpy())**2 +  (df['Target Y FWHM Arcsec'].to_numpy())**2)
	airmasses = df['Airmass'].to_numpy()
	rel_flux_T1 = df['Target Relative Flux'].to_numpy()

	if not np.all(exclude_comps) == None:
		recalc_rel_flux_T1,_ = calc_rel_flux(df,exclude_comps)
		rel_flux_T1 = recalc_rel_flux_T1
	return (df,bjds,rel_flux_T1,airmasses,widths)

def make_global_lists(mainpath,targetname,ffname,exclude_dates,complist,ap_radius='optimal'): #JGM: Jan 4, 2024. 
	# arrays to hold the full dataset
	full_bjd = []
	full_flux = []
	full_err = []
	full_flux_div_expt = [] # sometimes data from one star has different exposure times in a given night or between nights
	full_err_div_expt = []
	full_reg = None
	full_reg_err = None

	full_relflux = []
	full_exptime = []
	full_sky = []
	full_x = []
	full_y = []
	full_airmass = []
	full_fwhm = []
	#full_corr_relflux = [] 

	# array to hold individual nights
	bjd_save = []
	lcfolderlist = np.sort(glob.glob(mainpath+"/**/"+targetname))
	lcdatelist = [lcfolderlist[ind].split("/")[4] for ind in range(len(lcfolderlist))] 

	for ii,lcfolder in enumerate(lcfolderlist):
		print("Processing", lcdatelist[ii])

		# if date excluded, skip
		if np.any(exclude_dates == lcdatelist[ii]):
			print ("{} :  Excluded".format(lcdatelist[ii]))
			continue

		# read the .csv file
		try:
			df, optimal_lc = return_dataframe_onedate_forapradius(mainpath,targetname,lcdatelist[ii],ffname,ap_radius)
		except TypeError:
			continue

		bjds = df['BJD TDB'].to_numpy()
		flux = df['Target Source-Sky ADU']
		err = df['Target Source-Sky Error ADU']
		expt = df['Exposure Time']
		sky = df['Target Sky ADU']
		x = df['Target X']
		y = df['Target Y']
		airmass = df['Airmass']
		fwhm = np.nanmean([df['Target X FWHM Arcsec'],df['Target Y FWHM Arcsec']],axis=0)
		
		relflux = df['Target Relative Flux']
		#corr_relflux = df['Target Post-Processed Normalized Flux']
		print ('{} cadences'.format(len(bjds)))
		if sum(expt)/(3600) < 2:
			print('Less than 2 hour(s) of data, skipping.')
			continue 

		# get the comparison fluxes.
		comps = {}
		comps_err = {}
		for comp_num in complist:
			try:
				comps[comp_num] = df['Ref '+str(comp_num)+' Source-Sky ADU'] / expt  # divide by exposure time since it can vary between nights
				comps_err[comp_num] = df['Ref '+str(comp_num)+' Source-Sky Error ADU'] / expt
			except:
				print("Error with comp", str(comp_num))
				continue

		# make a list of all the comps
		regressors = []
		regressors_err = []
		for key in comps.keys():
			regressors.append(comps[key])
			regressors_err.append(comps_err[key])
		regressors = np.array(regressors)
		regressors_err = np.array(regressors_err)

		# add this night of data to the full data set
		full_bjd.extend(bjds)
		full_flux.extend(flux)
		full_err.extend(err)
		full_flux_div_expt.extend(flux/expt)
		full_err_div_expt.extend(err/expt)		
		bjd_save.append(bjds)

		full_relflux.extend(relflux)
		full_exptime.extend(expt)
		full_sky.extend(sky/expt)
		full_x.extend(x)
		full_y.extend(y)
		full_airmass.extend(airmass)
		full_fwhm.extend(fwhm)
		#full_corr_relflux.extend(corr_relflux)

		if full_reg is None:
			full_reg = regressors
		else:
			full_reg = np.concatenate((full_reg, regressors), axis=1) 
		
		if full_reg_err is None:
			full_reg_err = regressors_err
		else:
			full_reg_err = np.concatenate((full_reg_err, regressors_err), axis=1)

	# convert from lists to arrays
	full_bjd = np.array(full_bjd)
	full_flux = np.array(full_flux)
	full_err = np.array(full_err)
	full_reg_err = np.array(full_reg_err)
	full_flux_div_expt = np.array(full_flux_div_expt)
	full_err_div_expt =np.array(full_err_div_expt)
	full_relflux = np.array(full_relflux)
	full_exptime = np.array(full_exptime)
	full_sky = np.array(full_sky)
	full_x = np.array(full_x)
	full_y = np.array(full_y)
	full_airmass = np.array(full_airmass)
	full_fwhm = np.array(full_fwhm)
	#full_corr_relflux = np.array(full_corr_relflux)

	return full_bjd, bjd_save, full_flux, full_err, full_reg, full_reg_err, full_flux_div_expt, full_err_div_expt, full_relflux, full_exptime, full_sky, full_x, full_y, full_airmass, full_fwhm #, full_corr_relflux

def make_global_lists_refastarget(ref_as_target,mainpath,targetname,ffname,exclude_dates,complist,ap_radius='optimal'): #JGM: Jan 4, 2024. 
	# arrays to hold the full dataset
	full_bjd = []
	full_flux = []
	full_err = []
	full_flux_div_expt = [] # sometimes data from one star has different exposure times in a given night or between nights
	full_err_div_expt = []
	full_reg = None

	full_relflux = []
	#full_corr_relflux = [] 

	# array to hold individual nights
	bjd_save = []
	lcfolderlist = np.sort(glob.glob(mainpath+"/**/"+targetname))
	lcdatelist = [lcfolderlist[ind].split("/")[4] for ind in range(len(lcfolderlist))] 

	for ii,lcfolder in enumerate(lcfolderlist):
		print("Processing", lcdatelist[ii])

		# if date excluded, skip
		if np.any(exclude_dates == lcdatelist[ii]):
			print ("{} :  Excluded".format(lcdatelist[ii]))
			continue

		# read the .csv file
		try:
			df, optimal_lc = return_dataframe_onedate_forapradius(mainpath,targetname,lcdatelist[ii],ffname,ap_radius)
		except TypeError:
			continue

		bjds = df['BJD TDB'].to_numpy()
		flux = df['Ref '+str(ref_as_target)+' Source-Sky ADU']
		err = df['Ref '+str(ref_as_target)+' Source-Sky Error ADU']
		expt = df['Exposure Time']
		
		relflux = df['Target Relative Flux']

		print ('{} cadences'.format(len(bjds)))

		# get the comparison fluxes.
		comps = {}
		for comp_num in complist:
			try:
				comps[comp_num] = df['Ref '+str(comp_num)+' Source-Sky ADU'] / expt  # divide by exposure time since it can vary between nights
			except:
				print("Error with comp", str(comp_num))
				continue

		# make a list of all the comps
		regressors = []
		for key in comps.keys():
			regressors.append(comps[key])
		regressors = np.array(regressors)

		# add this night of data to the full data set
		full_bjd.extend(bjds)
		full_flux.extend(flux)
		full_err.extend(err)
		full_flux_div_expt.extend(flux/expt)
		full_err_div_expt.extend(err/expt)		
		bjd_save.append(bjds)

		full_relflux.extend(relflux)
		#full_corr_relflux.extend(corr_relflux)

		if full_reg is None:
			full_reg = regressors
		else:
			full_reg = np.concatenate((full_reg, regressors), axis=1) 

	# convert from lists to arrays
	full_bjd = np.array(full_bjd)
	full_flux = np.array(full_flux)
	full_err = np.array(full_err)
	full_flux_div_expt = np.array(full_flux_div_expt)
	full_err_div_expt =np.array(full_err_div_expt)
	full_relflux = np.array(full_relflux)

	return full_bjd, bjd_save, full_flux, full_err, full_reg, full_flux_div_expt, full_err_div_expt


# def make_global_lists(mainpath,targetname,ffname,exclude_dates,complist): JGM: Dec 2023
# 	# arrays to hold the full dataset
# 	full_bjd = []
# 	full_flux = []
# 	full_err = []
# 	full_reg = None

# 	full_relflux = []
# 	#full_corr_relflux = [] 

# 	# array to hold individual nights
# 	bjd_save = []
# 	lcfolderlist = np.sort(glob.glob(mainpath+"/**/"+targetname))
# 	lcdatelist = [lcfolderlist[ind].split("/")[4] for ind in range(len(lcfolderlist))] 

# 	for ii,lcfolder in enumerate(lcfolderlist):
# 		print("Processing", lcdatelist[ii])

# 		# if date excluded, skip
# 		if np.any(exclude_dates == lcdatelist[ii]):
# 			print ("{} :  Excluded".format(lcdatelist[ii]))
# 			continue

# 		# read the .csv file
# 		try:
# 			df, optimal_lc = return_dataframe_onedate(mainpath,targetname,lcdatelist[ii],ffname)
# 		except TypeError:
# 			continue

# 		bjds = df['BJD TDB'].to_numpy()
# 		flux = df['Target Source-Sky ADU']
# 		err = df['Target Source-Sky Error ADU']
# 		expt = df['Exposure Time']
		
# 		relflux = df['Target Relative Flux']
# 		#corr_relflux = df['Target Post-Processed Normalized Flux']

# 		print ('{} cadences'.format(len(bjds)))

# 		# get the comparison fluxes.
# 		comps = {}
# 		for comp_num in complist:
# 			try:
# 				comps[comp_num] = df['Ref '+str(comp_num)+' Source-Sky ADU'] / expt  # divide by exposure time since it can vary between nights
# 			except:
# 				print("Error with comp", str(comp_num))
# 				continue

# 		# make a list of all the comps
# 		regressors = []
# 		for key in comps.keys():
# 			regressors.append(comps[key])
# 		regressors = np.array(regressors)

# 		# add this night of data to the full data set
# 		full_bjd.extend(bjds)
# 		full_flux.extend(flux/expt)
# 		full_err.extend(err/expt)
# 		bjd_save.append(bjds)

# 		full_relflux.extend(relflux)
# 		#full_corr_relflux.extend(corr_relflux)

# 		if full_reg is None:
# 			full_reg = regressors
# 		else:
# 			full_reg = np.concatenate((full_reg, regressors), axis=1) 

# 	# convert from lists to arrays
# 	full_bjd = np.array(full_bjd)
# 	full_flux = np.array(full_flux)
# 	full_err = np.array(full_err)

# 	full_relflux = np.array(full_relflux)
# 	#full_corr_relflux = np.array(full_corr_relflux)

# 	return full_bjd, bjd_save, full_flux, full_err, full_reg, full_relflux#, full_corr_relflux


