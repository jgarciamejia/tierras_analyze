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
		optimal_lc_csv = open(optimal_lc_fname).read()
		df = pd.read_csv(optimal_lc_csv)
	except FileNotFoundError:
		print ("No photometric extraction for {} on {}".format(targetname,obsdate))
		return None
	return df, optimal_lc_csv

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

def make_global_lists(lcfolderlist):
	# arrays to hold the full dataset
	full_bjd = []
	full_flux = []
	full_err = []
	full_reg = None

	# array to hold individual nights
	bjd_save = []

	lcdatelist = [lcfolderlist[ind].split("/")[4] for ind in range(len(lcfolderlist))] 

	for ii,lcfolder in enumerate(lcfolderlist):
		print("Processing", lcdatelist[ii])

		# if date excluded, skip
		if np.any(exclude_dates == lcdatelist[ii]):
			print ("{} :  Excluded".format(lcdatelist[ii]))
			continue

		# read the .csv file
		try:
			df, optimal_lc = ld.return_dataframe_onedate(lcpath,target,lcdatelist[ii],ffname)
		except TypeError:
			continue

		bjds = df['BJD TDB'].to_numpy()
		flux = df['Target Source-Sky ADU']
		err = df['Target Source-Sky Error ADU']
		expt = df['Exposure Time']

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
		full_flux.extend(flux/expt)
		full_err.extend(err/expt)
		bjd_save.append(bjds)

		if full_reg is None:
			full_reg = regressors
		else:
			full_reg = np.concatenate((full_reg, regressors), axis=1) 

	# convert from lists to arrays
	full_bjd = np.array(full_bjd)
	full_flux = np.array(full_flux)
	full_err = np.array(full_err)

	return full_bjd, bjd_save, full_flux, full_err, full_reg