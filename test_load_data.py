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
