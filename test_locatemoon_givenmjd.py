import re
import math
import time 
import numpy as np

import palpy
import palutil
import twilight 

from astroquery.gaia import Gaia

def get_mjd(): # copied from autoobserve.py. JGM Jan 2024
  mjd = (time.time() / 86400.0) + 40587.0 # 86400 sec = 1 day // 40587 addition to account for fact that MJD starts from 4713 BC but time.time from 1970 AD
  return mjd 

def read_mjd_or_date(datestr): # copied from autoobserve.py. JGM Jan 2024
  m = re.match(r'^(\d{4})\-(\d{2})\-(\d{2})[Tt](\d+)\:(\d+)\:(\d+\.?\d*)$', datestr)
  if m is not None:
    gg = m.groups()
    yr, mn, dy = map(int, gg[0:3]) # take the date , month and year of the datestr and turn them into ints
    mjd = palpy.cldj(yr, mn, dy) # func to convert gregorian calendar to modified julian date 

    if len(gg) > 3:
      hh, mm, ss = gg[3:6]
      frac = ((int(hh)*60 + int(mm)) * 60 + float(ss)) / 86400.0 # convert time to seconds then fraction of day 
      mjd += frac # add fraction of day to MJD 

  else:
    mjd = float(datestr) # if datestr is provided as a Julian date, 
    if mjd > 2400000.5: 
      mjd -= 2400000.5 #convert it to Modified Julian Date

  return mjd 

def get_gaia_param_table(gdr2_id): # adapted from query_functions.py. JGM Jan 2024. Omit GDR2 from star id.
  job = Gaia.launch_job("select * "
  "from gaiaedr3.gaia_source as gaia "
  "inner join gaiaedr3.dr2_neighbourhood as xmatch "
  "on gaia.source_id = xmatch.dr3_source_id "
  "where xmatch.dr2_source_id = " + str(gdr2_id))
  r = job.get_results()
  return r

def apra_apde_fromparams(RA, Dec, pmra, pmdec, plx, epoch, mjd, latitude, longitude, height): # adapted from tobs.py. JGM Jan 2024
	# Units for input must be as they appear on Gaia DR2/DR3 catalogue 2023

	# Compute variables to pass to mean2ast
	ra2k = RA * palpy.DD2R
	de2k = Dec * palpy.DD2R
	 
	# Compute twilight and nautical times, and adjust night if needed.
	# mjd = get_mjd()+days_from_today
	mjds = twilight.twilight_init(mjd, latitude, longitude, height) 
	tstart, tend = mjds["nautstart"], mjds["nautend"]
	mjdmidnight = mjds["midnight"]
	lstmidnight = palutil.get_lst(mjdmidnight, mjdmidnight, longitude)

	# Compute precession, nutation, aberration, etc. for midnight.
	amprms = palpy.mappa(2000.0, mjdmidnight)

	# Compute current epoch coordinates via mean2ast
	astra, astde = palutil.mean2ast(ra2k, de2k,pmra / 1000.0, 
				   pmdec / 1000.0, plx / 1000.0, 0,epoch, 
				   mjdmidnight, amprms)

	#quick mean to apparent place (no prop motion or parallax)
	apra, apde = palpy.mapqkz(astra, astde, amprms)

	return apra, apde, mjdmidnight 

def get_moon_sep_and_illum(target_GDR2or3name, mjd, latitude, longitude, height):  # adapted from autoobserve.py. JGM Jan 2024

	# Set up Tierras site.
	#latitude = 31.680889 * palpy.DD2R #radians
	#longitude = -110.878750 * palpy.DD2R #radians
	#height = 2345.0 # meters
	sphi = np.sin(latitude) 
	cphi = np.cos(latitude)

	# get target's parameters from Gaia 
	gdr3_params = get_gaia_param_table(target_GDR2or3name)
	RA, Dec = float(gdr3_params["ra"]), float(gdr3_params["dec"]) #deg 
	pmra, pmdec = float(gdr3_params["pmra"]), float(gdr3_params["pmdec"]) #mas/yr
	plx, epoch = float(gdr3_params["parallax"]), float(gdr3_params["ref_epoch"]) #mas, ref epoch

	# Calculate target's apparent location at mjdmidnight
	apra, apde, mjdmidnight = apra_apde_fromparams(RA, Dec, pmra, pmdec, plx, epoch, mjd, latitude, longitude, height)

	# Calculate Sun and Moon;s apparent location at mjdmidnight.
	sun_apra, sun_apde, sun_diam = palpy.rdplan(mjdmidnight, 0, longitude, latitude) #rdplan: computes approx topocentric apparent RA and Dec of Solar Sysytem object in Radians. 
	moon_apra, moon_apde, moon_diam = palpy.rdplan(mjdmidnight, 3, longitude, latitude) #calculates location of moon at mjdmidnight.

	# Calculate cosine of geocentric elongation of the moon from the sun.
	cosphi = math.sin(sun_apde) * math.sin(moon_apde) + math.cos(sun_apde) * math.cos(moon_apde) * math.cos(sun_apra - moon_apra)

	# Calculate fraction of lunar surface illuminated.
	moon_illum = 0.5*(1.0 - cosphi)

	# Calulate separation between object and moon in radians
	moon_sep = palpy.dsep(apra, apde, moon_apra, moon_apde)

	return moon_sep * palpy.DR2D , moon_illum * 100 # degrees, percent
