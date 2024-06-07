import numpy as np 
import matplotlib.pyplot as plt 
plt.ion() 
import pandas as pd 
# import pyarrow.parquet as pq 
from glob import glob 
from scipy.stats import sigmaclip
from matplotlib import colors
from scipy.optimize import curve_fit 
from photutils.aperture import CircularAperture, aperture_photometry 
from astropy.modeling.functional_models import Gaussian2D
from astropy.visualization import simple_norm 
import copy 
import argparse
from ap_phot import t_or_f, get_flattened_files, set_tierras_permissions
from astropy.io import fits 
from astropy.wcs import WCS
from astropy.time import Time
from astropy.coordinates import SkyCoord
import astropy.units as u 
from astroquery.gaia import Gaia
Gaia.MAIN_GAIA_TABLE = 'gaiadr3.gaia_source'
from astroquery.vizier import Vizier

def flux_model(mags, a):
	return a*10**(-(mags-10)/2.5)

def gaia_query(file_list, plate_scale=0.432):	
	
	# write out the source csv file
	date = file_list[0].parent.parent.parent.name 
	target = file_list[0].parent.parent.name 
	ffname = file_list[0].parent.name 
	source_path = f'/data/tierras/photometry/{date}/{target}/{ffname}/{date}_{target}_sources.csv'

	# use the wcs to evaluate the coordinates of the central pixel in images over the night to determine average pointing
	central_ras = []
	central_decs = []
	for ii in range(len(file_list)):
		with fits.open(file_list[ii]) as hdul:
			wcs = WCS(hdul[0].header)
		im_shape = hdul[0].shape
		sc = wcs.pixel_to_world(im_shape[1]/2-1, im_shape[0]/2-1)
		central_ras.append(sc.ra.value)
		central_decs.append(sc.dec.value)

	# do a 3-sigma clipping and take the mean of the ra/dec lists to represent the average field center over the night 	
	v1, l1, h1 = sigmaclip(central_ras, 3, 3)
	avg_central_ra = np.mean(v1)
	v2, l2, h2 = sigmaclip(central_decs, 3, 3)
	avg_central_dec = np.mean(v2)


	# identify the image closest to the average position 
	central_im_file = file_list[np.argmin(((avg_central_ra-central_ras)**2+(avg_central_dec-central_decs)**2)**0.5)]
	with fits.open(central_im_file) as hdul:
		central_im = hdul[0].data
		header = hdul[0].header
		wcs = WCS(header)

	# get the epoch of these observations 
	tierras_epoch = Time(header['TELDATE'],format='decimalyear')

	# set up the region on sky that we'll query in Gaia
	# to be safe, set the width/height to be a bit larger than the estimates from plate scale alone, and cut to sources that actually fall on the chip after the query is complete
	#	after the query is complete
	coord = SkyCoord(avg_central_ra*u.deg, avg_central_dec*u.deg)
	width = u.Quantity(plate_scale*im_shape[0],u.arcsec)*1.5
	height = u.Quantity(plate_scale*im_shape[1],u.arcsec)*1.5


	# query Gaia DR3 for all the sources in the field brighter than the calculated magnitude limit
	job = Gaia.launch_job_async("""
									SELECT source_id, ra, ra_error, dec, dec_error, ref_epoch, pmra, pmra_error, pmdec, pmdec_error, parallax, parallax_error, parallax_over_error, ruwe, phot_bp_mean_mag, phot_g_mean_mag, phot_rp_mean_mag, phot_bp_mean_flux, phot_bp_mean_flux_error, phot_g_mean_flux, phot_g_mean_flux_error, phot_rp_mean_flux, phot_rp_mean_flux_error, bp_rp, bp_g, g_rp, phot_variable_flag,radial_velocity, radial_velocity_error, non_single_star, teff_gspphot, logg_gspphot, mh_gspphot
							 		FROM gaiadr3.gaia_source as gaia
									WHERE gaia.ra BETWEEN {} AND {} AND
											gaia.dec BETWEEN {} AND {}
							 				
									ORDER BY phot_rp_mean_mag ASC
								""".format(coord.ra.value-width.to(u.deg).value/2, coord.ra.value+width.to(u.deg).value/2, coord.dec.value-height.to(u.deg).value/2, coord.dec.value+height.to(u.deg).value/2)
								)
	res = job.get_results()
	res['SOURCE_ID'].name = 'source_id' # why does this get returned in all caps? 

	# cut to entries without masked pmra values; otherwise the crossmatch will break
	problem_inds = np.where(res['pmra'].mask)[0]

	# set the pmra, pmdec, and parallax of those indices to 0
	res['pmra'][problem_inds] = 0
	res['pmdec'][problem_inds] = 0
	res['parallax'][problem_inds] = 0

	# perform a crossmatch with 2MASS
	gaia_coords = SkyCoord(ra=res['ra'], dec=res['dec'], pm_ra_cosdec=res['pmra'], pm_dec=res['pmdec'], obstime=Time('2016',format='decimalyear'))
	v = Vizier(catalog="II/246",columns=['*','Date'], row_limit=-1)
	twomass_res = v.query_region(coord, width=width, height=height)[0]
	twomass_coords = SkyCoord(twomass_res['RAJ2000'],twomass_res['DEJ2000'])
	twomass_epoch = Time('2000-01-01')
	gaia_coords_tm_epoch = gaia_coords.apply_space_motion(twomass_epoch)
	gaia_coords_tierras_epoch = gaia_coords.apply_space_motion(tierras_epoch)

	idx_gaia, sep2d_gaia, _ = gaia_coords_tm_epoch.match_to_catalog_sky(twomass_coords)
	#Now set problem indices back to NaNs
	res['pmra'][problem_inds] = np.nan
	res['pmdec'][problem_inds] = np.nan
	res['parallax'][problem_inds] = np.nan
	
	# figure out source positions in the Tierras epoch 
	tierras_pixel_coords = wcs.world_to_pixel(gaia_coords_tierras_epoch)

	# add 2MASS data and pixel positions to the source table
	res.add_column(twomass_res['_2MASS'][idx_gaia],name='2MASS',index=1)
	res.add_column(tierras_pixel_coords[0],name='X pix', index=2)
	res.add_column(tierras_pixel_coords[1],name='Y pix', index=3)
	res.add_column(gaia_coords_tierras_epoch.ra, name='ra_tierras', index=4)
	res.add_column(gaia_coords_tierras_epoch.dec, name='dec_tierras', index=5)
	res['Jmag'] = twomass_res['Jmag'][idx_gaia]
	res['e_Jmag'] = twomass_res['e_Jmag'][idx_gaia]
	res['Hmag'] = twomass_res['Hmag'][idx_gaia]
	res['e_Hmag'] = twomass_res['e_Hmag'][idx_gaia]
	res['Kmag'] = twomass_res['Kmag'][idx_gaia]
	res['e_Kmag'] = twomass_res['e_Kmag'][idx_gaia]
	
	# determine which chip the sources fall on 
	# 0 = bottom, 1 = top 
	chip_inds = np.zeros(len(res),dtype='int')
	chip_inds[np.where(res['Y pix'] >= 1023)] = 1
	res.add_column(chip_inds, name='Chip')

	#Cut to sources that actually fall in the image
	use_inds = np.where((tierras_pixel_coords[0]>0)&(tierras_pixel_coords[0]<im_shape[1]-1)&(tierras_pixel_coords[1]>0)&(tierras_pixel_coords[1]<im_shape[0]-1))[0]
	res = res[use_inds]
	res_full = copy.deepcopy(res)	

	return res_full


def main(raw_args=None):
	# set some constants
	GAIN = 5.9 # e- ADU^-1
	PLATE_SCALE = 0.432 # " pix^-1
	bin_mins = 20 # size of bins in minutes for the binned panel
	contamination_limit = 0.1 
	contaminant_grid_size = 50

	ap = argparse.ArgumentParser()
	ap.add_argument("-field", required=True, help="Name of observed target field exactly as shown in raw FITS files.")
	ap.add_argument("-date", required=True, help="Calendar date of observations.")
	ap.add_argument("-ffname", required=False, default='flat0000', help="Name of flat directory")
	ap.add_argument("-cut_contaminated", required=False, default=True, help="whether or not to perform contamination analysis")

	args = ap.parse_args()
	date = args.date
	field = args.field 
	ffname = args.ffname
	cut_contaminated = t_or_f(args.cut_contaminated)


	# read source df so we can access Rp magnitudes and pixel positions
	sources = pd.read_csv(f'/data/tierras/photometry/{date}/{field}/{ffname}/{date}_{field}_sources.csv')

	# read ancillary data so we can access airmasses and exposure times (for scintillation calculation)
	ancillary_data = pd.read_parquet(f'/data/tierras/photometry/{date}/{field}/{ffname}/{date}_{field}_ancillary_data.parquet')

	# read photometry data so we can access mean sky counts
	# TODO: GENERALIZE!!!
	phot_df = pd.read_parquet(f'/data/tierras/photometry/{date}/{field}/{ffname}/{date}_{field}_circular_fixed_ap_phot_5.parquet')

	# identify all the light curves 
	# lcs = glob('input/internight_precision/data/**global_lc.csv')
	lcs = glob(f'/data/tierras/lightcurves/{date}/{field}/{ffname}/**_lc.csv')
	if len(lcs) == 0:
		raise RuntimeError(f'No light curves for {field} on {date}! Run analyze_night.py first.')

	images = get_flattened_files(date, field, ffname)

	rp_mags = []
	sigmas = []
	ap_radii = []
	source_fluxes = []
	source_ids = []
	skies = []

	times_save = []
	flux_save = []
	flux_err_save = []

	if cut_contaminated:
		# we don't perform photometry on *every* source in the field, so to get an accurate estimate of the contamination for each source, we need to query gaia for all sources (i.e. with no Rp mag limit) so that their fluxes can be modeled
		all_field_sources = gaia_query(images)
		xx, yy = np.meshgrid(np.arange(-int(contaminant_grid_size/2), int(contaminant_grid_size)/2), np.arange(-int(contaminant_grid_size/2), int(contaminant_grid_size/2))) # grid of pixels over which to simulate images for contamination estimate
		seeing_fwhm = np.nanmedian(ancillary_data['FWHM X']) / PLATE_SCALE # get median seeing on this night in pixels for contamination estimate
		seeing_sigma = seeing_fwhm / (2*np.sqrt(2*np.log(2))) # convert from FWHM in pixels to sigma in pixels (for 2D Gaussian modeling in contamination estimate)

	# read in data, skipping sources if they're contaminated
	n_lcs = len(lcs)
	for i in range(n_lcs):
		print(f'{i+1} of {n_lcs}')

		with open(lcs[i], 'r') as f:
			comment = f.readline()
		ap_radii.append(int(comment.split('_')[-1].split('.')[0]))

		df = pd.read_csv(lcs[i], comment='#', dtype=np.float64)
		source_id = lcs[i].split('_')[-2]
		if source_id == 'TIC362144730':
			TICi = i
			print ('index of TIC362144730 in rp_mags is {}'.format(TICi))
			source_id = 'Gaia DR3 4147112604476417792'
		source_id = int(source_id.split(' ')[-1])
		source_ind = np.where(sources['source_id'] == source_id)[0][0] 
		source_rp = sources['phot_rp_mean_mag'][source_ind]
		source_x = sources['X pix'][source_ind]
		source_y = sources['Y pix'][source_ind]

		if cut_contaminated:
			# estimate contamination 
			distances = np.array(np.sqrt((source_x-all_field_sources['X pix'])**2+(source_y-all_field_sources['Y pix'])**2))
			nearby_inds = np.where((distances <= contaminant_grid_size) & (distances != 0))[0]
			if len(nearby_inds) > 0:
				source_rp = sources['phot_rp_mean_mag'][source_ind]
				source_x = sources['X pix'][source_ind]
				source_y = sources['Y pix'][source_ind]
				source_G = sources['gq_photogeo'][source_ind]
				source_bp_rp = sources['bp_rp'][source_ind]

				nearby_rp = np.array(all_field_sources['phot_rp_mean_mag'][nearby_inds])
				nearby_x = np.array(all_field_sources['X pix'][nearby_inds] - source_x)
				nearby_y = np.array(all_field_sources['Y pix'][nearby_inds] - source_y)

				# sometimes the rp mag is nan, remove these entries
				use_inds = np.where(~np.isnan(nearby_rp))[0]
				nearby_rp = nearby_rp[use_inds]
				nearby_x = nearby_x[use_inds]
				nearby_y = nearby_y[use_inds]

				# model the source in question as a 2D gaussian
				source_model = Gaussian2D(x_mean=0, y_mean=0, amplitude=1/(2*np.pi*seeing_sigma**2), x_stddev=seeing_sigma, y_stddev=seeing_sigma)
				sim_img = source_model(xx, yy)

				# add in gaussian models for the nearby sources
				for jj in range(len(nearby_rp)):
					flux = 10**(-(nearby_rp[jj]-source_rp)/2.5)
					contaminant_model = Gaussian2D(x_mean=nearby_x[jj], y_mean=nearby_y[jj], amplitude=flux/(2*np.pi*seeing_sigma**2), x_stddev=seeing_sigma, y_stddev=seeing_sigma)
					contaminant = contaminant_model(xx,yy)
					sim_img += contaminant	

				# estimate contamination by doing aperture photometry on the simulated image
				# if the measured flux exceeds 1 by a chosen threshold, record the source's index so it can be removed
				ap = CircularAperture((sim_img.shape[1]/2, sim_img.shape[0]/2), r=ap_radii[i])
				phot_table = aperture_photometry(sim_img, ap)
				contamination = phot_table['aperture_sum'][0] - 1

				# if source_id == 4147121464971195776:
				# 	breakpoint()

				# if the contamination is over the limit, exclude this source from the analysis
				if contamination > contamination_limit and source_id != 'Gaia DR3 4147112604476417792':
					# plt.figure()
					# plt.imshow(sim_img, origin='lower', norm=simple_norm(sim_img, min_percent=1, max_percent=98))
					# breakpoint()
					continue

		rp_mags.append(source_rp)
		source_ids.append(source_id)
		
		times = np.array(df['BJD TDB'])
		flux = np.array(df['Flux'])
		flux_err = np.array(df['Flux Error'])

		nan_inds = ~np.isnan(flux)
		times = times[nan_inds]
		flux = flux[nan_inds]
		flux_err = flux_err[nan_inds]

		v, l, h = sigmaclip(flux)
		sc_inds = np.where((flux > l) & (flux < h))[0]
		times = times[sc_inds]
		flux = flux[sc_inds]
		flux_err = flux_err[sc_inds]

		times_save.append(times)
		flux_save.append(flux)
		flux_err_save.append(flux_err)
		sigmas.append(np.nanstd(flux))

		source_fluxes.append(np.nanmedian(phot_df[f'S{source_ind} Source-Sky']))
		skies.append(np.nanmedian(phot_df[f'S{source_ind} Sky']))
		if source_id == 4147126515852964736:
			breakpoint()

	n_lcs = len(rp_mags)
	rp_mags = np.array(rp_mags)
	source_ids = np.array(source_ids)
	sigmas = np.array(sigmas)
	ap_radii = np.array(ap_radii)
	source_fluxes = np.array(source_fluxes)
	skies = np.array(skies)
	times_save = np.array(times_save, dtype='object')
	flux_save = np.array(flux_save, dtype='object')
	flux_err_save = np.array(flux_err_save, dtype='object')

	sort = np.argsort(rp_mags)
	rp_mags = rp_mags[sort]
	print (rp_mags)
	tic_ind = np.argmin(np.abs(rp_mags - 11.882206))
	print (tic_ind)
	source_ids = source_ids[sort]
	sigmas = sigmas[sort]
	ap_radii = ap_radii[sort]
	source_fluxes = source_fluxes[sort]
	skies = skies[sort]
	times_save = times_save[sort]
	flux_save = flux_save[sort]
	flux_err_save = flux_err_save[sort]

	# determine the exposure time
	exp_times = np.unique(ancillary_data['Exposure Time'])
	if len(exp_times) > 1:
		raise RuntimeError('Handling of multiple exposure times not yet implemented!')
	else:
		exp_time = exp_times[0]

	# calculate mean scintillation estimate
	scint = np.mean(np.array(1.5*0.09*130**(-2/3)*ancillary_data['Airmass']**(7/4)*(2*ancillary_data['Exposure Time'])**(-1/2)*np.exp(-2306/8000)))

	# calculate mean sky 
	mean_sky = np.nanmean(skies)

	#   figure out which aperture radius was the most commonly selected
	#   we need this so we can get the number of pixels in the aperture for the sky background
	# noise calculation 
	#   NOTE that in its current format, the light curves use a variety of aperture sizes; each 
	# is the radius that minimized the scatter on 5-minute timescales. 
	#   generally, the faint stars will use smaller radii than the bright stars, since less of
	# PSFs emerge above the background noise
	#   as long as the faint stars are all using the same (or at least similar) aperture radii, 
	# the plot will make sense, since we only become background dominated for faint stars. 

	ap_radii_counts, ap_radii_bins = np.histogram(ap_radii, bins=np.arange(5,20)) # NOTE that this will fail if we introduce new aperture sizes!
	most_common_ap = ap_radii_bins[np.argmax(ap_radii_counts)]
	n_pix = np.pi*most_common_ap**2

	# set a grid of Rp mags over which we'll evaluate the components of our noise models
	rp_mag_grid = np.arange(int(np.floor(np.nanmin(rp_mags)))-0.1,int(np.ceil(np.nanmax(rp_mags)))+0.2,0.05)

	# calculate a smooth function of the flux for each source by fitting a*10**(-(rp-10)/2.5) to the source count rates
	fit_inds = np.where(rp_mags > 12)[0] # do the fitting on non-saturated sources. TODO figure out how to do programattically 
	fit_x = rp_mags[fit_inds]
	fit_y = source_fluxes[fit_inds]/exp_time*GAIN
	popt, pcov = curve_fit(flux_model, fit_x, fit_y, sigma=np.sqrt(fit_y), p0=[23635])

	# adjust so that the median is 1
	expected = popt[0]*10**(-(rp_mags-10)/2.5)*exp_time/GAIN
	observed = source_fluxes
	correction_factor = np.nanmedian(observed/expected)
	popt[0] *= correction_factor 

	plt.plot(rp_mags, source_fluxes, marker='.', ls='')
	plt.plot(rp_mag_grid, popt[0]*10**(-(rp_mag_grid-10)/2.5)*exp_time/GAIN)
	plt.yscale('log')
	# breakpoint()
	expected_fluxes = popt[0]*10**(-(rp_mag_grid-10)/2.5)*exp_time/GAIN


	sky_photon_noise = np.sqrt(n_pix*mean_sky*GAIN)/(expected_fluxes*GAIN)
	source_photon_noise = np.sqrt(expected_fluxes*GAIN)/(expected_fluxes*GAIN)
	scintillation_noise = np.zeros(len(rp_mag_grid))+scint
	# read_noise = np.sqrt(15.5**2*n_pix)/(expected_fluxes*GAIN)
	pwv_noise = np.zeros(len(rp_mag_grid)) + 230*1e-6
	total_noise_model = np.sqrt(source_photon_noise**2 + sky_photon_noise**2 + scintillation_noise**2 + pwv_noise**2)

	fig, ax = plt.subplots(1, 2, figsize=(11,6), sharey=True, sharex=True)
	for a in ax:
		a.spines['top'].set_linewidth(2)
		a.spines['right'].set_linewidth(2)
		a.spines['bottom'].set_linewidth(2)
		a.spines['left'].set_linewidth(2)
		a.tick_params(axis='both',width=2,length=6,direction='in',which='major')
		a.tick_params(axis='both',width=1,length=4,direction='in',which='minor')

	ax[0].set_title(f'Native cadence ({exp_time} s)', fontsize=14)
	ax[0].plot(rp_mag_grid, total_noise_model*1e6, lw=2, label='$\sigma_{total}$ = $\sqrt{ \sigma_{source}^2 + \sigma_{sky}^2 + \sigma_{scintillation}^2 + \sigma_{PWV}^2 }$', zorder=1)
	ax[0].plot(rp_mag_grid, source_photon_noise*1e6, label='$\sigma_{source}$', zorder=1)
	ax[0].plot(rp_mag_grid, sky_photon_noise*1e6, label='$\sigma_{sky}$', zorder=1)
	# ax[0].plot(rp_mag_grid, read_noise*1e6, label='$\sigma_{read}$', zorder=1)
	ax[0].plot(rp_mag_grid, scintillation_noise*1e6, label='$\sigma_{scintillation}$')
	ax[0].plot(rp_mag_grid, pwv_noise*1e6, label='$\sigma_{PWV} = 230 \, {ppm}$')

	ax[0].plot(rp_mags, sigmas*1e6, marker='.', color='k', alpha=0.4, ls='', zorder=0, ms=3)
	ax[0].plot(rp_mags[tic_ind], sigmas[tic_ind]*1e6, marker='*',  markeredgecolor='black', ms=10,markerfacecolor='white', ls='', zorder=10,label = 'LEP 1805-1422')


	# h2d_bins = [np.linspace(10.6, 17, 50), np.logspace(2,5,75)]
	# h2d_cmin = 10
	# h2d = ax[0].hist2d(rp_mags, sigmas*1e6, bins=h2d_bins, cmin=h2d_cmin, norm=colors.PowerNorm(0.5), zorder=3, alpha=0.9, lw=0)
	# cb = fig.colorbar(h2d[3], ax=ax[0], pad=0.02, label='N$_{sources}$')

	ax[0].set_xlabel('$G_{\mathrm{RP}}$', fontsize=14)
	ax[0].set_ylabel('$\sigma$ (ppm)', fontsize=14)
	ax[0].set_yscale('log')
	ax[0].grid(True, which='both', alpha=0.5)
	ax[0].set_xlim(10.6,np.ceil(np.nanmax(rp_mags))+0.1)
	ax[0].tick_params(labelsize=12)
	ax[0].legend()

	# now bin the data
	ppb = int(bin_mins*60/exp_time)
	binned_sigmas = np.zeros(n_lcs)
	for i in range(n_lcs):
		times = times_save[i]
		flux = flux_save[i] 
		flux_err = flux_err_save[i]

		n_bins = int(np.ceil(len(flux)/ppb))
		# bx = np.zeros(n_bins)
		by = np.zeros(n_bins)
		for j in range(n_bins):
			if j != n_bins-1:
				bin_inds = np.arange(j*ppb, (j+1)*ppb)
			else:
				continue # SKIP non-full bins
				# bin_inds = np.arange(j*ppb, len(flux))
			by[j] = np.nanmedian(flux[bin_inds])
		binned_sigmas[i] = np.nanstd(by[np.where(by!=0)[0]])

	ax[1].plot(rp_mags, binned_sigmas*1e6, marker='.', color='k', alpha=0.4, ls='', zorder=0, ms=3)
	ax[1].plot(rp_mags[tic_ind], sigmas[tic_ind]*1e6, marker='*',  markeredgecolor='black', ms=10,markerfacecolor='white', ls='', zorder=10,label = 'LEP 1805-1422')

	# h2d = ax[1].hist2d(rp_mags, binned_sigmas*1e6, bins=h2d_bins, cmin=h2d_cmin, norm=colors.PowerNorm(0.5), zorder=3, alpha=0.9, lw=0)
	# cb = fig.colorbar(h2d[3], ax=ax[1], pad=0.02, label='N$_{sources}$')

	ax[1].plot(rp_mag_grid, total_noise_model/np.sqrt(ppb)*1e6, lw=2)
	ax[1].plot(rp_mag_grid, source_photon_noise*1e6/np.sqrt(ppb), label='Source photon noise', zorder=1)
	ax[1].plot(rp_mag_grid, sky_photon_noise*1e6/np.sqrt(ppb), label='Sky photon noise', zorder=1)
	# ax[1].plot(rp_mag_grid, read_noise*1e6/np.sqrt(ppb), label='Read noise', zorder=1)
	ax[1].plot(rp_mag_grid, scintillation_noise*1e6/np.sqrt(ppb), label='Scintillation')
	ax[1].plot(rp_mag_grid, pwv_noise*1e6, label='PWV')
	ax[1].grid(True, which='both', alpha=0.5)
	ax[1].set_title(f'Binned ({ppb} min)', fontsize=14)
	ax[1].set_xlabel('$G_{\mathrm{RP}}$', fontsize=14)
	ax[1].tick_params(labelsize=12)
	ax[1].set_ylim(100, 3e5)
	plt.tight_layout()

	# plt.figure()
	# plt.plot(binned_sigmas/sigmas, ls='', marker='.', label='$\sigma_{'+f'{bin_mins}'+'-min}$/$\sigma_{60-s}$')
	# plt.axhline(1/np.sqrt(ppb), color='tab:orange', label=f'Expected decrease = {1/np.sqrt(ppb):.2f} (1/sqrt({ppb}))')
	# plt.axhline(np.nanmedian(binned_sigmas/sigmas), color='tab:red', label=f'Measured median decrease = {np.nanmedian(binned_sigmas/sigmas):.2f}')
	# plt.legend()
	plt.savefig(f'/data/tierras/lightcurves/{date}/{field}/{ffname}/{date}_{field}_precision.png',dpi=300, transparent=True)
	set_tierras_permissions(f'/data/tierras/lightcurves/{date}/{field}/{ffname}/{date}_{field}_precision.png')
	#plt.close()

if __name__ == '__main__':
	main()
