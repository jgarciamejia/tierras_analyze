import argparse
import numpy as np 
import pandas as pd 
import glob 
import matplotlib.pyplot as plt 
plt.ion()
from scipy.stats import sigmaclip
import pyarrow.parquet as pq 
from astropy.io import fits 
from ap_phot import get_flattened_files, t_or_f
from astropy.visualization import simple_norm 
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import TextBox, Button
from analyze_global import identify_target_gaia_id
from matplotlib import colors
from internight_precision_interactive import gaia_query
from photutils.aperture import CircularAperture, aperture_photometry 
from astropy.modeling.functional_models import Gaussian2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.widgets import CheckButtons

def main(raw_args=None):

	def on_click(event):
		global highlight 
		global currently_selected_source

		# print(event.inaxes)
		ax = event.inaxes
		if ax is not None:
			label = axes_mapping.get(ax, 'Unknown axis')
		else:
			return

		if label == 'ax1':
			if check_all.get_status()[0] is False:
				check_all.set_active(0) # set the quality flag check box to True if it is not already; by default, we plot newly selected targets with all quality flags enabled
			x_click = event.xdata
			y_click = event.ydata
			dists = ((rp_mags - x_click)**2+(np.log10(night_to_nights) - np.log10(y_click))**2)**0.5
			point = np.argmin(dists)
			# print(point)
			# print(x_click)
			# print(y_click)
		elif label == 'ax2':
			return
		elif label == 'ax3':
			if check_all.get_status()[0] is False:
				check_all.set_active(0) # set the quality flag check box to True if it is not already; by default, we plot newly selected targets with all quality flags enabled
			x_click = event.xdata
			y_click = event.ydata
			dists = ((x_pos - x_click)**2+(y_pos- y_click)**2)**0.5
			point = np.argmin(dists)
		elif label == 'ax4':
			if check_all.get_status()[0] is False:
				check_all.set_active(0) # set the quality flag check box to True if it is not already; by default, we plot newly selected targets with all quality flags enabled
			x_click = event.xdata
			y_click = event.ydata
			dists = ((bp_rp - x_click)**2+(G - y_click)**2)**0.5
			point = np.nanargmin(dists)
		else:
			return
				
		# all_mask = ~global_flux_flags[point] & ~global_wcs_flags[point] & ~global_position_flags[point] & ~global_fwhm_flags[point]
		mask_is_checked = check_all.get_status()[0]
		sc_is_checked = check_all.get_status()[1]

		if mask_is_checked and not sc_is_checked:
			all_mask = np.where(np.logical_not(global_wcs_flags[point]).astype(int) & np.logical_not(global_position_flags[point]).astype(int) & np.logical_not(global_fwhm_flags[point]).astype(int) & np.logical_not(global_flux_flags[point]).astype(int))[0] 
		elif sc_is_checked and not mask_is_checked:
			nan_inds = ~np.isnan(flux_arr[point])
			flux_ = flux_arr[point][nan_inds]
			v, l, h = sigmaclip(flux_, 4, 4)
			all_mask = (flux_arr[point] >= l) & (flux_arr[point] <= h)
				
		elif mask_is_checked and sc_is_checked:
			# make a combined mask using all the quality flags and a sigma clipping mask
			nan_inds = ~np.isnan(flux_arr[point])
			flux_ = flux_arr[point][nan_inds]
			v, l, h = sigmaclip(flux_, 4, 4)
			sc_mask = (flux_arr[point] < l) | (flux_arr[point] > h)
			all_mask = np.where(np.logical_not(global_wcs_flags[point]).astype(int) & np.logical_not(global_position_flags[point]).astype(int) & np.logical_not(global_fwhm_flags[point]).astype(int) & np.logical_not(global_flux_flags[point]).astype(int) & np.logical_not(sc_mask).astype(int))[0] 		
		else:
			all_mask = np.arange(len(flux_arr[point]))

		# recompute night medians 
		bx_ = np.zeros(len(night_medians))
		by_ = np.zeros_like(bx_)
		bye_ = np.zeros_like(bx_)
		for j in range(len(night_medians)):				
			times_night = times_list[j]
			inds = np.where((times_arr[point][all_mask] >= times_night[0]) & (times_arr[point][all_mask] <= times_night[-1]))[0]
			bx_[j] = np.nanmedian(times_arr[point][all_mask][inds])
			try:
				by_[j] = np.nanmedian(flux_arr[point][all_mask][inds])
			except:
				print(inds)
				
				breakpoint()
			n_points = sum(~np.isnan(flux_arr[point][all_mask][inds]))
			bye_[j] = np.nanstd(flux_arr[point][all_mask][inds]) / np.sqrt(n_points)


		print(f'Clicked on {source_ids[point]}, sigma_n2n = {night_to_nights[point]*1e6:.0f}')
		if highlight:
			ax1.lines[-1].remove()
			ax3.lines[-1].remove()
			ax4.lines[-1].remove()
			highlight = None

		highlight = ax1.plot(rp_mags[point], night_to_nights[point], marker='o', color='#FFAA33', mec='k', mew=1.5, ls='')

		# highlight_3 = ax3.plot(rp_mags[point], binned_sigmas[point]*1e6, 'mo')

		highlight_4 = ax4.plot(bp_rp[point], G[point], marker='o', color='#FFAA33', mec='k', mew=1.5, ls='')

		# clear out the previous lc and plot the new one
		ax2.cla()
		# ax2.errorbar(times_save[point], flux_save[point], flux_err_save[point], marker='.', ls='', ecolor='#b0b0b0', color='#b0b0b0')
		# ax2.errorbar(bin_times_save[point], bin_flux_save[point], bin_flux_err_save[point], marker='o', color='k', ecolor='k', mfc='none', zorder=3, mew=1.5, ms=7, ls='')
		ax2.errorbar(times_arr[point][all_mask], flux_arr[point][all_mask], flux_err_arr[point][all_mask], marker='.', color='#b0b0b0', ecolor='#b0b0b0', ls='', label=source_ids[point])
		ax2.errorbar(bx_, by_, bye_, marker='o', color='k', ls='', zorder=3)
		ax2.axhline(1, ls='--', lw=2, color='k', alpha=0.7, zorder=0)
		ax2.grid(alpha=0.5)
		ax2.tick_params(labelsize=12)
		ax2.set_ylabel('Normalized Flux', fontsize=14)
		# ax2.set_title(source_ids[point], fontsize=14)
		ax2.legend()

		# # update the field plot 
		ax3.set_xlim(x_pos[point]-50, x_pos[point]+50)
		ax3.set_ylim(y_pos[point]-50, y_pos[point]+50)

		# update sky background and x/y pixel positions
		# nan out any frames that are nan in the flux array for this source 
		nan_inds = np.where(np.isnan(flux_arr[point]))[0]
		sky_plot = global_sky_array[point]
		x_plot = global_x_array[point]
		y_plot = global_y_array[point]	
		sky_plot[nan_inds] = np.nan
		x_plot[nan_inds] = np.nan 
		y_plot[nan_inds] = np.nan 
		x_plot -= np.nanmedian(x_plot)
		y_plot -= np.nanmedian(y_plot)

		# plot airmass 
		ax5.cla()
		ax5.plot(global_ancillary_data['BJD TDB'][all_mask]-x_offset, global_ancillary_data['Airmass'][all_mask], marker='.', ls='')
		ax5.set_ylabel('Airmass')
		
		ax6.cla()
		ax6.plot(times[all_mask], sky_plot[all_mask], marker='.', ls='', color='tab:cyan')
		ax6.set_ylabel('Sky (ADU/s)')

		ax7.cla()
		ax7.plot(times[all_mask], x_plot[all_mask], marker='.', ls='', color='tab:green', label='X-med(X)')
		ax7.plot(times[all_mask], y_plot[all_mask], marker='.', ls='', color='tab:red', label='Y-med(Y)')		
		ax7.legend()
		ax7.set_ylabel('Pos (pix.)')

		plt.subplots_adjust(hspace=0.35)

		currently_selected_source = source_ids[point]

	def on_click_2(event):
		global highlight_2
		# print(event.inaxes)
		ax_2 = event.inaxes
		if ax_2 is not None:
			label = axes_mapping_2.get(ax_2, 'Unknown axis')
		else:
			return

		if label == 'ax1_2':
			x_click = event.xdata
			y_click = event.ydata
			dists = ((x_pos - x_click)**2+(y_pos - y_click)**2)**0.5
			point = np.nanargmin(dists)
			print(point)
			print(x_click)
			print(y_click)
		elif label == 'ax2_2':
			return
		else:
			return
		
		print(f'Clicked on {source_ids[point]}, sigma_n2n = {night_to_nights[point]*1e6:.0f}')
		if highlight_2:
			ax1_2.lines[-1].remove()
			highlight_2 = None

		highlight_2 = ax1_2.plot(x_pos[point], y_pos[point], marker='o', color='#FFAA33', mec='k', mew=1.5, ls='')

		# clear out the previous lc and plot the new one
		ax2_2.cla()
		ax2_2.errorbar(times_arr[point], flux_arr[point], flux_err_arr[point], marker='.', color='#b0b0b0', ecolor='#b0b0b0', ls='')
		ax2_2.errorbar(bx_arr[point], by_arr[point], bye_arr[point], marker='o', color='k', ls='', zorder=3)
		ax2_2.axhline(1, ls='--', lw=2, color='k', alpha=0.7, zorder=0)
		ax2_2.grid(alpha=0.5)
		ax2_2.tick_params(labelsize=12)
		ax2_2.set_xlabel('Time (BJD$_{TDB}$)', fontsize=14)
		ax2_2.set_ylabel('Normalized Flux', fontsize=14)
		ax2_2.set_title(source_ids[point], fontsize=14)
		return 

	def on_click_3(event):
		global highlight_3 
		# print(event.inaxes)
		ax_3 = event.inaxes
		if ax_3 is not None:
			label = axes_mapping_3.get(ax_3, 'Unknown axis')
		else:
			return

		if label == 'ax1_3':
			x_click = event.xdata
			y_click = event.ydata
			dists = ((bp_rp - x_click)**2+(G - y_click)**2)**0.5
			point = np.nanargmin(dists)
			print(point)
			print(x_click)
			print(y_click)
		elif label == 'ax2_3':
			return
		else:
			return
		
		print(f'Clicked on {source_ids[point]}, sigma_n2n = {night_to_nights[point]*1e6:.0f}')
		if highlight_3:
			ax1_3.lines[-1].remove()
			highlight_3 = None

		highlight_3 = ax1_3.plot(bp_rp[point], G[point], marker='o', color='#FFAA33', mec='k', mew=1.5, ls='')



		# clear out the previous lc and plot the new one
		ax2_3.cla()
		ax2_3.errorbar(times_arr[point], flux_arr[point], flux_err_arr[point], marker='.', color='#b0b0b0', ecolor='#b0b0b0', ls='')
		ax2_3.errorbar(bx_arr[point], by_arr[point], bye_arr[point], marker='o', color='k', ls='', zorder=3)
		ax2_3.axhline(1, ls='--', lw=2, color='k', alpha=0.7, zorder=0)
		ax2_3.grid(alpha=0.5)
		ax2_3.tick_params(labelsize=12)
		ax2_3.set_xlabel('Time (BJD$_{TDB}$)', fontsize=14)
		ax2_3.set_ylabel('Normalized Flux', fontsize=14)
		ax2_3.set_title(source_ids[point], fontsize=14)
		return 
	
	def update_all_flag_visibility(label):
		try:
			point = np.where(source_ids == currently_selected_source)[0][0]
		except:
			print('No source selected!')
			return 
		

		mask_is_checked = check_all.get_status()[0]
		sc_is_checked = check_all.get_status()[1]

		# if the box is checked, get the indices of the good exposures using the masks
		if mask_is_checked and not sc_is_checked:
			all_mask = np.where(np.logical_not(global_wcs_flags[point]).astype(int) & np.logical_not(global_position_flags[point]).astype(int) & np.logical_not(global_fwhm_flags[point]).astype(int) & np.logical_not(global_flux_flags[point]).astype(int))[0] 
		elif sc_is_checked and not mask_is_checked:
			nan_inds = ~np.isnan(flux_arr[point])
			flux_ = flux_arr[point][nan_inds]
			v, l, h = sigmaclip(flux_, 4, 4)
			all_mask = (flux_arr[point] >= l) & (flux_arr[point] <= h)
				
		elif mask_is_checked and sc_is_checked:
			# make a combined mask using all the quality flags and a sigma clipping mask
			nan_inds = ~np.isnan(flux_arr[point])
			flux_ = flux_arr[point][nan_inds]
			v, l, h = sigmaclip(flux_, 4, 4)
			sc_mask = (flux_arr[point] < l) | (flux_arr[point] > h)
			all_mask = np.where(np.logical_not(global_wcs_flags[point]).astype(int) & np.logical_not(global_position_flags[point]).astype(int) & np.logical_not(global_fwhm_flags[point]).astype(int) & np.logical_not(global_flux_flags[point]).astype(int) & np.logical_not(sc_mask).astype(int))[0] 		
		else:
			all_mask = np.arange(len(flux_arr[point]))

		# recompute night medians 
		bx_ = np.zeros(len(night_medians))
		by_ = np.zeros_like(bx_)
		bye_ = np.zeros_like(bx_)
		for j in range(len(night_medians)):				
			times_night = times_list[j]
			inds = np.where((times_arr[point][all_mask] >= times_night[0]) & (times_arr[point][all_mask] <= times_night[-1]))[0]
			bx_[j] = np.nanmedian(times_arr[point][all_mask][inds])
			by_[j] = np.nanmedian(flux_arr[point][all_mask][inds])
			n_points = sum(~np.isnan(flux_arr[point][all_mask][inds]))
			bye_[j] = np.nanstd(flux_arr[point][all_mask][inds]) / np.sqrt(n_points)

		ax2.cla()
		
		ax2.errorbar(times_arr[point][all_mask], flux_arr[point][all_mask], flux_err_arr[point][all_mask], marker='.', color='#b0b0b0', ecolor='#b0b0b0', ls='', label=source_ids[point])
		ax2.errorbar(bx_, by_, bye_, marker='o', color='k', ls='', zorder=3)
		ax2.axhline(1, ls='--', lw=2, color='k', alpha=0.7, zorder=0)
		ax2.grid(alpha=0.5)
		ax2.tick_params(labelsize=12)
		ax2.set_ylabel('Normalized Flux', fontsize=14)
		# ax2.set_title(source_ids[point], fontsize=14)
		ax2.legend()
		
		print(f'sigma_n2n = {np.nanstd(by_)*1e6:.0f}')

		# plot airmass 
		ax5.cla()
		ax5.plot(global_ancillary_data['BJD TDB'][all_mask]-x_offset, global_ancillary_data['Airmass'][all_mask], marker='.', ls='')
		ax5.set_ylabel('Airmass')

		ax6.cla()
		ax6.plot(times_arr[point][all_mask], global_sky_array[point][all_mask], marker='.', ls='', color='tab:cyan')
		ax6.set_ylabel('Sky (ADU/s)')

		ax7.cla()
		ax7.plot(times_arr[point][all_mask], global_x_array[point][all_mask], marker='.', ls='', color='tab:green', label='X-med(X)')
		ax7.plot(times_arr[point][all_mask], global_y_array[point][all_mask], marker='.', ls='', color='tab:red', label='Y-med(Y)')		
		ax7.legend()
		ax7.set_ylabel('Pos (pix.)')

		ax8.cla()
		ax8.plot(global_ancillary_data['BJD TDB'][all_mask]-x_offset, global_ancillary_data['FWHM X'][all_mask], marker='.', ls='', label='Major axis', color='tab:pink')
		ax8.plot(global_ancillary_data['BJD TDB'][all_mask]-x_offset, global_ancillary_data['FWHM Y'][all_mask], marker='.', ls='', label='Minor axis', color='tab:purple')
		ax8.legend()
		ax8.set_ylabel('FWHM (")')

		return

	
	# set some constants
	PLATE_SCALE = 0.432 # " pix^-1
	contamination_limit = 0.1 
	contaminant_grid_size = 50

	ap = argparse.ArgumentParser()
	ap.add_argument("-field", required=True, help="Name of observed target field exactly as shown in raw FITS files.")
	ap.add_argument("-ffname", required=False, default='flat0000', help="Name of flat directory")
	ap.add_argument("-cut_contaminated", required=False, default=False, help="whether or not to perform contamination analysis")
	args = ap.parse_args()
	field = args.field 
	ffname = args.ffname
	cut_contaminated = t_or_f(args.cut_contaminated)

	# get global ancillary data 
	global_ancillary_data = pd.read_csv(f'/data/tierras/fields/{field}/global_ancillary_data.csv')

	# get dates on which field was observed so we can plot an image of the field
	date_list = glob.glob(f'/data/tierras/photometry/**/{field}/{ffname}')	
	date_list = np.array(sorted(date_list, key=lambda x:int(x.split('/')[4]))) # sort on date so that everything is in order
	if field == 'TIC362144730':
		date_list = np.delete(date_list, [-1])

	# read in weights csv
	weights_df = pd.read_csv(f'/data/tierras/fields/{field}/sources/lightcurves/{ffname}/weights.csv')
	ref_ids = np.array(weights_df['Ref ID'])
	weights_df = np.array(weights_df)

	# read in the ancillary data from each night
	# ancillary_dfs = []
	best_fwhm_metric = 9999.
	for i in range(len(date_list)):
		date = date_list[i].split('/')[4]
		# ancillary_dfs.append(pq.read_table(date_list[i]+f'/{date}_{field}_ancillary_data.parquet'))
		try:
			tab = pq.read_table(date_list[i]+f'/{date}_{field}_ancillary_data.parquet')
		except:
			continue 
		fwhm_x = np.array(tab['FWHM X'])
		fwhm_y = np.array(tab['FWHM Y'])
		min_fwhm_diff_ind = np.argmin(fwhm_x-fwhm_y)
		fwhm_metric = fwhm_x[min_fwhm_diff_ind] * (fwhm_x[min_fwhm_diff_ind]-fwhm_y[min_fwhm_diff_ind]) 
		if fwhm_metric < best_fwhm_metric:
			print(fwhm_x[min_fwhm_diff_ind], fwhm_x[min_fwhm_diff_ind]-fwhm_y[min_fwhm_diff_ind])
			best_date = date 
			best_im_ind = min_fwhm_diff_ind
			best_fwhm_metric = fwhm_metric


	source_df = pd.read_csv(f'/data/tierras/photometry/{best_date}/{field}/{ffname}/{best_date}_{field}_sources.csv')	

	# get a photometry df so we can plot source positions
	# TODO generalize where this is read in from 
	try:
		phot_df = pq.read_table(f'/data/tierras/photometry/{best_date}/{field}/{ffname}/{best_date}_{field}_circular_fixed_ap_phot_5.parquet').to_pandas()
	except:
		phot_df = pq.read_table(f'/data/tierras/photometry/{best_date}/{field}/{ffname}/{best_date}_{field}_circular_fixed_ap_phot_5.0.parquet').to_pandas()

	flattened_files = get_flattened_files(best_date, field, ffname)
	source_image = fits.open(flattened_files[best_im_ind])[0].data

	# get the global light curves that have been created for this field
	lc_files = glob.glob(f'/data/tierras/fields/{field}/sources/lightcurves/{ffname}/**global_lc.csv')
	times = np.array(pd.read_csv(lc_files[0], comment='#')['BJD TDB'])
	x_offset = times[0]
	times -= x_offset

	weighted_ref_rp = []
	weighted_ref_n2n = []
	weighted_ref_bp_rp = []
	weighted_ref_G = []

	# get a list of indices of that correspond to different nights in the light curves
	time_deltas = np.array([times[i]-times[i-1] for i in range(1,len(times))])
	time_breaks = np.where(time_deltas > 0.4)[0]
	times_list = []
	times_inds = []
	for i in range(len(time_breaks)):
		if i == 0:
			times_list.append(times[0:time_breaks[i]+1])
			times_inds.append(np.arange(0,time_breaks[i]+1))
		else:
			times_list.append(times[time_breaks[i-1]+1:time_breaks[i]+1])
			times_inds.append(np.arange(time_breaks[i-1]+1,time_breaks[i]+1))
	if len(time_breaks) > 0:
		times_list.append(times[time_breaks[-1]+1:len(times)])
		times_inds.append(np.arange(time_breaks[-1]+1,len(times)))
	else:
		times_list.append(times)
		times_inds.append(np.arange(len(times)))

	night_to_nights = [] 
	night_to_nights_theory_calculated = []
	night_to_nights_theory_measured =[]
	rp_mags = []
	source_ids = []
	ap_rad = [] 
	times_arr = []
	flux_arr = []
	flux_err_arr = []
	bx_arr = []
	by_arr = []
	bye_arr = []
	x_pos = []
	y_pos = []
	rp = []
	bp_rp = []
	G = []
	global_sky_array = []
	global_x_array = []
	global_y_array = []
	global_flux_flags = []
	global_wcs_flags = []
	global_position_flags = []
	global_fwhm_flags = []
	
	if cut_contaminated:	
		images = get_flattened_files(date_list[0].split('/')[4], field, 'flat0000')
		# we don't perform photometry on *every* source in the field, so to get an accurate estimate of the contamination for each source, we need to query gaia for all sources (i.e. with no Rp mag limit) so that their fluxes can be modeled
		all_field_sources = gaia_query(images)
		xx, yy = np.meshgrid(np.arange(-int(contaminant_grid_size/2), int(contaminant_grid_size)/2), np.arange(-int(contaminant_grid_size/2), int(contaminant_grid_size/2))) # grid of pixels over which to simulate images for contamination estimate
		seeing_fwhm = np.nanmedian(fwhm_x) / PLATE_SCALE # get median seeing on this night in pixels for contamination estimate
		seeing_sigma = seeing_fwhm / (2*np.sqrt(2*np.log(2))) # convert from FWHM in pixels to sigma in pixels (for 2D Gaussian modeling in contamination estimate)
		contaminations = []

	# plt.figure(figsize=(10,10))
	for i in range(len(lc_files)):
		try:
			source = int(lc_files[i].split('Gaia DR3 ')[1].split('_')[0])
			print(f'Doing Gaia DR3 {source} ({i+1} of {len(lc_files)})')
			doing_target = False
		except:
			source = lc_files[i].split('/')[-1].split('_')[0]
			# get the source position using the cat-x/cat-y keys in the header
			hdr = fits.open(glob.glob(f'/data/tierras/flattened/{date_list[-1].split("/")[4]}/{field}/flat0000/*.fit')[0])[0].header
			targ_x_pix = hdr['CAT-X']
			targ_y_pix = hdr['CAT-Y']
			source = identify_target_gaia_id(source, source_df, x_pix=targ_x_pix, y_pix=targ_y_pix)
			target_gaia_id = source
			doing_target = True 

		with open(lc_files[i],'r') as f: 
			comment = f.readline()
		ap_rad.append(int(comment.split('_')[-1].split('.')[0]))

		source_ind = np.where(source_df['source_id'] == source)[0][0] 
		source_x = source_df['X pix'][source_ind]
		source_y = source_df['Y pix'][source_ind]
		source_rp = source_df['phot_rp_mean_mag'][source_ind]
		source_G = source_df['gq_photogeo'][source_ind]
		source_bp_rp = source_df['bp_rp'][source_ind]

		# sometimes targets aren't in the Bailer-Jones table and don't have a G mag. Calculate using parallax if this is the case.
		if doing_target and np.isnan(source_G): 
			source_parallax = source_df['parallax'][source_ind]
			source_distance = 1/(source_parallax/1000)
			source_G = source_df['phot_g_mean_mag'][source_ind] - 5*np.log10(source_distance) + 5

		contamination = 0
		if cut_contaminated and not doing_target:
			# estimate contamination 
			distances = np.array(np.sqrt((source_x-all_field_sources['X pix'])**2+(source_y-all_field_sources['Y pix'])**2))
			nearby_inds = np.where((distances <= contaminant_grid_size) & (distances != 0))[0]
			if len(nearby_inds) > 0:
				

				nearby_rp = np.array(all_field_sources['phot_rp_mean_mag'][nearby_inds])
				nearby_x = np.array(all_field_sources['X pix'][nearby_inds] - source_x)
				nearby_y = np.array(all_field_sources['Y pix'][nearby_inds] - source_y)

				# sometimes the rp mag is nan, remove these entries
				use_inds = np.where(~np.isnan(nearby_rp))[0]
				nearby_rp = nearby_rp[use_inds]
				nearby_x = nearby_x[use_inds]
				nearby_y = nearby_y[use_inds]

				# enforce that a nearby source cannot have the same rp magnitude as the source in question, that's almost certainly a duplicate
				use_inds = np.where(nearby_rp != source_rp)
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
				ap = CircularAperture((sim_img.shape[1]/2, sim_img.shape[0]/2), r=ap_rad[i])
				phot_table = aperture_photometry(sim_img, ap)
				contamination = phot_table['aperture_sum'][0] - 1

				# if source_id == 4147121464971195776:
				# 	breakpoint()

				# if the contamination is over the limit, exclude this source from the analysis
				if contamination > contamination_limit:
					# plt.figure()
					# plt.imshow(sim_img, origin='lower', norm=simple_norm(sim_img, min_percent=1, max_percent=98))
					# breakpoint()
					continue

		
		df = pd.read_csv(lc_files[i], comment='#')
		flux = np.array(df['Flux'])
		calculated_flux_err = np.array(df['Flux Error']) # calculated using photon noise from star/sky, read noise, dark current, ALC noise, and scintillation 
		measured_noise = np.zeros(len(times_inds)) # will hold the MEASURED noise from the standard deviation of each night.
		night_medians = np.zeros(len(times_inds))
		night_errs_on_meds = np.zeros(len(times_inds))

		# read in flags as boolean arrays
		global_flux_flags.append(np.array(df['Flux Flag']).astype(bool))
		global_wcs_flags.append(np.array(df['WCS Flag']).astype(bool))
		global_position_flags.append(np.array(df['Position Flag']).astype(bool))
		global_fwhm_flags.append(np.array(df['FWHM Flag']).astype(bool))

		# make mask for all flags; true = exposure is good, false = exposure is flagged
		all_mask = ~global_wcs_flags[i] & ~global_position_flags[i] & ~global_fwhm_flags[i] & ~global_flux_flags[i]

		# calculate the medians of each night 
		sc_flux_arr = []
		sc_times_arr = []
		sc_flux_err_arr = []
		sc_bx_arr = []
		sc_by_arr = []
		sc_bye_arr = []
		for j in range(len(night_medians)):
			
			night_times = np.array(times_list[j])
			inds = np.where((times >= night_times[0]) & (times <= night_times[-1]))[0]
			night_mask = all_mask[inds]

			# inds = np.array([i for i in inds if i in all_mask])
			night_flux = flux[inds]

			calculated_night_err = calculated_flux_err[inds][night_mask]
			# nan_inds = ~np.isnan(night_flux)

			# v, l, h = sigmaclip(night_flux[nan_inds], 4, 4)
			# sc_inds = np.where((night_flux >= l) & (night_flux <= h))[0]
			# bad_sc_inds = np.where((night_flux < l) | (night_flux > h))[0]

			# night_times[bad_sc_inds] = np.nan 
			# night_flux[bad_sc_inds] = np.nan
			# calculated_night_err[bad_sc_inds] = np.nan

			n_exp = sum(~np.isnan(night_flux[night_mask]))

			measured_noise[j] = 1.2533*np.nanstd(night_flux[night_mask])/(n_exp**(1/2))
			night_medians[j] = np.nanmedian(night_flux[night_mask])
			night_errs_on_meds[j] = 1.2533*np.nanmedian(calculated_night_err)/(n_exp**(1/2))

			sc_times_arr.extend(night_times)
			sc_flux_arr.extend(night_flux)
			sc_flux_err_arr.extend(calculated_flux_err[inds])
			sc_bx_arr.extend([np.nanmean(night_times)])
			sc_by_arr.extend([np.nanmedian(night_flux)])
			sc_bye_arr.extend([measured_noise[j]])

		
		times_arr.append(np.array(sc_times_arr))
		flux_arr.append(np.array(sc_flux_arr))
		flux_err_arr.append(np.array(sc_flux_err_arr))
		bx_arr.append(np.array(sc_bx_arr))
		by_arr.append(np.array(sc_by_arr))
		bye_arr.append(np.array(sc_bye_arr))

		global_sky_array.append(np.array(df['Sky Background (ADU/s)']))
		global_x_array.append(np.array(df['X']))
		global_y_array.append(np.array(df['Y']))
		

		n2n = np.nanstd(night_medians)
		night_to_nights.append(n2n)
		night_to_nights_theory_calculated.append(np.nanmedian(night_errs_on_meds))
		night_to_nights_theory_measured.append(np.nanmedian(measured_noise))

		source_ids.append(source)
		source_ind = np.where(source_df['source_id'] == source)[0][0]

		x_pos.append(phot_df[f'S{source_ind} X'][best_im_ind])
		y_pos.append(phot_df[f'S{source_ind} Y'][best_im_ind])
		rp.append(source_df['phot_rp_mean_mag'][source_ind])
		bp_rp.append(source_df['bp_rp'][source_ind])
		G.append(source_G)
		rp_mags.append(source_df['phot_rp_mean_mag'][source_ind])
		if source in ref_ids:
			ref_ind = np.where(source == ref_ids)[0][0]
			comment = []
			# with open(lc_files[i], 'r') as file:
			# 	comment.append(file.readline())
			ap_size = ap_rad[i]

			# check if this star was given a weight in *any* of the photometry files.
			# if so, count it as a ref star in the plot
			if np.nansum(weights_df[ref_ind][1:]) > 0:
				weighted_ref_rp.append(source_rp)
				weighted_ref_n2n.append(n2n)
				weighted_ref_bp_rp.append(source_bp_rp)
				weighted_ref_G.append(source_G)

	global_sky_array = np.array(global_sky_array)
	global_x_array = np.array(global_x_array)
	global_y_array = np.array(global_y_array)
	global_flux_flags = np.array(global_flux_flags)
	global_wcs_flags = np.array(global_wcs_flags)
	global_position_flags = np.array(global_position_flags)

	source_ids = np.array(source_ids, dtype='int')
	ap_rad = np.array(ap_rad)
	night_to_nights = np.array(night_to_nights)
	night_to_nights_theory_calculated = np.array(night_to_nights_theory_calculated)
	night_to_nights_theory_measured = np.array(night_to_nights_theory_measured)
	rp_mags = np.array(rp_mags)
	print(f'{len(rp_mags)} sources after contamination cuts.')

	# fig, ax = plt.subplots(1,3,figsize=(20,8), gridspec_kw={'width_ratios':[1,2,1]})
	fig = plt.figure(figsize=(19,12))
	gs = GridSpec(6, 4, width_ratios=[2,1,1,1], height_ratios=[1.5,1.5,0.5,0.5,0.5,0.5])

	rax = plt.axes([0.85, 0.85, 0.1, 0.1])  # Position for the checkbox
	check_all = CheckButtons(rax, ['All quality flags', 'Sigma clipping'], [True, False])
	check_all.on_clicked(update_all_flag_visibility)

	ax1 = fig.add_subplot(gs[0,0])
	ax2 = fig.add_subplot(gs[1,:])
	ax3 = fig.add_subplot(gs[0,1])
	ax4 = fig.add_subplot(gs[0,2])
	ax5 = fig.add_subplot(gs[2,:], sharex=ax2)
	ax6 = fig.add_subplot(gs[3,:], sharex=ax2)
	ax7 = fig.add_subplot(gs[4,:], sharex=ax2)
	ax8 = fig.add_subplot(gs[5,:], sharex=ax2)

	global currently_selected_source
	currently_selected_source = None

	axes_mapping = {ax1: 'ax1', ax2: 'ax2', ax3: 'ax3', ax4: 'ax4', ax5:'ax5', ax6:'ax6', ax7:'ax7', ax8:'ax8'}
	
	# ig.canvas.mpl_connect('motion_notify_event', on_plot_hover)
	global highlight
	highlight = None
	fig.canvas.mpl_connect('button_press_event', on_click)
  
	for i in range(len(rp_mags)):
	# for i in range(100):
		if i == 0:
			ax1.plot(rp_mags[i], night_to_nights[i], alpha=0.3, label='Measured', gid=source_ids[i], color='tab:blue', marker='.', ls='')
			ax1.plot(rp_mags[i], night_to_nights_theory_calculated[i], alpha=0.3, color='k', label='Theory (unc. / sqrt(N))', marker='.', ls='', gid=source_ids[i])
			# ax1.plot(rp_mags[i], night_to_nights_theory_measured[i], alpha=0.3, color='r', label='Theory (measured $\sigma$ / sqrt(N))', marker='.', ls='', gid=source_ids[i])
		else:
			ax1.plot(rp_mags[i], night_to_nights[i], alpha=0.3, gid=source_ids[i], color='tab:blue', marker='.', ls='')
			ax1.plot(rp_mags[i], night_to_nights_theory_calculated[i], alpha=0.3, color='k', marker='.', ls='', gid=source_ids[i])	
			# ax1.plot(rp_mags[i], night_to_nights_theory_measured[i], alpha=0.3, color='r', marker='.', ls='', gid=source_ids[i])
	
	# gaia_var_inds = np.where(source_df['phot_variable_flag'] == 'VARIABLE')[0]
	# ax1.plot(rp_mags[gaia_var_inds], night_to_nights[gaia_var_inds], marker='x', zorder=1, label='Gaia variable', color='m', ls='', mew=1.5, ms=6, alpha=1)
	ax1.plot(weighted_ref_rp,weighted_ref_n2n, marker='x', zorder=1, label='Ref star', color='m', ls='', mew=1.5, ms=6, alpha=1)

	# add an indicator for the target 
	try:
		targ_ind = np.where(source_ids == target_gaia_id)[0][0]
		ax1.plot(rp_mags[targ_ind], night_to_nights[targ_ind], gid=target_gaia_id, color='#EFBF04', marker='*', mew=1.5, mec='k', ms=18)
	except:
		pass

	ax1.set_xlim(np.nanmin(rp_mags)-0.1, np.nanmax(rp_mags)+0.1)
	ax1.set_yscale('log')
	# ax.invert_xaxis()
	ax1.set_xlabel('Rp', fontsize=14)
	ax1.set_ylabel('$\sigma_{N2N}$', fontsize=14)
	ax1.grid(True, which='both', alpha=0.3)
	ax1.legend(fontsize=10, loc='upper left')
	ax1.tick_params(labelsize=12)
	ax2.set_ylabel('Normalized Flux', fontsize=14)
	ax2.tick_params(labelsize=12)
	ax2.grid()

	ax3.imshow(source_image, origin='lower', interpolation='none', norm=simple_norm(source_image, min_percent=1, max_percent=99))
	for i in range(len(x_pos)):
		ax3.plot(x_pos[i], y_pos[i], 'rx', gid=source_ids[i])
	
		ax4.plot(bp_rp[i], G[i], marker='.', color='#b0b0b0', ls='', gid=source_ids[i], zorder=0)
	ax4.plot(weighted_ref_bp_rp,weighted_ref_G, marker='x', zorder=1, label='Ref star', color='m', ls='', mew=1.5, ms=6, alpha=1)
	try:
		ax4.plot(bp_rp[targ_ind], G[targ_ind], gid=target_gaia_id, color='#EFBF04', marker='*', mew=1.5, mec='k', ms=18)
	except:
		pass

	ax4.invert_yaxis()
	ax4.set_aspect('equal')
	ax4.set_xlabel('Bp-Rp', fontsize=14)
	ax4.set_ylabel('G', fontsize=14)

	# plot airmass 
	ax5.plot(global_ancillary_data['BJD TDB']-x_offset, global_ancillary_data['Airmass'], marker='.', ls='')
	ax5.set_ylabel('Airmass')

	# plot sky background 
	ax6.set_ylabel('Sky (ADU/s)')

	# plot x/y positions
	ax7.set_ylabel('Pos. (pix)')

	# plot fwhm 
	ax8.plot(global_ancillary_data['BJD TDB']-x_offset, global_ancillary_data['FWHM X'], marker='.', ls='', label='Major axis', color='tab:pink')
	ax8.plot(global_ancillary_data['BJD TDB']-x_offset, global_ancillary_data['FWHM Y'], marker='.', ls='', label='Minor axis', color='tab:purple')
	ax8.legend()
	ax8.set_ylabel('FWHM (")')

	fig.axes[-1].set_xlabel('BJD TDB', fontsize=14)
	plt.suptitle(f'{field} field', fontsize=14)
	plt.tight_layout()

	print(f'Median observed/theory (calculated uncertainties): {np.nanmedian(night_to_nights/night_to_nights_theory_calculated):.1f}')
	print(f'Median observed/theory (measured scatter): {np.nanmedian(night_to_nights/night_to_nights_theory_measured):.1f}')


	x = np.array(x_pos)
	y = np.array(y_pos)
	bp_rp = np.array(bp_rp)
	G = np.array(G)

	ratios = night_to_nights / night_to_nights_theory_measured
	perc = np.percentile(ratios, 10) # get the 10% best ratios for plotting purposes
	best_ratios = np.where(ratios< perc)[0]

	# plot chip positions colored by ratios

	fig2 = plt.figure(figsize=(12,10))
	gs2 = GridSpec(2, 1, height_ratios=[1.5,1])
	ax1_2 = fig2.add_subplot(gs2[0])
	ax2_2 = fig2.add_subplot(gs2[1])
	axes_mapping_2 = {ax1_2: 'ax1_2', 'ax2_2': ax2_2}
	global highlight_2
	highlight_2 = None
	fig2.canvas.mpl_connect('button_press_event', on_click_2)

	sc = ax1_2.scatter(x,y, c=np.log10(ratios), vmin=0, vmax=1, gid=source_ids)
	try:
		ax1_2.plot(x[targ_ind], y[targ_ind], gid=target_gaia_id, color='#EFBF04', marker='*', mew=1.5, mec='k', ms=18)
	except:
		pass

	# cbar = plt.colorbar(sc, label='log$_{10}$(measured/theory)')
	ax1_2.scatter(x[best_ratios], y[best_ratios], c=np.log10(ratios[best_ratios]), vmin=0, vmax=1, edgecolors='m', linewidth=1.5) # plot the best 10% on top
	# cbar.ax.plot([0,1], [np.log10(perc),np.log10(perc)], lw=3, color='m')
	ax2_2.set_xlabel('Time (BJD$_{TDB}$)', fontsize=14)
	ax2_2.set_ylabel('Normalized Flux', fontsize=14)
	ax2_2.set_title('Init.', fontsize=14)
	plt.tight_layout()

	# plot CMD colored by ratios 

	fig3 = plt.figure(figsize=(10,12))
	gs3 = GridSpec(2, 1, height_ratios=[1.5,1])
	ax1_3 = fig3.add_subplot(gs3[0])
	ax2_3 = fig3.add_subplot(gs3[1])

	axes_mapping_3 = {ax1_3: 'ax1_3', ax2_3: 'ax2_3'}

	global highlight_3
	highlight_3 = None
	fig3.canvas.mpl_connect('button_press_event', on_click_3)

	sc = ax1_3.scatter(bp_rp, G, c=np.log10(ratios), vmin=0, vmax=1)
	try:
		ax1_3.plot(bp_rp[targ_ind], G[targ_ind], gid=target_gaia_id, color='#EFBF04', marker='*', mew=1.5, mec='k', ms=18)
	except:
		pass
	divider = make_axes_locatable(ax1_3)
	cax = divider.append_axes('right', size='5%', pad=0.05)
	cbar = fig3.colorbar(sc, cax=cax, label='log$_{10}$(measured/theory)')

	ax1_3.invert_yaxis()
	ax1_3.set_xlabel('Bp-Rp', fontsize=14)
	ax1_3.set_ylabel('G', fontsize=14)
	ax1_3.tick_params(labelsize=12)
	ax1_3.scatter(bp_rp[best_ratios], G[best_ratios], c=np.log10(ratios[best_ratios]), vmin=0, vmax=1, edgecolors='m', linewidth=1.5) # plot the best 10% on top
	cbar.ax.plot([0,1], [np.log10(perc),np.log10(perc)], lw=3, color='m')

	ax2_3.set_xlabel('Time (BJD$_{TDB}$)', fontsize=14)
	ax2_3.set_ylabel('Normalized Flux', fontsize=14)
	ax2_3.set_title('Init.', fontsize=14)
	plt.tight_layout()

	breakpoint()
	


if __name__ == '__main__':

	main()