import argparse
import numpy as np 
import pandas as pd 
import glob 
import matplotlib.pyplot as plt 
plt.ion()
from scipy.stats import sigmaclip
import pyarrow.parquet as pq 
from astropy.io import fits 
from ap_phot import get_flattened_files
from astropy.visualization import simple_norm 
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import TextBox, Button
from analyze_global import identify_target_gaia_id
from matplotlib import colors

def main(raw_args=None):

	# def on_plot_hover(event):
	# 	# Iterating over each data member plotted
	# 	for curve in ax1.get_lines():
	# 		# Searching which data member corresponds to current mouse position
	# 		if curve.contains(event)[0]:
	# 			if curve.get_gid() != None:
	# 				ax2.cla()
	# 				ax1.patches.clear()
	# 				ax2.patches.clear()
	# 				ax3.patches.clear()
	# 				ax4.patches.clear()
	# 				print("over %d" % curve.get_gid())

	# 				# df = pd.read_csv(f'/data/tierras/fields/{field}/sources/lightcurves/Gaia DR3 {curve.get_gid()}_global_lc.csv', comment='#')
	# 				# times = np.array(df['BJD TDB'])
	# 				# flux = np.array(df['Flux'])
					
	# 				# nan_inds = ~np.isnan(flux)
	# 				# times = times[nan_inds]
	# 				# flux = flux[nan_inds]

	# 				# v, l, h = sigmaclip(flux)
	# 				# inds = np.where((flux > l) & (flux < h))
	# 				source_ind = np.where(source_ids == curve.get_gid())[0][0]
					
	# 				ax2.set_title(f'Gaia DR3 {source_ids[source_ind]}', fontsize=14)
	# 				ax2.errorbar(times_arr[source_ind], flux_arr[source_ind], flux_err_arr[source_ind], marker='.', color='#b0b0b0', ecolor='#b0b0b0', ls='')
	# 				ax2.errorbar(bx_arr[source_ind], by_arr[source_ind], bye_arr[source_ind], marker='o', color='k', ls='', zorder=3)
	# 				ax2.grid()
	# 				ax2.axhline(1, ls='--', lw=2, color='k', alpha=0.7, zorder=0)

	# 				ax3.set_xlim(x_pos[source_ind]-40, x_pos[source_ind]+40)
	# 				ax3.set_ylim(y_pos[source_ind]-40, y_pos[source_ind]+40)

	# 				circ1 = plt.Circle((bp_rp[source_ind], G[source_ind]), 0.2, color='k', fill=False, linewidth=2, zorder=3)
	# 				ax4.add_patch(circ1)
		
	# 	# Iterating over each data member plotted
	# 	for curve in ax3.get_lines():
	# 		# Searching which data member corresponds to current mouse position
	# 		if curve.contains(event)[0]:
	# 			if curve.get_gid() != None:
	# 				ax2.cla()
	# 				ax1.patches.clear()
	# 				ax2.patches.clear()
	# 				ax3.patches.clear()
	# 				ax4.patches.clear()
	# 				print("over %d" % curve.get_gid())

	# 				# df = pd.read_csv(f'/data/tierras/fields/{field}/sources/lightcurves/Gaia DR3 {curve.get_gid()}_global_lc.csv', comment='#')
	# 				# times = np.array(df['BJD TDB'])
	# 				# flux = np.array(df['Flux'])
					
	# 				# nan_inds = ~np.isnan(flux)
	# 				# times = times[nan_inds]
	# 				# flux = flux[nan_inds]

	# 				# v, l, h = sigmaclip(flux)
	# 				# inds = np.where((flux > l) & (flux < h))
	# 				source_ind = np.where(source_ids == curve.get_gid())[0][0]
					
	# 				ax2.set_title(f'Gaia DR3 {source_ids[source_ind]}', fontsize=14)
	# 				ax2.errorbar(times_arr[source_ind], flux_arr[source_ind], flux_err_arr[source_ind], marker='.', color='#b0b0b0', ecolor='#b0b0b0', ls='')
	# 				ax2.errorbar(bx_arr[source_ind], by_arr[source_ind], bye_arr[source_ind], marker='o', color='k', ls='', zorder=3)
	# 				ax2.grid()
	# 				ax2.axhline(1, ls='--', lw=2, color='k', alpha=0.7, zorder=0)

	# 				ax3.set_xlim(x_pos[source_ind]-40, x_pos[source_ind]+40)
	# 				ax3.set_ylim(y_pos[source_ind]-40, y_pos[source_ind]+40)	

	# 				circ1 = plt.Circle((bp_rp[source_ind], G[source_ind]), 0.2, color='k', fill=False, linewidth=2, zorder=3)
	# 				ax4.add_patch(circ1)

	# 	# Iterating over each data member plotted
	# 	for curve in ax4.get_lines():
	# 		# Searching which data member corresponds to current mouse position
	# 		if curve.contains(event)[0]:
	# 			if curve.get_gid() != None:
	# 				ax2.cla()
	# 				ax1.patches.clear()
	# 				ax2.patches.clear()
	# 				ax3.patches.clear()
	# 				ax4.patches.clear()
	# 				print("over %d" % curve.get_gid())

	# 				# df = pd.read_csv(f'/data/tierras/fields/{field}/sources/lightcurves/Gaia DR3 {curve.get_gid()}_global_lc.csv', comment='#')
	# 				# times = np.array(df['BJD TDB'])
	# 				# flux = np.array(df['Flux'])
					
	# 				# nan_inds = ~np.isnan(flux)
	# 				# times = times[nan_inds]
	# 				# flux = flux[nan_inds]

	# 				# v, l, h = sigmaclip(flux)
	# 				# inds = np.where((flux > l) & (flux < h))
	# 				source_ind = np.where(source_ids == curve.get_gid())[0][0]
					
	# 				ax2.set_title(f'Gaia DR3 {source_ids[source_ind]}', fontsize=14)
	# 				ax2.errorbar(times_arr[source_ind], flux_arr[source_ind], flux_err_arr[source_ind], marker='.', color='#b0b0b0', ecolor='#b0b0b0', ls='')
	# 				ax2.errorbar(bx_arr[source_ind], by_arr[source_ind], bye_arr[source_ind], marker='o', color='k', ls='', zorder=3)
	# 				ax2.grid()
	# 				ax2.axhline(1, ls='--', lw=2, color='k', alpha=0.7, zorder=0)

	# 				ax3.set_xlim(x_pos[source_ind]-40, x_pos[source_ind]+40)
	# 				ax3.set_ylim(y_pos[source_ind]-40, y_pos[source_ind]+40)	

	# 				circ1 = plt.Circle((bp_rp[source_ind], G[source_ind]), 0.2, color='k', fill=False, linewidth=2, zorder=3)
	# 				ax4.add_patch(circ1)

	def on_click(event):
		global highlight 
		# print(event.inaxes)
		ax = event.inaxes
		if ax is not None:
			label = axes_mapping.get(ax, 'Unknown axis')
			if highlight:
				ax1.lines[-1].remove()
				ax3.lines[-1].remove()
				ax4.lines[-1].remove()
				highlight = None
		else:
			label = ''

		if label == 'ax1':
			x_click = event.xdata
			y_click = event.ydata
			dists = ((rp_mags - x_click)**2+(np.log10(night_to_nights) - np.log10(y_click))**2)**0.5
			point = np.argmin(dists)
			# print(point)
			# print(x_click)
			# print(y_click)
		elif label == 'ax3':
			x_click = event.xdata
			y_click = event.ydata
			dists = ((x_pos - x_click)**2+(y_pos- y_click)**2)**0.5
			point = np.argmin(dists)
		elif label == 'ax4':
			x_click = event.xdata
			y_click = event.ydata
			dists = ((bp_rp - x_click)**2+(G - y_click)**2)**0.5
			point = np.nanargmin(dists)
		else:
			return
		
		print(f'Clicked on {source_ids[point]}')

		highlight = ax1.plot(rp_mags[point], night_to_nights[point], 'mo')

		# highlight_3 = ax3.plot(rp_mags[point], binned_sigmas[point]*1e6, 'mo')

		highlight_4 = ax4.plot(bp_rp[point], G[point], 'mo')

		# clear out the previous lc and plot the new one
		ax2.cla()
		# ax2.errorbar(times_save[point], flux_save[point], flux_err_save[point], marker='.', ls='', ecolor='#b0b0b0', color='#b0b0b0')
		# ax2.errorbar(bin_times_save[point], bin_flux_save[point], bin_flux_err_save[point], marker='o', color='k', ecolor='k', mfc='none', zorder=3, mew=1.5, ms=7, ls='')
		ax2.errorbar(times_arr[point], flux_arr[point], flux_err_arr[point], marker='.', color='#b0b0b0', ecolor='#b0b0b0', ls='')
		ax2.errorbar(bx_arr[point], by_arr[point], bye_arr[point], marker='o', color='k', ls='', zorder=3)
		ax2.axhline(1, ls='--', lw=2, color='k', alpha=0.7, zorder=0)
		ax2.grid(alpha=0.5)
		ax2.tick_params(labelsize=12)
		ax2.set_xlabel('Time (BJD$_{TDB}$)', fontsize=14)
		ax2.set_ylabel('Normalized Flux', fontsize=14)
		ax2.set_title(source_ids[point], fontsize=14)

		# # update the field plot 
		ax3.set_xlim(x_pos[point]-50, x_pos[point]+50)
		ax3.set_ylim(y_pos[point]-50, y_pos[point]+50)


	ap = argparse.ArgumentParser()
	ap.add_argument("-field", required=True, help="Name of observed target field exactly as shown in raw FITS files.")
	ap.add_argument("-ffname", required=False, default='flat0000', help="Name of flat directory")
	args = ap.parse_args()
	field = args.field 
	ffname = args.ffname

	# get dates on which field was observed so we can plot an image of the field
	date_list = glob.glob(f'/data/tierras/photometry/**/{field}/{ffname}')	
	date_list = np.array(sorted(date_list, key=lambda x:int(x.split('/')[4]))) # sort on date so that everything is in order
	if field == 'TIC362144730':
		date_list = np.delete(date_list, [-1])

	# read in weights csv
	weights_df = pd.read_csv(f'/data/tierras/fields/{field}/sources/weights.csv')
	ref_ids = np.array(weights_df['Ref ID'])

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
	phot_df = pq.read_table(f'/data/tierras/photometry/{best_date}/{field}/{ffname}/{best_date}_{field}_circular_fixed_ap_phot_5.parquet').to_pandas()

	flattened_files = get_flattened_files(best_date, field, ffname)
	source_image = fits.open(flattened_files[best_im_ind])[0].data

	# get the global light curves that have been created for this field
	lc_files = glob.glob(f'/data/tierras/fields/{field}/sources/lightcurves/**.csv')
	times = np.array(pd.read_csv(lc_files[0], comment='#')['BJD TDB'])
	times -= times[0] 

	# read in positions of sources in the reference image
	x_pos = np.zeros(len(lc_files))
	y_pos = np.zeros_like(x_pos)
	rp = np.zeros_like(x_pos)
	bp_rp = np.zeros_like(x_pos)
	G = np.zeros_like(x_pos)
	for i in range(len(lc_files)):
		try:
			source = int(lc_files[i].split('Gaia DR3 ')[1].split('_')[0])
		except:
			source = identify_target_gaia_id(str(source), source_df, x_pix=2048, y_pix=512)
		ind = np.where(source_df['source_id'] == source)[0][0]
		x_pos[i] = phot_df[f'S{ind} X'][best_im_ind]
		y_pos[i] = phot_df[f'S{ind} Y'][best_im_ind]	
		rp[i] = source_df['phot_rp_mean_mag'][ind]
		bp_rp[i] = source_df['bp_rp'][ind]
		G[i] = source_df['gq_photogeo'][ind]
		# except:
		# 	x_pos[i] = np.nan
		# 	y_pos[i] = np.nan

	weighted_ref_rp = []
	weighted_ref_n2n = []
	weighted_ref_bp_rp = []

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
	times_list.append(times[time_breaks[-1]+1:len(times)])
	times_inds.append(np.arange(time_breaks[-1]+1,len(times)))

	night_to_nights = np.zeros(len(lc_files))
	night_to_nights_theory_calculated = np.zeros(len(lc_files))
	night_to_nights_theory_measured = np.zeros(len(lc_files))
	rp_mags = np.zeros(len(lc_files))
	source_ids = []

	times_arr = []
	flux_arr = []
	flux_err_arr = []
	bx_arr = []
	by_arr = []
	bye_arr = []
	# plt.figure(figsize=(10,10))
	for i in range(len(lc_files)):
		try:
			source = int(lc_files[i].split('Gaia DR3 ')[1].split('_')[0])
			source_ids.append(source)	
			print(f'Doing Gaia DR3 {source} ({i+1} of {len(lc_files)})')
		except:
			source = identify_target_gaia_id(str(source), source_df, x_pix=2048, y_pix=512)
			source_ids.append(source)
			# print('Could not identify Gaia DR3 ID, skipping')
			# breakpoint()
			# source_ids.append(0)
			# times_arr.append(np.nan)
			# flux_arr.append(np.nan)
			# flux_err_arr.append(np.nan)
			# bx_arr.append(np.nan)
			# by_arr.append(np.nan)
			# bye_arr.append(np.nan)
			# night_to_nights[i] = np.nan 
			# night_to_nights_theory_calculated[i] = np.nan
			# night_to_nights_theory_measured[i] = np.nan
			# rp_mags[i] = np.nan 
			# continue



		df = pd.read_csv(lc_files[i], comment='#')
		flux = np.array(df['Flux'])
		calculated_flux_err = np.array(df['Flux Error']) # calculated using photon noise from star/sky, read noise, dark current, ALC noise, and scintillation 
		measured_noise = np.zeros(len(times_inds)) # will hold the MEASURED noise from the standard deviation of each night.
		night_medians = np.zeros(len(times_inds))
		night_errs_on_meds = np.zeros(len(times_inds))

		sc_flux_arr = []
		sc_times_arr = []
		sc_flux_err_arr = []
		sc_bx_arr = []
		sc_by_arr = []
		sc_bye_arr = []
		for j in range(len(night_medians)):
				
			# TODO: outlier rejection 
			# TODO: eliminate nans from sqrt(len(times_inds[j]))
			night_times = np.array(times_list[j])
			try:
				night_flux = flux[times_inds[j]]
			except:
				breakpoint()
			
			calculated_night_err = calculated_flux_err[times_inds[j]]
			nan_inds = ~np.isnan(night_flux)

			night_times = night_times[nan_inds]
			night_flux = night_flux[nan_inds]
			calculated_night_err = calculated_night_err[nan_inds]

			v, l, h = sigmaclip(night_flux, 4, 4)
			sc_inds = np.where((night_flux >= l) & (night_flux <= h))[0]
			sc_times = night_times[sc_inds]
			sc_flux = night_flux[sc_inds]
			sc_err = calculated_night_err[sc_inds]

			n_exp = len(sc_flux)

			measured_noise[j] = 1.2533*np.nanstd(sc_flux)/(n_exp**(1/2))
			night_medians[j] = np.nanmedian(sc_flux)
			night_errs_on_meds[j] = 1.2533*np.nanmedian(sc_err)/(n_exp**(1/2))

			sc_times_arr.extend(sc_times)
			sc_flux_arr.extend(sc_flux)
			sc_flux_err_arr.extend(sc_err)
			sc_bx_arr.extend([np.nanmean(sc_times)])
			sc_by_arr.extend([np.nanmedian(sc_flux)])
			sc_bye_arr.extend([measured_noise[j]])

		
		times_arr.append(np.array(sc_times_arr))
		flux_arr.append(np.array(sc_flux_arr))
		flux_err_arr.append(np.array(sc_flux_err_arr))
		bx_arr.append(np.array(sc_bx_arr))
		by_arr.append(np.array(sc_by_arr))
		bye_arr.append(np.array(sc_bye_arr))

		night_to_nights[i] = np.nanstd(night_medians)
		night_to_nights_theory_calculated[i] = np.nanmedian(night_errs_on_meds)
		night_to_nights_theory_measured[i] = np.nanmedian(measured_noise)
		
		source_ind = np.where(source_df['source_id'] == source)[0][0]
		rp_mags[i] = source_df['phot_rp_mean_mag'][source_ind]
		if source in ref_ids:
			ref_ind = np.where(source == ref_ids)[0][0]
			comment = []
			with open(lc_files[i], 'r') as file:
				comment.append(file.readline())
			ap_size = comment[0].split('using ')[1].split('_')[-1].split('.')[0]

			weight = weights_df[ap_size][ref_ind]
			if weight != 0:
				weighted_ref_rp.append(rp_mags[i])
				weighted_ref_n2n.append(night_to_nights[i])
				weighted_ref_bp_rp.append(bp_rp[i])
		
	source_ids = np.array(source_ids, dtype='int')

	# fig, ax = plt.subplots(1,3,figsize=(20,8), gridspec_kw={'width_ratios':[1,2,1]})
	fig = plt.figure(figsize=(15,10))
	gs = GridSpec(2, 3, width_ratios=[3,1,0.5], height_ratios=[1.5,1])
	ax1 = fig.add_subplot(gs[0,0])
	ax2 = fig.add_subplot(gs[1,:])
	ax3 = fig.add_subplot(gs[0,1])
	ax4 = fig.add_subplot(gs[0,2])

	axes_mapping = {ax1: 'ax1', ax2: 'ax2', ax3: 'ax3', ax4: 'ax4'}
	
	# ig.canvas.mpl_connect('motion_notify_event', on_plot_hover)
	global highlight
	highlight = None
	fig.canvas.mpl_connect('button_press_event', on_click)
  
	for i in range(len(lc_files)):
	# for i in range(100):
		if i == 0:
			ax1.plot(rp_mags[i], night_to_nights[i], alpha=0.3, label='Measured', gid=source_ids[i], color='tab:blue', marker='.', ls='')
			#ax1.plot(rp_mags[i], night_to_nights_theory_calculated[i], alpha=0.3, color='k', label='Theory (unc. / sqrt(N))', marker='.', ls='', gid=source_ids[i])
			ax1.plot(rp_mags[i], night_to_nights_theory_measured[i], alpha=0.3, color='r', label='Theory (measured $\sigma$ / sqrt(N))', marker='.', ls='', gid=source_ids[i])
		else:
			ax1.plot(rp_mags[i], night_to_nights[i], alpha=0.3, gid=source_ids[i], color='tab:blue', marker='.', ls='')
			#ax1.plot(rp_mags[i], night_to_nights_theory_calculated[i], alpha=0.3, color='k', marker='.', ls='', gid=source_ids[i])	
			ax1.plot(rp_mags[i], night_to_nights_theory_measured[i], alpha=0.3, color='r', marker='.', ls='', gid=source_ids[i])
	
	# gaia_var_inds = np.where(source_df['phot_variable_flag'] == 'VARIABLE')[0]
	# ax1.plot(rp_mags[gaia_var_inds], night_to_nights[gaia_var_inds], marker='x', zorder=1, label='Gaia variable', color='m', ls='', mew=1.5, ms=6, alpha=1)
	ax1.plot(weighted_ref_rp,weighted_ref_n2n, marker='x', zorder=1, label='Ref star', color='m', ls='', mew=1.5, ms=6, alpha=1)
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
	ax2.set_xlabel('BJD TDB', fontsize=14)
	ax2.grid()

	ax3.imshow(source_image, origin='lower', norm=simple_norm(source_image, min_percent=1, max_percent=99))
	for i in range(len(lc_files)):
		ax3.plot(x_pos[i], y_pos[i], 'rx', gid=source_ids[i])
	
		ax4.plot(bp_rp[i], G[i], marker='.', color='#b0b0b0', ls='', gid=source_ids[i])
	
	ax4.invert_yaxis()
	ax4.set_aspect('equal')
	ax4.set_xlabel('Bp-Rp', fontsize=14)
	ax4.set_ylabel('G', fontsize=14)

	plt.tight_layout()

	print(f'Median observed/theory (calculated uncertainties): {np.nanmedian(night_to_nights/night_to_nights_theory_calculated):.1f}')
	print(f'Median observed/theory (measured scatter): {np.nanmedian(night_to_nights/night_to_nights_theory_measured):.1f}')

	plt.figure()

	median_ref_bp_rp = np.nanmedian(weighted_ref_bp_rp)

	colormap = plt.cm.viridis
	norm = colors.Normalize(vmax=3,clip=True)
	sc = plt.scatter(rp_mags, night_to_nights/night_to_nights_theory_measured, c=bp_rp-median_ref_bp_rp, norm=norm, cmap=colormap)
	plt.colorbar(sc, label=f'(Bp-Rp)-{median_ref_bp_rp:.2f}')
	plt.tick_params(labelsize=12)
	plt.xlabel('Rp', fontsize=14)
	plt.ylabel('$\sigma_{N2N, measured}$/$\sigma_{N2N, theory}$', fontsize=14)
	plt.yscale('log')
	plt.grid(True, which='both', alpha=0.7)
	plt.tight_layout()

	if field == 'TIC362144730':
		output_dict = {'source_id':source_ids, 'night_to_nights':night_to_nights, 'night_to_nights_theory':night_to_nights_theory_measured}
		output_df = pd.DataFrame(output_dict)
		output_df.to_csv('/home/ptamburo/tierras/pat_scripts/TIC362144730_n2n.csv', index=0)
	breakpoint()
	


if __name__ == '__main__':

	main()