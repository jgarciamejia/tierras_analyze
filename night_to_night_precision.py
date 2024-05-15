import argparse
import numpy as np 
import pandas as pd 
import glob 
import matplotlib.pyplot as plt 
plt.ion()
from scipy.stats import sigmaclip

def main(raw_args=None):

	def on_plot_hover(event):
		# Iterating over each data member plotted
		for curve in ax.get_lines():
			# Searching which data member corresponds to current mouse position
			if curve.contains(event)[0]:
				if curve.get_gid() != None:
					print("over %s" % curve.get_gid())
				# if curve.get_gid() != None:
				# 	ax.patches.clear()
				# 	ind = np.where(source_df['source_id'] == int(str(curve.get_gid())))[0][0]

					# circ1 = plt.Circle((rp_mags[ind], night_to_nights[ind]), 0.2, color='k', zorder=3, fill=False, linewidth=3)
					# ax.add_patch(circ1)  
			
	ap = argparse.ArgumentParser()
	ap.add_argument("-field", required=True, help="Name of observed target field exactly as shown in raw FITS files.")

	args = ap.parse_args()
	field = args.field 

	# TODO: generalize where the source df is read from 
	source_df = pd.read_csv('/data/tierras/photometry/20240512/TIC362144730/flat0000/20240512_TIC362144730_sources.csv')

	lc_files = glob.glob(f'/data/tierras/fields/{field}/sources/lightcurves/**.csv')
	times = np.array(pd.read_csv(lc_files[0], comment='#')['BJD TDB'])
	times -= times[0] 

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
	night_to_nights_theory = np.zeros(len(lc_files))
	rp_mags = np.zeros(len(lc_files))
	source_ids = []
	# plt.figure(figsize=(10,10))
	for i in range(len(lc_files)):
		try:
			source = int(lc_files[i].split('Gaia DR3 ')[1].split('_')[0])
			source_ids.append(source)
			if field == 'TIC362144730' and source in [4147017737256800128, 4146828071505344384, 4146919296602628736]:
				print('Saturated source, skipping.')
				night_to_nights[i] = np.nan 
				night_to_nights_theory[i] = np.nan
				rp_mags[i] = np.nan
				continue
			print(f'Doing Gaia DR3 {source} ({i+1} of {len(lc_files)})')
		except:
			print('Could not identify Gaia DR3 ID, skipping')
			source_ids.append(np.nan)
			night_to_nights[i] = np.nan 
			night_to_nights_theory[i] = np.nan
			rp_mags[i] = np.nan 
			continue
		df = pd.read_csv(lc_files[i], comment='#')
		flux = np.array(df['Flux'])
		flux_err = np.array(df['Flux Error'])
		night_medians = np.zeros(len(times_inds))
		night_errs_on_meds = np.zeros(len(times_inds))
		for j in range(len(night_medians)):
			# TODO: outlier rejection 
			# TODO: eliminate nans from sqrt(len(times_inds[j]))
			night_times = np.array(times_list[j])
			night_flux = flux[times_inds[j]]
			night_err = flux_err[times_inds[j]]
			nan_inds = ~np.isnan(night_flux)

			night_times = night_times[nan_inds]
			night_flux = night_flux[nan_inds]
			night_err = night_err[nan_inds]

			v, l, h = sigmaclip(night_flux, 4, 4)
			sc_inds = np.where((night_flux >= l) & (night_flux <= h))[0]
			sc_times = night_times[sc_inds]
			sc_flux = night_flux[sc_inds]
			sc_err = night_err[sc_inds]

			n_exp = len(sc_flux)

			night_medians[j] = np.nanmedian(sc_flux)
			night_errs_on_meds[j] = 1.2533*np.nanmedian(sc_err)/np.sqrt(n_exp)

		night_to_nights[i] = np.nanstd(night_medians)
		night_to_nights_theory[i] = np.nanmedian(night_errs_on_meds)
		
		source_ind = np.where(source_df['source_id'] == source)[0][0]
		rp_mags[i] = source_df['phot_rp_mean_mag'][source_ind]
		
	fig, ax = plt.subplots(1,1,figsize=(10,10))
	fig.canvas.mpl_connect('motion_notify_event', on_plot_hover)  
	for i in range(len(lc_files)):
		if i == 0:
			ax.plot(rp_mags[i], night_to_nights[i], alpha=0.3, label='Measured', gid=source_ids[i], color='tab:blue', marker='.', ls='')
			ax.plot(rp_mags[i], night_to_nights_theory[i], alpha=0.3, color='k', label='Theory', marker='.', ls='', gid=source_ids[i])
		else:
			ax.plot(rp_mags[i], night_to_nights[i], alpha=0.3, gid=source_ids[i], color='tab:blue', marker='.', ls='')
			ax.plot(rp_mags[i], night_to_nights_theory[i], alpha=0.3, color='k', marker='.', ls='', gid=source_ids[i])	
	ax.set_xlim(np.nanmin(rp_mags)-0.1, np.nanmax(rp_mags)+0.1)
	ax.set_yscale('log')
	# ax.invert_xaxis()
	ax.set_xlabel('Rp', fontsize=14)
	ax.set_ylabel('$\sigma_{N2N}$', fontsize=14)
	ax.grid()
	ax.legend(fontsize=11)
	ax.tick_params(labelsize=12)

	print(f'Median inflation: {np.nanmedian(night_to_nights/night_to_nights_theory):.1f}')
	breakpoint()
	


if __name__ == '__main__':

	main()