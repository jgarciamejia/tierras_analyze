import numpy as np 
import matplotlib.pyplot as plt 
plt.ion()
import pandas as pd 
import argparse
from glob import glob
from scipy.stats import sigmaclip
from astropy.io import fits 
from analyze_global import identify_target_gaia_id

def main(raw_args=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('-field', required=True, help='Name of field')
    ap.add_argument('-ffname', required=False, default='flat0000', help='Name of flat directory')
    args = ap.parse_args(raw_args)
    field = args.field
    ffname = args.ffname

    field_path = f'/data/tierras/fields/{field}/sources/lightcurves/{ffname}/'
    global_lcs = sorted(glob(field_path+'*_global_lc.csv'))
    sources_path = f'/data/tierras/fields/{field}/sources/{field}_common_sources.csv'
    sources_df = pd.read_csv(sources_path)
    if len(global_lcs) != len(sources_df):
        raise RuntimeError(f'There is a mismatch between the number of global light curves in {field_path} and the number of common sources recorded in {sources_path}.')

    # need a list of dates for the field so we can identify the gaia id of the target
    date_list = glob(f'/data/tierras/photometry/**/{field}/{ffname}')	
    date_list = np.array(sorted(date_list, key=lambda x:int(x.split('/')[4]))) # sort on date so that everything is in order
    dates = np.array([i.split('/')[4] for i in date_list])
    fpath = '/data/tierras/flattened/'
    hdr = fits.open(glob(fpath+f'{dates[-1]}'+f'/{field}/{ffname}/*.fit')[0])[0].header
    targ_x_pix = hdr['CAT-X']
    targ_y_pix = hdr['CAT-Y']

    target_ind = None
    median_fluxes = np.zeros(len(global_lcs))
    gaia_rps = np.zeros_like(median_fluxes)
    for i in range(len(global_lcs)):
        path = global_lcs[i]
        if field in path.split('/')[-1]:
            target_ind = i
            target = field
            breakpoint()
            gaia_id = identify_target_gaia_id(target, x_pix=targ_x_pix, y_pix=targ_y_pix)
        else:
            gaia_id = int(global_lcs[i].split('/')[-1].split('_')[0].split(' ')[-1])

        df = pd.read_csv(path, comment='#')
        raw_flux = np.array(df['Raw Flux (ADU)'])
        raw_flux_err = np.array(df['Raw Flux Error (ADU)'])
        nan_inds = ~np.isnan(raw_flux)
        raw_flux = raw_flux[nan_inds]
        raw_flux_err = raw_flux_err[nan_inds]

        try:
            source_ind = np.where(sources_df['source_id'] == gaia_id)[0][0]
        except:
            gaia_rps[i] = np.nan 
            median_fluxes[i] = np.nan 
            print(f'could not find corresponding entry for Gaia DR3 {gaia_id} in the sources csv')
            continue 
        
        gaia_rps[i] = sources_df['phot_rp_mean_mag'][source_ind]

        v, lo, hi = sigmaclip(raw_flux)
        median_fluxes[i] = np.median(v)
    
    plt.plot(gaia_rps, median_fluxes, 'k.')
    plt.plot(gaia_rps[target_ind], median_fluxes[target_ind], 'ro')
    plt.yscale('log')
    breakpoint()


if __name__ == '__main__':
    main()