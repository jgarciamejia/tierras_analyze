import numpy as np 
import matplotlib.pyplot as plt
plt.ion()
import glob 
import pandas as pd
from scipy.stats import sigmaclip
import os 

def noise_component_plot(ap_rad=10, exp_time=30, sky=100):
    conversion = 1e6 # 1e3 for ppt, 1e6 for ppm 
    
    n_pix = np.pi*ap_rad**2
    READ_NOISE = 18.5
    DARK_CURRENT = 0.19

    source_photon_counts = np.logspace(np.log10(1e4),np.log10(3e7),1000)

    sky_photon_counts = sky*n_pix
    source_photon_noise = np.sqrt(source_photon_counts) / source_photon_counts
    sky_photon_noise = np.sqrt(sky_photon_counts) / source_photon_counts
    read_noise = np.sqrt((np.zeros_like(source_photon_counts) + n_pix * READ_NOISE**2)) / source_photon_counts
    dc_noise = np.sqrt((np.zeros_like(source_photon_counts) + n_pix * DARK_CURRENT * exp_time)) / source_photon_counts
    pwv_noise = (np.zeros_like(source_photon_counts) + 250/1e6) / source_photon_counts

    noise_floor = 250 / 1e6 * source_photon_counts / source_photon_counts

    total_noise = (np.sqrt(source_photon_counts + sky_photon_counts  + n_pix*(DARK_CURRENT*exp_time + READ_NOISE**2)) + noise_floor*source_photon_counts)/ source_photon_counts
    
    fig, ax = plt.subplots(1,1,figsize=(10,7))
    ax.plot(source_photon_counts, source_photon_noise*conversion, label='Source', ls='--')
    ax.plot(source_photon_counts, sky_photon_noise*conversion, label=f'Sky ({sky}) phot./pix.', ls='--')
    #ax.plot(source_photon_counts, read_noise*conversion, label='Read noise')
    #ax.plot(source_photon_counts, dc_noise*conversion, label='Dark current')
    ax.plot(source_photon_counts, noise_floor*conversion, label='Noise floor', ls='--')
    ax.plot(source_photon_counts, total_noise*conversion, label='Total noise', lw=2)

    ax.legend()
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.invert_xaxis()
    ax.tick_params(labelsize=14)
    ax.set_xlabel('Source counts (photons)', fontsize=16)
    if conversion == 1e3:
        ax.set_ylabel('$\sigma$ (ppt)', fontsize=16)
    elif conversion == 1e6:
        ax.set_ylabel('$\sigma$ (ppm)', fontsize=16)
    else:
        ax.set_ylabel('$\sigma$', fontsize=16)
    ax.set_ylim(80,1e6)
    ax.set_xlim(max(source_photon_counts),min(source_photon_counts))
    plt.tight_layout()
    return fig, ax 

if __name__ == '__main__':
    plot_exptime = 30 #seconds #JGM add
    fig, ax = noise_component_plot(exp_time=plot_exptime)
    targets = glob.glob('/data/tierras/targets/*')
    for i in range(len(targets)):
        target = targets[i].split('/')[-1]
        dates = glob.glob(f'/data/tierras/lightcurves/**/{target}')
        for j in range(len(dates)):
            date = dates[j].split('/')[-2]
            print(target, date)
            try:
                optimal_lc_path = f'/data/tierras/lightcurves/{date}/{target}/flat0000/optimal_lc.txt'
                if os.path.exists(optimal_lc_path):
                    with open(optimal_lc_path) as f:
                        path = f.readline()
                else:
                    optimal_lc_path = f'/data/tierras/lightcurves/{date}/{target}/flat000/optimal_lc.txt'
                    if os.path.exists(optimal_lc_path):
                        with open(optimal_lc_path) as f:
                            path = f.readline()
                    else:
                        print(f'No optimal output for {target} on {date}, skipping.')
                        continue
                df = pd.read_csv(path)
            except:
                continue
            n_refs = int(df.keys()[-1].split(' ')[1])
            target_rel_flux = np.array(df['Target Relative Flux'])
            target_raw_flux = np.array(df['Target Source-Sky e'])
            native_exptime = np.array(df['Exposure Time'])
            v, l, h = sigmaclip(target_rel_flux)
            use_inds = np.where((target_rel_flux>l)&(target_rel_flux<h))[0]
            target_rel_flux = target_rel_flux[use_inds]
            target_raw_flux = target_raw_flux[use_inds]
            norm = np.median(target_rel_flux)
            target_rel_flux /= norm
            stddev = np.nanstd(target_rel_flux)*np.sqrt(native_exptime/plot_exptime)*1e6 #JGM mod
            ax.plot(np.mean(target_raw_flux), stddev, '.', color='k', ls='', alpha=0.4, ms=3)
            for k in range(n_refs):
                ref_rel_flux = np.array(df[f'Ref {k+1} Relative Flux'])
                ref_raw_flux = np.array(df[f'Ref {k+1} Source-Sky e'])
                v, l, h = sigmaclip(ref_rel_flux)
                use_inds = np.where((ref_rel_flux>l)&(ref_rel_flux<h))[0]
                ref_rel_flux = ref_rel_flux[use_inds]
                ref_raw_flux = ref_raw_flux[use_inds]
                norm = np.median(ref_rel_flux)
                ref_rel_flux /= norm
                stddev = np.nanstd(ref_rel_flux)*np.sqrt(native_exptime/plot_exptime)*1e6 #JGM mod
                ax.plot(np.mean(ref_raw_flux), stddev, '.', color='k', ls='', alpha=0.4, ms=3)
    breakpoint()