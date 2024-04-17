import pdb
import numpy as np
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
plt.ion()
import copy 
from astropy.time import Time 
from ap_phot import regression 
from scipy.stats import pearsonr 

def reference_flux_correction(bjd, bjd_list,  regressors, regressors_err, cs, cs_unc, complist, weights, airmass, fwhm):
    regressors_adj = np.zeros_like(regressors) #Adjust the regressor fluxes by cs
    regressors_adj_norm = np.zeros_like(regressors)
    regressors_adj_err = np.zeros_like(regressors)
    regressors_adj_err_norm = np.zeros_like(regressors)
    ancillary_dict = {'Airmass':airmass, 'FWHM':fwhm}
    for ii in range(regressors.shape[0]):
        regressors_adj[ii,:] = regressors[ii,:]*10**(cs/(-2.5))
        norm = np.nanmean(regressors_adj[ii,:])
        regressors_adj_norm[ii,:] = regressors_adj[ii,:] / norm #Create normalized adjusted fluxes
        regressors_adj_err[ii,:] = np.sqrt((10**(-0.4*cs)*regressors_err[ii,:])**2 + (-0.921034*regressors[ii,:]*np.exp(-0.921034*cs)*cs_unc)**2)
        # regressors_adj_err[ii,:] = np.sqrt((10**(-0.4*cs)*regressors_err[ii,:])**2)
        regressors_adj_err_norm[ii,:] = regressors_adj_err[ii,:]/ np.nanmean(regressors_adj[ii,:])

        # correct the adjusted flux with regression agains airmass and FWHM 
        # reg_flux, intercept, coeffs, ancillary_dict_return = regression(regressors_adj_norm[ii], ancillary_dict, pval_threshold=1e-3, verbose=False)
        # regressors_adj_norm[ii,:] = reg_flux
        # regressors_adj[ii,:] = reg_flux * norm


    binned_flux = np.zeros((regressors.shape[0], len(bjd_list)))

    fig, ax = plt.subplots(2, len(bjd_list), figsize=(16,5), sharey='row', sharex=True)
    colors=  ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan']
    markers = ['.','*','s','v','^','p','P']
    random_offsets = np.random.uniform(-0.5, 0.5, regressors.shape[0])
    for ii in range(len(bjd_list)):
        use_inds = np.where((bjd >= bjd_list[ii][0])&(bjd <= bjd_list[ii][-1]))[0]
        plot_times = bjd[use_inds]
        ut_date = Time(plot_times[-1],format='jd',scale='tdb').iso.split(' ')[0]
        date_label = ut_date.split('-')[1].lstrip('0')+'/'+ut_date.split('-')[2].lstrip('0')+'/'+ut_date[2:4]
        plot_times -= plot_times[0]
        plot_times *= 24 
        ax[0,ii].set_title(date_label, rotation=45, fontsize=8)
        
        marker_ind = 0
        for jj in range(regressors.shape[0]):
            color = colors[jj%len(colors)]
            if (jj > 0) and (jj%len(colors)) == 0:
                marker_ind += 1 
            marker = markers[marker_ind]

            binned_flux[jj,ii] = np.median(regressors_adj_norm[jj][use_inds])

            if weights[jj] == 0:
                continue 


            ax[0,ii].plot(plot_times, regressors[jj][use_inds]/np.mean(regressors[jj]), marker='.', ls='' ,color=color, alpha=0.5, mec='none', zorder=0)
            ax[1,ii].plot(plot_times, regressors_adj_norm[jj][use_inds], marker=marker, ls='' ,color=color, alpha=0.5, mec='none', zorder=0)
            mean_bjd = np.mean(plot_times)
            if ii == len(bjd_list)-1:
                ax[1,ii].plot(mean_bjd+random_offsets[jj], binned_flux[jj,ii], marker=marker,color=color, mec='k', mew=1.5, zorder=1,label=f'R{complist[jj]}, {100*weights[jj]:.1f}%', ls='', ms=10)
            else:
                ax[1,ii].plot(mean_bjd+random_offsets[jj], binned_flux[jj,ii], marker=marker,color=color, mec='k', mew=1.5, zorder=1, ls='', ms=10)

    ax[1,-1].legend(loc='center left', bbox_to_anchor=(1, 1), ncol=2)
    plt.tight_layout()
    plt.subplots_adjust(left=0.04,right=0.8,hspace=0,wspace=0)
    
    return fig, ax, binned_flux, regressors_adj_norm