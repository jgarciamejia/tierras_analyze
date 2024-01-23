import pdb
import numpy as np
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
plt.ion()
import copy 
from astropy.time import Time 

def reference_flux_correction(bjds, targ_flux, regressors, regressors_err, cs, cs_unc, complist, weights, plot=False):
    regressors_adj = np.zeros_like(regressors) #Adjust the regressor fluxes by cs
    regressors_adj_norm = np.zeros_like(regressors)
    regressors_adj_err = np.zeros_like(regressors)
    regressors_adj_err_norm = np.zeros_like(regressors)
    for ii in range(regressors.shape[0]):
        regressors_adj[ii,:] = regressors[ii,:]*10**(cs/(-2.5))
        regressors_adj_norm[ii,:] = regressors_adj[ii,:] / np.nanmean(regressors_adj[ii,:]) #Create normalized adjusted fluxes
        regressors_adj_err[ii,:] = np.sqrt((10**(-0.4*cs)*regressors_err[ii,:])**2 + (-0.921034*regressors[ii,:]*np.exp(-0.921034*cs)*cs_unc)**2)
        # regressors_adj_err[ii,:] = np.sqrt((10**(-0.4*cs)*regressors_err[ii,:])**2)
        regressors_adj_err_norm[ii,:] = regressors_adj_err[ii,:]/ np.nanmean(regressors_adj[ii,:])
        # breakpoint()

    #Figure out which indises correspond to which date by checking the spacing between exposure times
    time_deltas = np.array([bjds[i+1]-bjds[i] for i in range(len(bjds)-1)])
    date_ends = bjds[np.where(time_deltas>0.5)[0]] #If the separation is > half a day, say that that is where the date ends
    date_inds = []
    for ii in range(len(date_ends)+1):
        if ii == 0:
            date_inds.append(np.where(bjds <= date_ends[ii])[0])
        elif ii < len(date_ends):
            date_inds.append(np.where((bjds > date_ends[ii-1]) & (bjds <= date_ends[ii]))[0])
        else:
            date_inds.append(np.where(bjds > date_ends[-1])[0])

    #Calculate mean fluxes for each star on each night 
        
    binned_flux = np.zeros((regressors.shape[0], len(date_inds)))
    for ii in range(regressors.shape[0]):
        for jj in range(len(date_inds)):
            binned_flux[ii,jj] = np.median(regressors_adj_norm[ii][date_inds[jj]])
    if not plot:
        return binned_flux
    else:
        #Plot uncorrected/corrected reference stars
        colors=  ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan']
        markers = ['.','*','s','v','^','p','P']
        fig, ax = plt.subplots(2, len(date_inds), figsize=(16,5), sharey='row', sharex=True)
        random_offsets = np.random.uniform(-0.5, 0.5, regressors.shape[0])
        #flux_norm = flux / np.nanmean(flux)
        #flux_original_norm = flux_original / np.nanmean(flux_original)
        marker_ind = 0
        for ii in range(regressors.shape[0]):
            color = colors[ii%len(colors)]
            if (ii > 0) and (ii%len(colors)) == 0:
                marker_ind += 1 
            marker = markers[marker_ind]

            for jj in range(len(date_inds)):
                plot_times = bjds[date_inds[jj]]
                ut_date = Time(plot_times[-1],format='jd',scale='tdb').iso.split(' ')[0]
                date_label = ut_date.split('-')[1].lstrip('0')+'/'+ut_date.split('-')[2].lstrip('0')+'/'+ut_date[2:4]
                plot_times -= plot_times[0]
                plot_times *= 24 

                ax[0,jj].set_title(date_label, rotation=45, fontsize=8)
                ax[0,jj].plot(plot_times, regressors[ii][date_inds[jj]]/np.mean(regressors[ii]), marker='.', ls='' ,color=color, alpha=0.5, mec='none', zorder=0)
                ax[1,jj].plot(plot_times, regressors_adj_norm[ii][date_inds[jj]], marker=marker, ls='' ,color=color, alpha=0.5, mec='none', zorder=0)
                mean_bjd = np.mean(plot_times)
                if jj == len(date_inds)-1:
                    ax[1,jj].plot(mean_bjd+random_offsets[ii], binned_flux[ii,jj], marker=marker,color=color, mec='k', mew=1.5, zorder=1,label=f'R{complist[ii]}, {100*weights[ii]:.1f}%', ls='', ms=10)
                else:
                    ax[1,jj].plot(mean_bjd+random_offsets[ii], binned_flux[ii,jj], marker=marker,color=color, mec='k', mew=1.5, zorder=1, ls='', ms=10)

            #plt.plot(bjds, flux/np.mean(flux), marker='.', ls='', label='Target')
        
        # # Plot the corrected target_flux 
        # targ_flux /= np.median(targ_flux)
        # for jj in range(len(date_inds)):
        #     plot_times = bjds[date_inds[jj]]
        #     plot_times -= plot_times[0]
        #     plot_times *= 24 
        #     plot_times = np.mean(plot_times)
        #     mean_flux = np.mean(targ_flux[date_inds[jj]])
        #     ax[1,jj].plot(plot_times,mean_flux, marker='o', mfc='w', mec='k', mew=2, alpha=1, zorder=5, ms=8)
        # breakpoint()

        ax[1,-1].legend(loc='center left', bbox_to_anchor=(1, 1), ncol=2)
        plt.tight_layout()
        plt.subplots_adjust(left=0.04,right=0.8,hspace=0,wspace=0)
        return fig, ax, binned_flux, regressors_adj_norm