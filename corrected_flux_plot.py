import pdb
import numpy as np
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
plt.ion()
import copy 

def corrected_flux_plot(bjds, regressors, cs):
    regressors_adj = np.zeros_like(regressors) #Adjust the regressor fluxes by cs
    regressors_adj_norm = np.zeros_like(regressors)
    for ii in range(regressors.shape[0]):
        regressors_adj[ii,:] = regressors[ii,:]*10**(cs/(-2.5))
        regressors_adj_norm[ii,:] = regressors_adj[ii,:] / np.nanmean(regressors_adj[ii,:])

    time_deltas = np.array([bjds[i+1]-bjds[i] for i in range(len(bjds)-1)])
    date_ends = bjds[np.where(time_deltas>0.8)[0]]
    date_inds = []
    for ii in range(len(date_ends)+1):
        if ii == 0:
            date_inds.append(np.where(bjds <= date_ends[ii])[0])
        elif ii < len(date_ends):
            date_inds.append(np.where((bjds > date_ends[ii-1]) & (bjds <= date_ends[ii]))[0])
        else:
            date_inds.append(np.where(bjds > date_ends[-1])[0])
    colors=  ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan']
    markers = ['.','*']
    binned_flux = np.zeros((regressors.shape[0], len(date_inds)))
    fig, ax = plt.subplots(2, len(date_inds), figsize=(16,5), sharey='row', sharex=True)
    random_offsets = np.random.uniform(-0.01, 0.01, regressors.shape[0])
    #flux_norm = flux / np.nanmean(flux)
    #flux_original_norm = flux_original / np.nanmean(flux_original)
    for ii in range(regressors.shape[0]):
        color = colors[ii%len(colors)]
        if ii < 10:
            marker = markers[0]
        else:
            marker = markers[1]
        for jj in range(len(date_inds)):
            plot_times = bjds[date_inds[jj]]
            plot_times -= plot_times[0]
            plot_times *= 24 
            ax[0,jj].plot(plot_times, regressors[ii][date_inds[jj]]/np.mean(regressors[ii]), marker='.', ls='' ,color=color, alpha=1, mec='none', zorder=0)
            #ax[0,jj].plot(bjds[date_inds[jj]], 10**(cs[date_inds[jj]]/(-2.5)), lw=2, color='k')
            ax[1,jj].plot(plot_times, regressors_adj_norm[ii][date_inds[jj]], marker=marker, ls='' ,color=color, alpha=1, mec='none', zorder=0)
            #if ii == 0:
                #ax[0,jj].plot(plot_times,flux_original_norm[date_inds[jj]], marker=marker, color='k', ls='', zorder=3)
                #ax[1,jj].plot(plot_times,flux_norm[date_inds[jj]], marker='.', color='k', ls='', zorder=3)
            mean_bjd = np.mean(plot_times)
            mean_flux = np.median(regressors_adj_norm[ii][date_inds[jj]])
            binned_flux[ii, jj] = mean_flux
            if jj == 0:
                ax[1,jj].plot(mean_bjd+random_offsets[ii], mean_flux, marker=marker,color=color, mec='k', mew=1.5, zorder=1,label=f'Ref {ii+1}', ls='', ms=10)
            else:
                ax[1,jj].plot(mean_bjd+random_offsets[ii], mean_flux, marker=marker,color=color, mec='k', mew=1.5, zorder=1, ls='', ms=10)

        #plt.plot(bjds, flux/np.mean(flux), marker='.', ls='', label='Target')
    ax[1,jj].legend()
    plt.tight_layout()
    plt.subplots_adjust(hspace=0,wspace=0)
    return fig, ax, binned_flux