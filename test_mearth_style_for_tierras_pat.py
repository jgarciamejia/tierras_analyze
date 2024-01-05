import pdb
import numpy as np
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
plt.ion()
import copy 

# Written by Emily Pass, inspired by Jonathan Irwin's MEarth Pipeline

def mearth_style(bjds, flux, err, regressors):

    """ Use the comparison stars to derive a frame-by-frame zero-point magnitude. Also filter and mask bad cadences """
    """ it's called "mearth_style" because it's inspired by the mearth pipeline """
    #pdb.set_trace()
    mask = np.ones_like(flux, dtype='bool')  # initialize a bad data mask
    mask[np.where(flux <= 0)[0]] = 0  # if target counts are less than or equal to 0, this cadence is bad

    # if one of the reference stars has negative flux, this cadence is also bad
    for ii in range(regressors.shape[0]): # for each comp star in 2D array of comps 
        mask[np.where(regressors[ii, :] <= 0)[0]] = 0 # mask out all the places where comp flux values below or equal to 0 

    # apply mask
    regressors = regressors[:, mask]
    flux = flux[mask]
    err = err[mask]
    bjds = bjds[mask]

    tot_regressor = np.sum(regressors, axis=0)  # the total regressor flux at each time point = sum of comp star fluxes in each exposure
    c0s = -2.5*np.log10(np.nanpercentile(tot_regressor, 90)/tot_regressor)  # initial guess of magnitude zero points

    mask = np.ones_like(c0s, dtype='bool')  # initialize another bad data mask
    mask[np.where(c0s < -0.24)[0]] = 0  # if regressor flux is decremented by 20% or more, this cadence is bad

    # apply mask
    regressors = regressors[:, mask]
    flux = flux[mask]
    err = err[mask]
    bjds = bjds[mask]

    # repeat the cs estimate now that we've masked out the bad cadences
    phot_regressor = np.nanpercentile(regressors, 90, axis=1)  # estimate photometric flux level for each star
    cs = -2.5*np.log10(phot_regressor[:,None]/regressors)  # estimate c for each star
    c_noise = np.nanstd(cs, axis=0)  # estimate the error in c
    c_unc = (np.nanpercentile(cs, 84, axis=0) - np.nanpercentile(cs, 16, axis=0)) / 2.  # error estimate that ignores outliers

    ''' c_unc will overestimate error introduced by zero-point offset because it is correlated. Attempt to correct
    for this by only considering the additional error compared to the cadence where c_unc is minimized '''
    c_unc_best = np.min(c_unc)
    c_unc = np.sqrt(c_unc**2 - c_unc_best**2)

    cs = np.nanmedian(cs, axis=0)  # take the median across all regressors

    # one more bad data mask: don't trust cadences where the regressors have big discrepancies
    mask = np.ones_like(flux, dtype='bool')
    mask[np.where(c_noise > 3*np.median(c_noise))[0]] = 0

    # apply mask
    flux = flux[mask]
    err = err[mask]
    bjds = bjds[mask]
    cs = cs[mask]
    c_unc = c_unc[mask]
    regressors = regressors[:,mask]
    # flux_original = copy.deepcopy(flux)
    err = 10**(cs/(-2.5)) * np.sqrt(err**2 + (c_unc*flux*np.log(10)/(-2.5))**2)  # propagate error
    flux *= 10**(cs/(-2.5))  # adjust the flux based on the calculated zero points

    return bjds, flux, err

def mearth_style_pat(bjds, flux, err, regressors):

    """ Use the comparison stars to derive a frame-by-frame zero-point magnitude. Also filter and mask bad cadences """
    """ it's called "mearth_style" because it's inspired by the mearth pipeline """
    #pdb.set_trace()
    mask = np.ones_like(flux, dtype='bool')  # initialize a bad data mask
    mask[np.where(flux <= 0)[0]] = 0  # if target counts are less than or equal to 0, this cadence is bad

    # if one of the reference stars has negative flux, this cadence is also bad
    for ii in range(regressors.shape[0]): # for each comp star in 2D array of comps 
        mask[np.where(regressors[ii, :] <= 0)[0]] = 0 # mask out all the places where comp flux values below or equal to 0 

    # apply mask
    regressors = regressors[:, mask]
    flux = flux[mask]
    err = err[mask]
    bjds = bjds[mask]

    tot_regressor = np.sum(regressors, axis=0)  # the total regressor flux at each time point = sum of comp star fluxes in each exposure
    c0s = -2.5*np.log10(np.nanpercentile(tot_regressor, 90)/tot_regressor)  # initial guess of magnitude zero points

    mask = np.ones_like(c0s, dtype='bool')  # initialize another bad data mask
    mask[np.where(c0s < -0.24)[0]] = 0  # if regressor flux is decremented by 20% or more, this cadence is bad

    # apply mask
    regressors = regressors[:, mask]
    flux = flux[mask]
    err = err[mask]
    bjds = bjds[mask]

    # repeat the cs estimate now that we've masked out the bad cadences
    phot_regressor = np.nanpercentile(regressors, 90, axis=1)  # estimate photometric flux level for each star
    cs = -2.5*np.log10(phot_regressor[:,None]/regressors)  # estimate c for each star
    c_noise = np.nanstd(cs, axis=0)  # estimate the error in c
    c_unc = (np.nanpercentile(cs, 84, axis=0) - np.nanpercentile(cs, 16, axis=0)) / 2.  # error estimate that ignores outliers

    ''' c_unc will overestimate error introduced by zero-point offset because it is correlated. Attempt to correct
    for this by only considering the additional error compared to the cadence where c_unc is minimized '''
    c_unc_best = np.min(c_unc)
    c_unc = np.sqrt(c_unc**2 - c_unc_best**2)

    cs = np.nanmedian(cs, axis=0)  # take the median across all regressors

    # one more bad data mask: don't trust cadences where the regressors have big discrepancies
    mask = np.ones_like(flux, dtype='bool')
    mask[np.where(c_noise > 3*np.median(c_noise))[0]] = 0

    # apply mask
    flux = flux[mask]
    err = err[mask]
    bjds = bjds[mask]
    cs = cs[mask]
    c_unc = c_unc[mask]
    regressors = regressors[:,mask]

    # flux_original = copy.deepcopy(flux)
    err = 10**(cs/(-2.5)) * np.sqrt(err**2 + (c_unc*flux*np.log(10)/(-2.5))**2)  # propagate error
    flux *= 10**(cs/(-2.5))  # adjust the flux based on the calculated zero points


    return bjds, flux, err, regressors, cs, c_unc
