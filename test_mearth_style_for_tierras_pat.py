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

def mearth_style_pat(bjds, flux, err, regressors, regressors_err, exp_times, skies):

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
    regressors_err = regressors_err[:, mask]
    flux = flux[mask]
    err = err[mask]
    bjds = bjds[mask]
    exp_times = exp_times[mask]
    skies = skies[mask]

    tot_regressor = np.sum(regressors, axis=0)  # the total regressor flux at each time point = sum of comp star fluxes in each exposure
    c0s = -2.5*np.log10(np.nanpercentile(tot_regressor, 90)/tot_regressor)  # initial guess of magnitude zero points
    mask = np.ones_like(c0s, dtype='bool')  # initialize another bad data mask
    mask[np.where(c0s < -0.24)[0]] = 0  # if regressor flux is decremented by 20% or more, this cadence is bad

    # apply mask
    regressors = regressors[:, mask]
    regressors_err = regressors_err[:, mask]
    flux = flux[mask]
    err = err[mask]
    bjds = bjds[mask]
    exp_times = exp_times[mask]
    skies = skies[mask]

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
    regressors = regressors[:, mask]
    regressors_err = regressors_err[:, mask]
    exp_times = exp_times[mask]
    skies = skies[mask]

    # flux_original = copy.deepcopy(flux)
    err = 10**(cs/(-2.5)) * np.sqrt(err**2 + (c_unc*flux*np.log(10)/(-2.5))**2)  # propagate error
    flux *= 10**(cs/(-2.5))  #cs, adjust the flux based on the calculated zero points


    return bjds, flux, err, regressors, regressors_err, cs, c_unc, exp_times, skies

def mearth_style_pat_weighted(bjds, flux, err, regressors, regressors_err, exp_times, skies, x_pos, y_pos, airmass, fwhm, bjds_list):

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
    regressors_err = regressors_err[:, mask]
    flux = flux[mask]
    err = err[mask]
    bjds = bjds[mask]
    exp_times = exp_times[mask]
    skies = skies[mask]
    x_pos = x_pos[mask]
    y_pos = y_pos[mask]
    airmass = airmass[mask]
    fwhm = fwhm[mask]

    tot_regressor = np.sum(regressors, axis=0)  # the total regressor flux at each time point = sum of comp star fluxes in each exposure
    c0s = -2.5*np.log10(np.nanpercentile(tot_regressor, 90)/tot_regressor)  # initial guess of magnitude zero points
    mask = np.ones_like(c0s, dtype='bool')  # initialize another bad data mask
    mask[np.where(c0s < -0.24)[0]] = 0  # if regressor flux is decremented by 20% or more, this cadence is bad

    # apply mask
    regressors = regressors[:, mask]
    regressors_err = regressors_err[:, mask]
    flux = flux[mask]
    err = err[mask]
    bjds = bjds[mask]
    exp_times = exp_times[mask]
    skies = skies[mask]
    x_pos = x_pos[mask]
    y_pos = y_pos[mask]
    airmass = airmass[mask]
    fwhm = fwhm[mask]

    # do a mask on airmass values
    mask = np.ones_like(airmass, dtype='bool')
    mask[np.where(airmass > 2)[0]] = 0

    # apply mask
    regressors = regressors[:, mask]
    regressors_err = regressors_err[:, mask]
    flux = flux[mask]
    err = err[mask]
    bjds = bjds[mask]
    exp_times = exp_times[mask]
    skies = skies[mask]
    x_pos = x_pos[mask]
    y_pos = y_pos[mask]
    airmass = airmass[mask]
    fwhm = fwhm[mask]

    # repeat the cs estimate now that we've masked out the bad cadences
    phot_regressor = np.nanpercentile(regressors, 90, axis=1)  # estimate photometric flux level for each star
    cs = -2.5*np.log10(phot_regressor[:,None]/regressors)  # estimate c for each star
    c_noise = np.nanstd(cs, axis=0)  # estimate the error in c
    c_unc = (np.nanpercentile(cs, 84, axis=0) - np.nanpercentile(cs, 16, axis=0)) / 2.  # error estimate that ignores outliers



    ''' c_unc will overestimate error introduced by zero-point offset because it is correlated. Attempt to correct
    for this by only considering the additional error compared to the cadence where c_unc is minimized '''
    c_unc_best = np.min(c_unc)
    c_unc = np.sqrt(c_unc**2 - c_unc_best**2)

    # Initialize weights using average fluxes of the regressors
    weights_init = np.mean(regressors, axis=1)
    weights_init /= np.sum(weights_init) # Normalize weights to sum to 1

    
    cs = np.matmul(weights_init, cs) # Take the *weighted mean* across all regressors
    # cs = np.nanmedian(cs, axis=0)  # take the median across all regressors

    # one more bad data mask: don't trust cadences where the regressors have big discrepancies
    mask = np.ones_like(flux, dtype='bool')
    mask[np.where(c_noise > 3*np.median(c_noise))[0]] = 0

    # apply mask
    flux = flux[mask]
    err = err[mask]
    bjds = bjds[mask]
    cs = cs[mask]
    c_unc = c_unc[mask]
    regressors = regressors[:, mask]
    regressors_err = regressors_err[:, mask]
    exp_times = exp_times[mask]
    skies = skies[mask]
    x_pos = x_pos[mask]
    y_pos = y_pos[mask]
    airmass = airmass[mask]
    fwhm = fwhm[mask]

    # loop over nights and remove those with little data 
    nights_remove = []
    for jj in range(len(bjds_list)):
        night_inds = np.where((bjds>=bjds_list[jj][0])&(bjds<bjds_list[jj][-1]))[0]
        if sum(exp_times[night_inds])/3600 < 2:
            nights_remove.append(jj)
            flux = np.delete(flux, night_inds)
            err = np.delete(err, night_inds)
            bjds = np.delete(bjds, night_inds)
            cs = np.delete(cs, night_inds)
            c_unc = np.delete(c_unc, night_inds)
            regressors = np.delete(regressors, night_inds, axis=1)
            regressors_err = np.delete(regressors_err, night_inds, axis=1)
            exp_times = np.delete(exp_times, night_inds)
            skies = np.delete(skies, night_inds)
            x_pos = np.delete(x_pos, night_inds)
            y_pos = np.delete(y_pos, night_inds)
            airmass = np.delete(airmass, night_inds)
            fwhm = np.delete(fwhm, night_inds)
    nights_remove = np.array(nights_remove)
    bjds_list = np.delete(bjds_list, nights_remove)

    cs_original = cs
    delta_weights = np.zeros(len(regressors))+999 # initialize
    threshold = 1e-4 # delta_weights must converge to this value for the loop to stop
    weights_old = weights_init
    full_ref_inds = np.arange(len(regressors))
    while len(np.where(delta_weights>threshold)[0]) > 0:
        stddevs = np.zeros(len(regressors))
        cs = -2.5*np.log10(phot_regressor[:,None]/regressors)

        for jj in range(len(regressors)):
            use_inds = np.delete(full_ref_inds, jj)
            weights_wo_jj = weights_old[use_inds]
            weights_wo_jj /= np.sum(weights_wo_jj)
            cs_wo_jj = np.matmul(weights_wo_jj, cs[use_inds])
            corr_jj = regressors[jj] * 10**(-cs_wo_jj/2.5)
            corr_jj /= np.mean(corr_jj)
            stddevs[jj] = np.std(corr_jj)

        weights_new = 1/stddevs**2
        weights_new /= np.sum(weights_new)
        delta_weights = abs(weights_new-weights_old)

        weights_old = weights_new

    weights = weights_new

    # determine if any references should be totally thrown out based on the ratio of their measured/expected noise
    regressors_err_norm = (regressors_err.T / np.median(regressors,axis=1)).T
    noise_ratios = stddevs / np.median(regressors_err_norm)       
    weights[np.where(noise_ratios>10)[0]] = 0
    weights /= sum(weights)

    if len(np.where(weights == 0)[0]) > 0:
        # now repeat the weighting loop with the bad refs removed 
        delta_weights = np.zeros(len(regressors))+999 # initialize
        threshold = 1e-6 # delta_weights must converge to this value for the loop to stop
        weights_old = weights
        full_ref_inds = np.arange(len(regressors))
        count = 0
        while len(np.where(delta_weights>threshold)[0]) > 0:
            stddevs = np.zeros(len(regressors))
            cs = -2.5*np.log10(phot_regressor[:,None]/regressors)

            for jj in range(len(regressors)):
                if weights_old[jj] == 0:
                    continue
                use_inds = np.delete(full_ref_inds, jj)
                weights_wo_jj = weights_old[use_inds]
                weights_wo_jj /= np.nansum(weights_wo_jj)
                cs_wo_jj = np.matmul(weights_wo_jj, cs[use_inds])
                corr_jj = regressors[jj] * 10**(-cs_wo_jj/2.5)
                corr_jj /= np.nanmean(corr_jj)
                stddevs[jj] = np.nanstd(corr_jj)
            weights_new = 1/(stddevs**2)
            weights_new /= np.sum(weights_new[~np.isinf(weights_new)])
            weights_new[np.isinf(weights_new)] = 0
            delta_weights = abs(weights_new-weights_old)
            weights_old = weights_new
            count += 1

    weights = weights_new

    # calculate the zero-point correction
    cs = -2.5*np.log10(phot_regressor[:,None]/regressors)
    cs = np.matmul(weights, cs)
    
    corrected_regressors = regressors * 10**(-cs/2.5)

    # flux_original = copy.deepcopy(flux)
    err = 10**(cs/(-2.5)) * np.sqrt(err**2 + (c_unc*flux*np.log(10)/(-2.5))**2)  # propagate error
    flux *= 10**(cs/(-2.5))  #cs, adjust the flux based on the calculated zero points


    return bjds, flux, err, regressors, regressors_err, cs, c_unc, exp_times, skies, weights, x_pos, y_pos, airmass, fwhm, bjds_list