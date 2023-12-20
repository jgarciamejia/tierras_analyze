import numpy as np
import pdb
import pandas as pd
from astropy.timeseries import LombScargle
from glob import glob
from time import time
import matplotlib.pylab as plt
import pymc3 as pm  # installed as part of exoplanet
import pymc3_ext as pmx  # installed as part of exoplanet
from celerite2.theano import terms, GaussianProcess  # installed as part of exoplanet

# Author: EPass
def build_model(x,y,err,mask,period1,N=1):
    # a pymc3 model, inspired by the "exoplanet" package case studies
    # note that this model does not fit for period -- it fixes period at the LS value

    with pm.Model() as model:

        # an offset term
        mean_lc = pm.Normal("mean_lc", mu=0.0, sd=50.0)

        # I don't actually want to use a GP but I want to use "exoplanet"'s marginalization framework, so I've set the
        # GP kernel to have a zero amplitude hyperparameter (i.e., it can't do anything).
        kernel_lc = terms.SHOTerm(sigma=0., rho=1., Q=1.0 / 3)

        # How many sinusoids to use in the spot model? N=1 is fundamental mode only, N=2 adds the 2nd harmonic, etc.
        #N = 1
        print ('spot model using {} sinusoids'.format(N))
        coeffs_a = []
        phases_a = []

        for ii in range(N):
            # create appropriate fitting parameters based on N
            coeffs_a.append(
                pm.Uniform("coeffa_" + str(ii), lower=0, upper=1, testval=0.01))  # spot harmonic coefficients
            phases_a.append(pm.Uniform("phasea_" + str(ii), lower=0., upper=1, testval=0.2))  # spot harmonic phases

        # this spot model is defined in equation 1 of Hartman et al. 2018
        # it's called spota, because in the version of the code with multiple stars there'd also be a spotb
        def spota(t):
            total = coeffs_a[0] * (np.cos(2 * np.pi * (t / period1 + phases_a[0])))
            for ii in range(1, N):
                total += coeffs_a[0] * coeffs_a[ii] * np.cos(
                    2 * np.pi * (t * (ii + 1) / period1 + phases_a[ii] + (ii + 1) * phases_a[0]))
            return total

        # our model is the spot model plus a y-axis offset
        def model_lc(t):
            return mean_lc + 1e3 * spota(t)

        # Condition the light curve model on the data
        gp_lc = GaussianProcess(kernel_lc, t=x[mask], yerr=err[mask])
        gp_lc.marginal("obs_lc", observed=y[mask] - model_lc(x[mask]))

        # Optimize the logp
        map_soln = model.test_point
        map_soln = pmx.optimize(map_soln)

        # retain important variables for later
        extras = dict(x=x[mask], y=y[mask], yerr=err[mask], model_lc=model_lc, gp_lc_pred=gp_lc.predict(y[mask] - model_lc(x[mask])), spota=spota)

    return model, map_soln, extras


def sigma_clip(x,y,err,period1,N):
    # this sigma clipping routine is from "exoplanet" and iteratively removes outliers from the fit

    mask = np.ones(len(x), dtype=bool)
    num = len(mask)

    for i in range(10):
        model, map_soln, extras = build_model(x,y,err,mask,period1,N)

        with model:
            mdl = pmx.eval_in_model(
                extras["model_lc"](extras["x"]) + extras["gp_lc_pred"],
                map_soln,
            )

        resid = y[mask] - mdl
        sigma = np.sqrt(np.median((resid - np.median(resid)) ** 2))
        mask[mask] = np.abs(resid - np.median(resid)) < 7 * sigma
        print("Sigma clipped {0} light curve points".format(num - mask.sum()))
        if num - mask.sum() < 10:
            break
        num = mask.sum()

    return model, map_soln, extras
