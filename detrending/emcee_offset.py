import detrending as dt 
import numpy as np
import matplotlib.pyplot as plt 
import emcee

def log_likelihood(params, X, relfluxes, fluxerr):
    model = dt.f(X,*params)
    sigma2 = fluxerr**2 #+ model**2 * np.exp(2 * log_f)
    return -0.5 * np.sum((relfluxes - model) ** 2 / sigma2 + np.log(sigma2))

def log_prior(params):
    airmass,width,offset = params
    return 0

def log_probability(params, X, relfluxes, fluxerr):
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(params, X, relfluxes, fluxerr)


obsdates = ['20220504','20220505','20220506', '20220507', '20220508',
			'20220509','20220510', '20220511', '20220512','20220513', 
			'20220514','20220515','20220516','20220517']
dateind = 2
date = obsdates[dateind]
data = np.loadtxt('emcee_coeffs.txt')
dateinds,bjds,relfluxes,airmasses,widths,flag = dt.ret_data_arrays([date],5)

opt_model = dt.f((dateinds,airmasses,widths), *data[dateind,:])
per33,per66 = np.percentile(bjds,(33,66))
cond = (bjds > per33) & (bjds < per66)
rmss = np.std(relfluxes[cond] - opt_model[cond])
fluxerrs = np.repeat(rmss,len(relfluxes)) 

pos = np.median(data,axis=0) + np.std(data,axis=0) * np.random.randn(100, 3)
nwalkers, ndim = pos.shape

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, 
	args=((dateinds,airmasses,widths), relfluxes, fluxerrs))
sampler.run_mcmc(pos, 5000, progress=True);

flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
fig = corner.corner(flat_samples)

