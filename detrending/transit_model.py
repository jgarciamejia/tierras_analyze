import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from medsig import *
from bin_lc import *
from astropy.time import Time
from scipy.optimize import curve_fit
import pdb
import sys


def f(X,*params):
	bjds,airmasses,widths,t0,duration = X
	params = list(params)
	assert len(params) == 4 
	model = airmasses * params[0] 
	model += widths * params[1]
	model += box_model(bjds,t0,duration,params[2])
	model += params[3]
	return model

def fit_everything(bjds,relfluxes,airmasses,widths,t0,duration):
	# initial guesses for fit 
	params = np.zeros(4)
	params[3] = np.median(relfluxes)
	relfluxes -= params[3]
	params[0],offset0 = np.polyfit(airmasses,relfluxes,1)
	p0 = np.poly1d([params[0],offset0])
	relfluxes -= p0(airmasses)
	params[1],offset1 = np.polyfit(widths,relfluxes,1)
	p1 = np.poly1d([params[1],offset1])
	relfluxes -= p1(widths)
	# define depth initial guess as relflux median diff AT t0,
	#params[2] = t0 
	#params[3] = duration
	params[2],_,_ = init_guess_depth(bjds,relfluxes,t0,duration)
	# fit model
	popt,pcov = curve_fit(f,(bjds,airmasses,widths),relfluxes,p0=params) 
	return popt,params

def init_guess_depth(bjds,relfluxes,t0,duration):
	assert t0 >= bjds[0] and t0 <= bjds[-1]
	duration /= (60*24) 
	t_ingress,t_egress = t0 - (duration/2),t0 + (duration/2)
	intransit_cond = (bjds > t_ingress) & (bjds < t_egress)
	intransit_med = np.median(relfluxes[intransit_cond])
	t_outleft,t_outright = t_ingress - (duration/2), t_egress + (duration/2)
	outtransit_cond = (bjds >= t_outleft) & (bjds <= t_outright) 
	outtransit_cond = (outtransit_cond) & (np.logical_not(intransit_cond))
	outtransit_med = np.median(relfluxes[outtransit_cond])
	depth_guess = outtransit_med - intransit_med
	return depth_guess, intransit_cond, outtransit_cond

def box_model(bjds,t0,duration,depth):
	assert t0 >= bjds[0] and t0 <= bjds[-1]
	t_ingress,t_egress = t0 - (duration/2),t0 + (duration/2)
	intransit_cond = (bjds > t_ingress) & (bjds < t_egress)
	return numpy.where(intransit_cond, -depth, 0.0)



# t_0 = np.array([])
# durations = np.array([30,60,90,120]) #mins
# for duration in durations:
# 	for date in obsdates:
# 		dateinds,bjds,relfluxes,airmasses,widths,flag = ret_data_arrays([date],5)
# 		popt,init_params = fit_everything(dateinds[flag],relfluxes[flag],airmasses[flag],widths[flag])
# 	# fit coeffs for airmass,width,depth
# 	# calculate chi square of model and save it 
# 	# 
