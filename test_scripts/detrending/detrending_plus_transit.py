import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pdb 

def box_model(bjds,t0,duration,depth):
	# bjds,t0 and duration MUST be in the same units
	assert t0 >= bjds[0] and t0 <= bjds[-1]
	t_ingress,t_egress = t0 - (duration/2),t0 + (duration/2)
	intransit_cond = (bjds > t_ingress) & (bjds < t_egress)
	return np.where(intransit_cond, -depth, 0.0)

def f2(t0,duration):
	def f(X,*params):
		bjds,airmasses,widths = X
		params = list(params)
		assert len(params) == 4
		model = airmasses * params[0] 
		model += widths * params[1]
		model += box_model(bjds,t0,duration,params[2])
		model += params[3]
		return model
	return f

def f_comps(X,*params):
	bjds,airmasses,widths,t0,duration = X
	params = list(params)
	assert len(params) == 4
	model0 = airmasses * params[0] 
	model1 = widths * params[1]
	model2 = box_model(bjds,t0,duration,params[2])
	model3 = params[3]
	return model0, model1, model2, model3

def fit_everything(bjds,relfluxes,airmasses,widths,t0,duration):
	# initial guesses for fit 
	params = np.zeros(4)
	param_bounds = ((-np.inf,-np.inf,0,-np.inf),(np.inf,np.inf,np.inf,np.inf))
	params[0],_ = np.polyfit(airmasses,relfluxes,1)
	params[1],_ = np.polyfit(widths,relfluxes,1)
	params[2],_,_ = init_guess_depth(bjds,relfluxes,t0,duration)
	init_model = f2(t0,duration)((bjds,airmasses,widths), *params)
	params[3] = np.median(relfluxes - init_model)
	popt,pcov = curve_fit(f2(t0,duration),(bjds,airmasses,widths),
				relfluxes,p0=params, bounds = param_bounds) 
	return params, popt

def init_guess_depth(bjds,relfluxes,t0,duration): # TBD: use box model
	assert t0 >= bjds[0] and t0 <= bjds[-1]
	t_ingress,t_egress = t0 - (duration/2),t0 + (duration/2)
	intransit_cond = (bjds > t_ingress) & (bjds < t_egress)
	intransit_med = np.median(relfluxes[intransit_cond])
	t_outleft,t_outright = t_ingress - (duration/2), t_egress + (duration/2)
	outtransit_cond = (bjds >= t_outleft) & (bjds <= t_outright) 
	outtransit_cond = (outtransit_cond) & (np.logical_not(intransit_cond))
	outtransit_med = np.median(relfluxes[outtransit_cond])
	depth_guess = outtransit_med - intransit_med
	if depth_guess < 0: depth_guess = 0 
	return depth_guess, intransit_cond, outtransit_cond

def calc_model_chisq(bjds,norm_relfluxes,airmasses,widths,
					t0_guess,duration,rms):
	popt,pinit = fit_everything(bjds,norm_relfluxes,airmasses,
								widths,t0_guess,duration)
	opt_model = f2(t0_guess,duration)((bjds,airmasses,widths), *popt)
	init_model = f2(t0_guess,duration)((bjds,airmasses,widths), *pinit)
	chisq = np.sum(((norm_relfluxes - opt_model)/(rms))**2)/len(norm_relfluxes)
	return chisq, popt, pinit, opt_model, init_model

def calc_chisq_vs_bjd(obsdate,bjds,norm_relfluxes,airmasses,widths,
					  t0_step,duration,rms,rmsind,window_size,plot=True):
	chisqs = []
	best_chisq = 1e6
	# iterate through t0s to ID model with min chi sq
	for t0_guess in bjds[::t0_step]:
		chisq, popt, pinit, opt_model, init_model = calc_model_chisq(bjds,
													norm_relfluxes,airmasses,
													widths,t0_guess,duration,rms)
		chisqs.append(chisq)

		if chisq < best_chisq:
			best_chisq = chisq
			best_t0 = t0_guess
			best_params = popt
			best_model = opt_model

	if plot:
		# plot data,model with min chi sq and chi sq vs BJD
		fig, (ax1,ax2) = plt.subplots(1,2,figsize=(18,8))
		
		ax1.scatter(bjds[::t0_step],chisqs,color='tab:green')
		ax1.scatter(best_t0,best_chisq,color='tab:orange')
		ax1.axvspan(bjds[rmsind],bjds[rmsind+window_size-1],color='gray',alpha=0.3)
		ax1.set_xlabel('t0 guess (BJD)')
		ax1.set_ylabel('Chi Square')
		ax1.get_xaxis().set_major_formatter(plt.FormatStrFormatter('%.2f'))

		ax2.scatter(bjds,norm_relfluxes,label='Data',color='tab:blue')
		ax2.plot(bjds,best_model,label ='Model w/ Min Chi Sq',color='tab:orange')
		ax2.set_xlabel('BJD')
		ax2.set_ylabel('Norm. Flux')
		ax2.axvspan(bjds[rmsind],bjds[rmsind+window_size-1],color='gray',alpha=0.3,label='Min. RMS = {0:.3f} ppt, No. pts = {1} (fixed)'.format(rms*1e3,window_size))
		ax2.get_xaxis().set_major_formatter(plt.FormatStrFormatter('%.2f'))

		fig.suptitle(obsdate)
		fig.tight_layout()
		plt.legend(loc='upper right')
		plt.show()
	return np.array(chisqs), best_chisq, best_params, best_model

def calc_rms_region(norm_relfluxes, airmasses, skylevs, 
					maxairmass, maxskylev,window_size):
	airmass_flag = airmasses <= maxairmass
	skylev_flag = skylevs <= maxskylev
	comb_flag = airmass_flag & skylev_flag

	start_ind,last_ind = 0, window_size
	rmss = []
	while last_ind <= len(norm_relfluxes):
		rmss.append(np.std(norm_relfluxes[start_ind:last_ind]))
		start_ind += 1
		last_ind += 1
	rmss = np.array(rmss)
	min_rms, min_ind = np.amin(rmss), np.argmin(rmss)
	return rmss,min_rms, min_ind
