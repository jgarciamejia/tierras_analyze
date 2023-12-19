import numpy as np
import pdb
import matplotlib.pyplot as plt
import importlib 

import detrending_plus_transit as dpt
import load_data as ld
import bin_lc as bl

importlib.reload(dpt)
importlib.reload(ld)
importlib.reload(bl)

# User-defined 
mainpath = '/Users/jgarciamejia/Documents/TierrasProject/SCIENCE/AIJ_Output_Ryan_fixedaperture/'
targetname = 'toi2013'
date = '20220509'
threshold = 5
t0_step = 4
duration = 69.68303985893726 / (24*60) # Calculated from Ryan's transit fit 

airmass_cut = 2
skylev_cut = 700
windowsize = 200 

# Get data
df,bjds,relfluxes,airmasses,widths,flag = ld.return_data_onedate(mainpath,
		                                  targetname,date,threshold, flag_output=True)

# Normalize relative fluxes and shift to zero scale
nrelfluxes = np.copy(relfluxes)/np.median(relfluxes) - 1.0
# Generate mask over which to calculate rms of normalized relfluxes
skylevs = df['Sky/Pixel_T1'].to_numpy()[flag]
rms,minrms,minind = dpt.calc_rms_region(nrelfluxes, airmasses, skylevs, 
					airmass_cut, skylev_cut,windowsize)
#Calculate chisq vs bjd, and plot data+model with min chi sq
chisqs, best_chisq, best_params, best_model = dpt.calc_chisq_vs_bjd(date,
											  bjds,nrelfluxes,airmasses,
											  widths,t0_step,duration,
											  minrms,minind,windowsize,plot=True)



#t0 = 2459709.8829805 #BJD - calculates via calc_transit_times.py



