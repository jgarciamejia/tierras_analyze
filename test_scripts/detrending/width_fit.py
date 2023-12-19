import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from medsig import *
from bin_lc import *


# Choose date
date = '20220509'

# Load data
basepath = '/Users/jgarciamejia/Documents/TierrasProject/SCIENCE/'
fullpath = basepath+'AIJ_Output_Ryan_fixedaperture/TOI2013_'+date+'/'
df = pd.read_table(fullpath+'toi2013_'+date+'-Tierras_1m3-I_measurements.xls')
bjds = df['BJD_TDB_MOBS'].to_numpy()
width = df['Width_T1'].to_numpy()
sky = df['Sky/Pixel_T1']
rel_flux_T1 = df['rel_flux_T1'].to_numpy()
rel_flux_T1 /= np.median(rel_flux_T1)

print (len(rel_flux_T1))

# Clip outliers 

# By rel flux
medflux, sigflux = medsig(rel_flux_T1)
thisflag = np.absolute(rel_flux_T1 - medflux) < 3*sigflux
bjds = bjds[thisflag]
width = width[thisflag]
rel_flux_T1 = rel_flux_T1[thisflag]
sky = sky[thisflag]

print (len(rel_flux_T1))

# By sky brightness 
medflux2, sigflux2 = medsig(sky)
flag2 = np.absolute(sky - medflux2) < 3*sigflux2
bjds = bjds[flag2]
width = width[flag2]
rel_flux_T1 = rel_flux_T1[flag2]
sky = sky[flag2]

print (len(rel_flux_T1))

# Fit line and plot data and line
p = np.poly1d(np.polyfit(width,rel_flux_T1,1))
plt.scatter(width, rel_flux_T1)
plt.plot(width,p(width))
plt.show()

# Detrend flux and plot
det_flux_T1 = rel_flux_T1 / p(width)
plt.scatter(width,det_flux_T1)
plt.show()

# Compute rms vs bin size for detrended flux
binsizes = np.arange(1,11) #mins
rmss = np.array([])
for binsize in binsizes:
	xbin,ybin,_ = bin_lc_binsize(bjds,det_flux_T1,binsize)
	rmss = np.append(rmss, np.std(ybin))

# add rms of original data with texp = 30 sec
texp = 0.5 #min
rms_det_flux = np.std(det_flux_T1)
rmss = np.insert(rmss,0,rms_det_flux)
binsizes = np.insert(binsizes,0,texp)

plt.plot(binsizes,rmss*1e3)
plt.show()



