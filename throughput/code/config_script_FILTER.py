#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 12:53:53 2020

This file exclusively loads data for later use by other scripts in FILTER folder. 

"""

#==============================================================================
# Run this code in Folder: '/Users/jgarciamejia/Documents/2019-TierrasProject/CCD/Code'
#==============================================================================

#==============================================================================
# Load Libraries, Dependencies
#==============================================================================

import numpy as np

import sys
sys.path.insert(1, '/Users/jgarciamejia/Documents/2019:20-TierrasProject/DESIGN_CODE/PWV_code')

import PWV_func_lib as funcs
from jgm_read_FITS import read_fits
from scipy.interpolate import interp1d

#==============================================================================
# User Input: Define/Check the following variables
#==============================================================================

### Define System throughput 
sys_thrupt = 0.58                   # Derived using code Calculate_System_Throughput.py

#==============================================================================
#  Define Constants and Telescope Characteristics
#==============================================================================

alt = 2.3823                            # Observatory altitude, [km]
h = 6.62606885 * 10**(-27)              # Planck, [erg*s]
c = 2.99792458 * 10**(10)               # speed of light [cm/s]
D = 128                                 # effective clear telescope diameter, [cm]
d_hole = 20                             # primary hole diameter, [cm]
A = (np.pi / 4) * (D**2 - d_hole**2)    # effective telescope aperture, [cm^2]

# Conversion factors
AU_to_Rsun = 214.93946938362
pc_to_AU = 2.063 * 10 ** 5
decimal_to_ppm = 10**(6)

#==============================================================================
# Load Data
#==============================================================================

print ('Loading Data...')
bpath = '/Users/jgarciamejia/Documents/TierrasProject_2018-2023/DESIGN_CODE/Data_Files_for_Code/'

# TAPAS telluric spectrum - Units: nm, fraction
h2o = np.loadtxt(bpath+'tapas_600nm_1150nm_H2O.txt', skiprows=21)
h2o_lambda = h2o[:,0] * 10  # convert nm to angstroms
h2o_flux = h2o[:,1]

# Two M-Dwarf Spectra from HST Calspec (Jonathan suggestion)
# M3.5v - Units: ang, erg/s/cm2/ang
Gl555wvs, Gl555flux = read_fits(bpath+'bd11d3759_stis_001.fits') 
dGl555 = 6.25 #pc
# M7v - Units: ang, erg/s/cm2/ang
vb8wvs, vb8flux = read_fits(bpath+'vb8_stiswfcnic_001.fits')
dvb8 = 6.5 #pc
# G2v 
sco18wvs, sco18flux = read_fits(bpath+'18sco_stis_001.fits')
d18sco = 14.13 #pc

# Experimental CCD QE curve 
QEccd = open(bpath+'CCD231-84-x-F64.txt','r').readlines()
QEccd_lam = np.array([]) #nm
QEccd_flx = np.array([]) #units?
for line in QEccd:
    QEccd_lam = np.append(QEccd_lam, float(line.split(', ')[0]))
    QEccd_flx = np.append(QEccd_flx, float(line.split(', ')[1].rstrip()))
QEccd_lam *= 10 #(nm->ang)
QEccd_flx *= 10**(-2)
QEccd_flx = np.clip(QEccd_flx, 0.0,1.0)

# ASAHI Filters

# Experimental Filter Data,  - 5 Locations, LO RES 
path = '/Users/jgarciamejia/Documents/TierrasProject_2018-2023/FILTER/ASAHI/ASAHI_Fab_Test_Data/'
filtdata_lores = np.loadtxt(path+'Tierras_Bandpass_Data_LoRes.txt', delimiter=',')
waves, L1, L2, L3, L4, L5 = filtdata_lores[:,0]*10, filtdata_lores[:,1], filtdata_lores[:,2], filtdata_lores[:,3], filtdata_lores[:,4], filtdata_lores[:,5] # convert nm-> ang

# Experimental Filter Data,  - Center of Filter, HI RES 
waves_hires, L5_hires = np.array([]), np.array([])
filtdata_hires = open(path+'Tierras_Bandpass_Data_HiRes.txt')
for line in filtdata_hires:
    waves_hires = np.append(waves_hires, float(line.split(',')[0]))
    L5_hires = np.append(L5_hires, float(line.split(',')[1].rstrip()))

#==============================================================================
# path = '/Users/jgarciamejia/Documents/2019:20-TierrasProject/FILTER/ASAHI_Quote&Deliverables/'
# #Filter 1
# filtdata1 = np.loadtxt(path+'Quote1/ASAHI_FilterData.txt', delimiter=',')
# waves1, T_AOI0deg1, T_AOI4deg1 = filtdata1[:,0]*10, filtdata1[:,1], filtdata1[:,2] # convert nm->ang
# 
# #Filter 2
# filtdata2 = np.loadtxt(path+'Quote2/ASAHI_Filter2Data.txt', delimiter=',')
# waves2, T_AOI0deg2 = filtdata2[:,0]*10, filtdata2[:,1] # convert nm->ang
#==============================================================================
#==============================================================================
# Interpolate Data onto Comon Wave Grid
#==============================================================================

print ('Interpolating Data...')

# Define common wavelength grid of 44000 data points between 6000-9000 nm 
# Above params selected to match resolution of G. Zhous's Regulus Spec. 
comm_wvs = np.linspace(min(h2o_lambda), 10000, 50000)

# Interpolate ALL data onto common wavelength grid

F_h2o = interp1d(h2o_lambda, h2o_flux, kind = 'linear')
comm_h2o_flux = np.clip(F_h2o(comm_wvs),0.0,1.0)

F_Gl555 = interp1d(Gl555wvs, Gl555flux, kind = 'linear')
comm_Gl555_flux = F_Gl555(comm_wvs)

F_vb8 = interp1d(vb8wvs, vb8flux, kind = 'linear')
comm_vb8_flux = F_vb8(comm_wvs)

F_sco18 = interp1d(sco18wvs, sco18flux, kind = 'linear')
comm_sco18_flux = F_sco18(comm_wvs)

F_ccd = interp1d(QEccd_lam, QEccd_flx, kind='linear')
comm_QEccd_flux = F_ccd(comm_wvs)

F_filterhires = interp1d(waves_hires*10, L5_hires)
comm_filterhires = F_filterhires(comm_wvs)
#==============================================================================
#F_filter1AOI0deg = interp1d(waves1, T_AOI0deg1)
#comm_T1_AOI0deg = F_filter1AOI0deg(comm_wvs)
# 
# F_filter1AOI4deg = interp1d(waves1, T_AOI4deg1)
# comm_T1_AOI4deg = F_filter1AOI4deg(comm_wvs)
# 
# F_filter2AOI0deg = interp1d(waves2, T_AOI0deg2)
# comm_T2_AOI0deg = F_filter2AOI0deg(comm_wvs)
#==============================================================================



