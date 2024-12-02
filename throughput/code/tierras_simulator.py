#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 11:14:04 2019

@author: jgarciamejia

This code generates photometric error (ppm) vs. Integration time 
plots for different stars, given their spectra. Based on 
photerr_vs_inttime_TAPAS_SPIEplot.py. 
"""
import os 
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

import PWV_func_lib as funcs
from jgm_read_FITS import read_fits

def main():
    # Argument parser for command-line inputs
    parser = argparse.ArgumentParser(description="Generate photometric error vs. integration time plots for stellar spectra.")
    parser.add_argument("-fits_file", type=str, help="Path to the stellar spectrum FITS file.")
    parser.add_argument("-distance", type=float, help="Distance to place the star at (in pc).")
    args = parser.parse_args()

    # Extract arguments
    fits_file = args.fits_file
    distance_pc = args.distance

    ### LOAD DATA 

    # bpath = '/Users/jgarciamejia/Documents/TierrasProject_2024+/THROUGHPUT/input/'
    # Get the absolute path of the script's directory
    script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))

    # Compute the parent directory and define the data path
    parent_directory = os.path.abspath(os.path.join(script_dir, os.pardir))
    bpath = os.path.join(parent_directory, "input/")

    # Ensure bpath exists
    if not os.path.exists(bpath):
        raise FileNotFoundError(f"The expected data directory does not exist: {bpath}")
                            
    # TAPAS telluric spectrum - Units: nm, fraction
    h2o = np.loadtxt(bpath + 'tapas_600nm_1150nm_H2O.txt', skiprows=21)
    h2o_lambda = h2o[:, 0] * 10  # convert nm to angstroms
    h2o_flux = h2o[:, 1]

    # Tierras CCD QE curve 
    QEccd = open(bpath + 'CCD231-84-x-F64.txt', 'r').readlines()
    QEccd_lam = np.array([]) #nm
    QEccd_flx = np.array([]) #units?
    for line in QEccd:
        QEccd_lam = np.append(QEccd_lam, float(line.split(', ')[0]))
        QEccd_flx = np.append(QEccd_flx, float(line.split(', ')[1].rstrip()))
    QEccd_lam *= 10  # (nm->ang)
    QEccd_flx *= 10 ** (-2)
    QEccd_flx = np.clip(QEccd_flx, 0.0, 1.0)

    # Tierras Filter Curve
    filtdata_lores = open(bpath + 'Tierras_Bandpass_Data_LoRes.txt')
    waves_lores, L5_lores = np.array([]), np.array([])
    for line in filtdata_lores:
        waves_lores = np.append(waves_lores, float(line.split(',')[0]))
        L5_lores = np.append(L5_lores, float(line.split(',')[1].rstrip()))

    # Define System throughput 
    sys_thrupt = 0.58  # Derived using code Calculate_System_Throughput.py

    # Stellar Spectra from HST Calspec in units of: ang, erg/s/cm2/ang
    # User-provided stellar spectrum
    user_wvs, user_flux = read_fits(fits_file)

    # G2v 
    sco18wvs, sco18flux = read_fits(bpath + '18sco_stis_001.fits')

    # Interpolate Data onto Common Wave Grid
    comm_wvs = np.linspace(min(h2o_lambda), 10000, 50000)

    F_h2o = interp1d(h2o_lambda, h2o_flux, kind='linear')
    comm_h2o_flux = np.clip(F_h2o(comm_wvs), 0.0, 1.0)

    F_user = interp1d(user_wvs, user_flux, kind='linear')
    comm_user_flux = F_user(comm_wvs)

    F_sco18 = interp1d(sco18wvs, sco18flux, kind='linear')
    comm_sco18_flux = F_sco18(comm_wvs)

    F_ccd = interp1d(QEccd_lam, QEccd_flx, kind='linear')
    comm_QEccd_flux = F_ccd(comm_wvs)

    F_filterlores = interp1d(waves_lores * 10, L5_lores)
    comm_TloresL5 = F_filterlores(comm_wvs)
    comm_TloresL5 /= 100  # ASAHI data in %

    # Telescope Characteristics
    alt = 2.3823
    h = 6.62606885 * 10 ** (-27)
    c = 2.99792458 * 10 ** (10)
    D = 128
    d_hole = 20
    A = (np.pi / 4) * (D ** 2 - d_hole ** 2)

    # Conversion factors
    pc_to_AU = 2.063 * 10 ** 5
    decimal_to_ppm = 10 ** 6

    # User-provided star flux adjusted for distance
    star_flux_adjusted = comm_user_flux * (15 / distance_pc) ** 2

    # Define integration times to test
    int_times = np.arange(5, 125, 5)

    scint_errs, phot_errs, phot_counts = [], [], []

    for t in int_times:
        scint_errs.append(funcs.scintillation1p75(1.5, D, t, alt) * decimal_to_ppm)
        phot_count, phot_noise = funcs.calc_precision_nodilution(
            comm_wvs, comm_h2o_flux, star_flux_adjusted,
            comm_QEccd_flux, comm_TloresL5, A, t, sys_thrupt)
        phot_counts.append(phot_count)
        phot_errs.append(phot_noise * decimal_to_ppm)

    gain = 5.9
    ADUs = np.array(phot_counts) / gain

    # Create and print DataFrame
    df = pd.DataFrame({
        'Integration Time (s)': int_times,
        'Photon Counts': phot_counts,
        'ADUs': ADUs
    })

    print(df)

if __name__ == "__main__":
    main()







