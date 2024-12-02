#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 21:51:05 2017

@author: jgarciamejia
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 20:46:24 2017

@author: jgarciamejia
"""
print ('Loaded PWV Library of Functions')

from scipy.integrate import trapezoid
import numpy as np

# Conversion factors
AU_to_Rsun = 214.93946938362
pc_to_AU = 2.063 * 10 ** 5

# The following is a library of functions to be used in precipital water vapor studies, 
# and studies to determine the ideal bandpass for a scintillation-limited observation. 

def scale_to_PWV(h2o_flux, pwv_target, pwv_spec): 
    """This function takes a telluric (H2O-only) spectrum, and scales it to the desired PWV amount.
    The wavelength grid is preserved.
    Inputs:
        - h20_flux: flux values for a water-only telluric spectrum from TAPAS. (Numpy array)
        - pwv_spec: amount of precipitable water vapour(in mm) in the input spectra. In TAPAS files it should appear under H20cv
        - pwv_target: amount of pwv (in mm) that one wants to scale the output spec to be at.
    Outputs: H2O spectrum with target precipitable water vapour.
        """
    return h2o_flux ** (pwv_target/pwv_spec)
    
def star_flux_at_Earth(lambda_grid, telluric, stellar_flux, ccd_response, bandpass, R_star, d_star):
    """
    The following function calculates the stellar flux received at Earth for a given star.
    NOTE: This code assumes that all flux/response arrays are interpolated onto the same
    (wavelength array) lambda_grid.
    Inputs:
        - lambda_grid: wavelength grid shared by telluric and stellar flux, as well as CCD and bpass. 
        Take care that grid is in units that agree with stellar_flux array.
        - Telluric: Terrestrial atmospheric transmission model. Range: [0,1].
        - ccd_response: QE curve of the CCD. Range: [0,1]
        - bandpass: filter array to integrate over (or to bound integration). Range: [0,1]
        - R_star: radius of the star, in Rsun
        - d_star: distance to star, in pc
    Outputs:
        - meas_flux: flux of the star measured through the system at Earth
    """
    # define constants
    AU_to_Rsun = 214.93946938362
    pc_to_AU = 2.063 * 10 ** 5
    dilution = (R_star / (d_star * pc_to_AU * AU_to_Rsun)) ** 2   ## can be improved
    
    # Calculate stellar flux through system
    sys_response = ccd_response * telluric * bandpass
    meas_flux = trapezoid(sys_response * stellar_flux * dilution, lambda_grid) 

    return (meas_flux)

def integrated_flux(lambda_grid, telluric, stellar_flux, ccd_response, bandpass, sys_thruput):
    """
    The following function calculates the integrated flux for a given star through a given filter.
    It assumes that stellar spectrum inputted is measured ON EARTH, NOT SURFACE OF STAR.
    NOTE: This code assumes that all flux/response arrays are interpolated onto the same
    (wavelength array) lambda_grid.
    Inputs:
        - lambda_grid: wavelength grid shared by telluric and stellar flux, as well as CCD and bpass. 
        Take care that grid is in Angstroms
        - Telluric: Terrestrial atmospheric transmission model. Range: [0,1].
        - Stellar flux as measured at Earth. Units: erg/s/cm^2/Ang
        - ccd_response: QE curve of the CCD. Range: [0,1]
        - bandpass: filter array to integrate over (or to bound integration). Range: [0,1]
        - sys_thruput: fraction of light that makes it thru system due to optical reflections. Range: [0,1].
        If unknown, simply write 1
    Outputs:
        - meas_flux: flux of the star measured through the system at Earth
    """
    # Calculate stellar flux through system
    sys_response = ccd_response * telluric * bandpass * sys_thruput
    meas_flux = trapezoid(sys_response * stellar_flux, lambda_grid) 
    #print ('new integrated_flux function was used')
    #print ('Measured flux(erg/s/cm^2) = '+str(meas_flux))
    return meas_flux

def calc_color_err(lambda_grid, telluric, target_flux, compar_flux, ccd_response, bandpass, 
                   aperture, exp_time, target_R, compar_R, d_target, d_comparison, minPWV, maxPWV):
    """Function to quantify PWV error, which is the absolute change in the 
    magnitude difference between two stars measured when the column water 
    vapor changes from maxPV to minPV. m1=m star, m2=gstar 
    Inputs:
        - lambda_grid: wavelength grid shared by telluric and stellar flux, as well as CCD. In Angstroms (SERIOUSLY!).
        - Telluric: Atmospheric transmission model. Range: [0,1]
        - target_flux: flux of the target star at its SURFACE, in units of ergs/cm^2/s/Ang
        - comparison_flux = flux of the comparison star at its SURFACE, in units of ergs/cm^2/s/Ang
        - ccd_response: QE curve of the CCD. Available curves: deep dep., front ill., bac ill. Range: [0,1]
        - bandpass: filter to integrate over (or to bound integration). Range: [0,1]
        - aperture: effective area of the telescope primary. Units: cm^2
        - exp_time: integration/ exposure time. Units: seconds
        - target_R: radius of the target star, in Rsun
        - comparison_R: radius of the comparison star, in Rsun
        - d_target: distance to star, in pc
        - d_comparison: distance to comparison star
        - minPV, maxPV: minimum and maximum PWV values, in mm """
    # generate filter and define PWVs

    PWVs = [minPWV, maxPWV]
    
    # calculate |(m1-m2)@PWV=12 - (m1-m2)@PWV=0|
    m1_m2_diff = 0
    for PWV_val in PWVs:
        scaled_telluric = scale_to_PWV(telluric, PWV_val, 9.592)
        mstar_flux = star_flux_at_Earth(lambda_grid, scaled_telluric, target_flux, ccd_response, bandpass, target_R, d_target) 
        gstar_flux = star_flux_at_Earth(lambda_grid, scaled_telluric, compar_flux, ccd_response, bandpass, compar_R, d_comparison)
        #print (PWV_val, 2.5*np.log10(mstar_flux/gstar_flux))
        if PWV_val == maxPWV:
            m1_m2_diff += -2.5 * np.log10(mstar_flux / gstar_flux)
        elif PWV_val == minPWV:
            m1_m2_diff += 2.5 * np.log10(mstar_flux / gstar_flux)
            
    return np.abs(m1_m2_diff)
    
    # calculate |(m1-m2)@PWV=12 - (m1-m2)@PWV=0|
    #m1_m2_diff = 0
    for PWV_val in PWVs:
        scaled_telluric = scale_to_PWV(telluric, PWV_val, 9.592)
        mstar_flux = star_flux_at_Earth(lambda_grid, scaled_telluric, target_flux, ccd_response, bandpass, target_R, d_target) 
        gstar_flux = star_flux_at_Earth(lambda_grid, scaled_telluric, compar_flux, ccd_response, bandpass, compar_R, d_comparison)
        #print (PWV_val, 2.5*np.log10(mstar_flux/gstar_flux))
        #if PWV_val == maxPWV:
        #    m1_m2_diff += -2.5 * np.log10(mstar_flux / gstar_flux)
        #elif PWV_val == minPWV:
        #    m1_m2_diff += 2.5 * np.log10(mstar_flux / gstar_flux)
            
    return (mstar_flux, gstar_flux)


def calc_color_err_nodilution(lambda_grid, telluric, target_flux, compar_flux, ccd_response, bandpass, 
                   aperture, exp_time, telluricPWV, minPWV, maxPWV, sys_thruput):
    """Function to quantify PWV error assuming one has real stellar spectra, 
    where PWV is the absolute change in the magnitude difference between two 
    stars measured when the column water vapor changes from maxPV to minPV. m1=m star, m2=gstar
    Inputs:
        - lambda_grid: wavelength grid shared by telluric and stellar flux, as well as CCD. In Angstroms (SERIOUSLY!).
        - Telluric: Atmospheric transmission model. Range: [0,1]
        - target_flux: flux of the target star at THE EARTH, in units of ergs/cm^2/s/Ang
        - comparison_flux = flux of the comparison star at THE EARTH, in units of ergs/cm^2/s/Ang
        - ccd_response: QE curve of the CCD. Available curves: deep dep., front ill., bac ill. Range: [0,1]
        - bandpass: filter to integrate over (or to bound integration). Range: [0,1]
        - aperture: effective area of the telescope primary. Units: cm^2
        - exp_time: integration/ exposure time. Units: seconds
        - target_R: radius of the target star, in Rsun
        - comparison_R: radius of the comparison star, in Rsun
        - d_target: distance to star, in pc
        - d_comparison: distance to comparison star
        - telluricOWV: water column value of the input telluric spectrum, in mm
        - minPV, maxPV: minimum and maximum PWV values, in mm 
        - sys_thruput: fraction of light that makes it thru system due to optical reflections. Range: [0,1].
        If unknown, simply write 1.0. """
    # generate filter and define PWVs

    PWVs = [minPWV, maxPWV]
    
    # calculate |(m1-m2)@PWV=12 - (m1-m2)@PWV=0|
    m1_m2_diff = 0
    for PWV_val in PWVs:
        #print (PWV_val)
        scaled_telluric = scale_to_PWV(telluric, PWV_val, telluricPWV)
        #print ('M star')
        target_int_flux = integrated_flux(lambda_grid, scaled_telluric, target_flux, ccd_response, bandpass, sys_thruput)
        #print ('G star')
        comparison_int_flux = integrated_flux(lambda_grid, scaled_telluric, compar_flux, ccd_response, bandpass, sys_thruput)
        if PWV_val == maxPWV:
            m1_m2_diff += -2.5 * np.log10(target_int_flux / comparison_int_flux)
        elif PWV_val == minPWV:
            m1_m2_diff += 2.5 * np.log10(target_int_flux / comparison_int_flux)
    #print ('new color error function was used')       
    return np.abs(m1_m2_diff)

##print ('Dec 13, 2020 - Added calc_color_err_nodilution_fluxnotmag to fix ppm vs mag error! ')
def calc_color_err_nodilution_fluxnotmag(lambda_grid, telluric, target_flux, compar_flux, ccd_response, bandpass, 
                   aperture, exp_time, telluricPWV, minPWV, maxPWV, sys_thruput):
    """Function to quantify PWV error assuming one has real stellar spectra, 
    where PWV is the absolute change in the magnitude difference between two 
    stars measured when the column water vapor changes from maxPV to minPV. m1=m star, m2=gstar
    Inputs:
        - lambda_grid: wavelength grid shared by telluric and stellar flux, as well as CCD. In Angstroms (SERIOUSLY!).
        - Telluric: Atmospheric transmission model. Range: [0,1]
        - target_flux: flux of the target star at THE EARTH, in units of ergs/cm^2/s/Ang
        - comparison_flux = flux of the comparison star at THE EARTH, in units of ergs/cm^2/s/Ang
        - ccd_response: QE curve of the CCD. Available curves: deep dep., front ill., bac ill. Range: [0,1]
        - bandpass: filter to integrate over (or to bound integration). Range: [0,1]
        - aperture: effective area of the telescope primary. Units: cm^2
        - exp_time: integration/ exposure time. Units: seconds
        - target_R: radius of the target star, in Rsun
        - comparison_R: radius of the comparison star, in Rsun
        - d_target: distance to star, in pc
        - d_comparison: distance to comparison star
        - telluricOWV: water column value of the input telluric spectrum, in mm
        - minPV, maxPV: minimum and maximum PWV values, in mm 
        - sys_thruput: fraction of light that makes it thru system due to optical reflections. Range: [0,1].
        If unknown, simply write 1.0. """
    # generate filter and define PWVs

    PWVs = [minPWV, maxPWV]
    
    # calculate |(m1-m2)@PWV=12 - (m1-m2)@PWV=0|
    #m1_m2_diff = 0
    fluxes = np.array([])
    for PWV_val in PWVs:
        scaled_telluric = scale_to_PWV(telluric, PWV_val, telluricPWV)
        target_int_flux = integrated_flux(lambda_grid, scaled_telluric, target_flux, ccd_response, bandpass, sys_thruput)
        fluxes = np.append(fluxes, target_int_flux)
        comparison_int_flux = integrated_flux(lambda_grid, scaled_telluric, compar_flux, ccd_response, bandpass, sys_thruput)
        fluxes = np.append(fluxes, comparison_int_flux)
    return (np.abs(1-(fluxes[0]/fluxes[1])/(fluxes[2]/fluxes[3])))
        #if PWV_val == maxPWV:
        #    m1_m2_diff += -2.5 * np.log10(target_int_flux / comparison_int_flux)
        #elif PWV_val == minPWV:
        #    m1_m2_diff += 2.5 * np.log10(target_int_flux / comparison_int_flux)
    #print ('new color error function was used')       
    #return np.abs(m1_m2_diff)
    #return (fluxes)


def calc_precision(lambda_grid, telluric, stellar_flux,ccd_response, bandpass,aperture, exp_time, r_star, d_star):
    """The following function calculates the achievable photometric precision of a ground-based
    telescope (with a given aperture, CCD, bandpass and losses) when measuring differential flux
    of a given star. The precision is obtained from the number of photons. 
    for a certain type of star. 
    NOTE: dilution factor included because function assumes stellar flux inputted is at surface. 
    Inputs:
        - lambda_grid: wavelength grid shared by telluric and stellar flux, as well as CCD. In Angstroms (SERIOUSLY!).
        - Telluric: Atmospheric transmission model. Range: [0,1]
        - stellar_flux: flux of the star at the SURFACE, in units of ergs/cm^2/s/Ang
        - ccd_response: QE curve of the CCD. Available curves: deep dep., front ill., bac ill. Range: [0,1]
        - bandpass: filter to integrate over (or to bound integration). Range: [0,1]
        - aperture: effective area of the telescope primary. Units: cm^2
        - exp_time: integration/ exposure time. Units: seconds.
        - r_star: radius of the star, in Rsun
        - d_star: distance to star, in pc
    Outputs:
        - Tuple (nphotons, precision): 
        [0] number of photons 
        [2] achievable photometric precision, as given by sqrt(N)/N, where N = number of photons hitting
        primary in a unit of time exp_time
    NOTE: This code assumes that all fluxes are interpolated onto the same wavelength array. 
        """
    # define constants
    AU_to_Rsun = 214.93946938362
    pc_to_AU = 2.063 * 10 ** 5
    dilution = (r_star / (d_star * pc_to_AU * AU_to_Rsun)) ** 2
    flux_to_phot = ( dilution * stellar_flux ) * 5.03 * 10**7 * lambda_grid  # using Ashley Baker's conversion suggestion
    #norm_f_lambda = flux_to_phot / max(flux_to_phot)

    #System response
    sys_response = ccd_response * telluric * bandpass
    # Integral for photons/area/time
    phot_per_ar_per_t = trapezoid(sys_response * flux_to_phot, lambda_grid) 
    # Number of photons
    n_photons = phot_per_ar_per_t * aperture * exp_time
    # Precision
    return (n_photons, 1/np.sqrt(n_photons))

def calc_precision_nodilution(lambda_grid, telluric, stellar_flux,ccd_response, bandpass,aperture, exp_time, sys_thruput):
    """The following function calculates the achievable photometric precision of a ground-based
    telescope (with a given aperture, CCD, bandpass and losses) when measuring differential flux
    of a given star. The precision is obtained from the number of photons. 
    for a certain type of star. 
    NOTE: dilution factor NOT included because function assumes stellar flux is measured at Earth,
    not stellar surface. This function is appropriate when stellar spectra are experimental and NOT
    synthetic models. 
    Inputs:
        - lambda_grid: wavelength grid shared by telluric and stellar flux, as well as CCD and filter. In Angstroms (SERIOUSLY!).
        - Telluric: Atmospheric transmission model. Range: [0,1]
        - stellar_flux: flux of the star at the EARTH, in units of ergs/cm^2/s/Ang
        - ccd_response: Empirical QE curve of the CCD. Range: [0,1]
        - bandpass: filter to integrate over (or to bound integration). Range: [0,1]
        - aperture: effective area of the telescope primary. Units: cm^2
        - exp_time: integration/ exposure time. Units: seconds.
        - r_star: radius of the star, in Rsun
        - d_star: distance to star, in pc
        - sys_thruput: fraction of light that makes it thru system due to optical reflections. Range: [0,1].
        If unknown, simply write 1
    Outputs:
        - Tuple (nphotons, precision): 
        [0] number of photons 
        [2] achievable photometric precision, as given by sqrt(N)/N, where N = number of photons hitting
        primary in a unit of time exp_time
    NOTE: This code assumes that all fluxes are interpolated onto the same wavelength array. 
        """
    # convert flux from ergs to photons
    one_hc = 5.034 * 10**7    # = 1/ hc 
    flux_to_phot = stellar_flux * one_hc * lambda_grid  # ergs/cm^2/s/Ang -> photons/cm^2/s/Ang
    #System response
    sys_response = ccd_response * telluric * bandpass * sys_thruput
    # Integral for electrons/area/time (QE curve gives you photons -> electrons conversion)
    phot_per_ar_per_t = trapezoid(sys_response * flux_to_phot, lambda_grid) 
    # Number of electrons (or source photons you actually measure)
    n_photons = phot_per_ar_per_t * aperture * exp_time
    # Return No. of Photons collected AND Precision
    #print ('New calc_precision_nodilution function in use!')
    return (n_photons, 1/np.sqrt(n_photons))


def calc_delf_overf(ref_photons_star1, ref_photons_star2, n_photons_star1, n_photons_star2):
    """The following function calculates the flux difference (\deltaf / f) between two stars 
    at two different airmasses. 
    Inputs:
    - ref_photons_starX: amount of photons calculated for the xth star at the reference PWV values (1 mm in our case).
    - n_photons_starX: amount of photons calculated for the xth star using calc_precision[0]
    Outputs:
    - \deltaf / f for a given PWV. The underlying assumption is that this is for one bandpass only."""
    ref_ratio = ref_photons_star1 / ref_photons_star2
    comp_ratio = n_photons_star1 / n_photons_star2
    return ( (ref_ratio) - (comp_ratio) ) / (ref_ratio)

def generate_filter(lambda_grid, lambda_0, width):
    """ Generates a filter of a given width, with its left corner at a given wavelength. 
    The top of the filter may be flat, or modelled as a sinusoid (future upgrade). 
    The filter is interpolated into the given wavelength grid. 
    Note: if filter does not fit into the given wavelength grid, function will throw an IndexError
    UNITS: nanometers nm.
    
    Inputs:
        - lambda_grid: wavelength grid shared by telluric (flux) and stellar flux. In nm. 
        - lambda_0: wavelength value user wants at left-most corner of filter. In nm.
        - width: define how wide you want the filter to be, in nm. Width will span to the left.
        User need not worry about checking wavelength grid.
    Outputs:
        - filter array fulfilling user specs as closely as possible, and interpolated into the lambda_grid.
    """
    # check that given left of filter is within lambda_grid:
    if lambda_0 < min(lambda_grid) or lambda_0 > max(lambda_grid):
        print ("The desired filter's left corner is outside the given wavelength grid in wavelength space.")
        return None
    else:
    # find the index value closest to lambda_0
        for index in range(len(lambda_grid)): # range started at index=90000 to make code more efficient
            if lambda_0 >= lambda_grid[index] and lambda_0 <= lambda_grid[index+1]:
                left_index = index
        # approximate \delta \lambda as average for entire wavelength grid (change Jan23)
        delta_lambda = np.mean(lambda_grid[1:] - lambda_grid[0:-1])
        # get right index
        right_index = left_index + int(round(width / delta_lambda))       
        # estimate final width
        final_width = lambda_grid[right_index] - lambda_grid[left_index]
        #print (" Your filter width will be "+str(final_width)+" nm")
        
        filter_array = np.array([])
        for index in range(len(lambda_grid)): 
            if index >= left_index and index <= right_index:
                filter_array = np.append(filter_array, 1.0)
            elif index <= left_index or index >= right_index:
                filter_array = np.append(filter_array, 0.0)
            else:
                print ("Go check your generate_filter function because it ain't doing what you want.")
        
        return filter_array, lambda_grid[left_index], lambda_grid[right_index], final_width
        # return filter_array # this is if you only want the filter array

def generate_filter_2(lambda_grid, lambda_0, width, width_err):
    """ Generates a filter of a given width, with its left corner at a given wavelength. 
    The top of the filter may be flat, or modelled as a sinusoid (future upgrade). 
    The filter is interpolated into the given wavelength grid. 
    Note: if filter does not fit into the given wavelength grid, function will throw an IndexError
    UNITS: nanometers nm.
    
    Inputs:
        - lambda_grid: wavelength numpy array shared by telluric (flux) and stellar flux. 
        - lambda_0: wavelength value user wants at left-most corner of filter.
        - width: define how wide you want the filter to be. Width will span to the left.
        - widtherr: user-defined maximum allowable absolute difference between the desired and actual filter width 
        Note: all of the above should be in the same units.
    Outputs:
        - filter array fulfilling user specs as closely as possible, and interpolated into the lambda_grid.
    """
    # check that given left of filter is within lambda_grid:
    if lambda_0 < min(lambda_grid) or lambda_0 > max(lambda_grid):
        print ("The desired filter's left corner is outside the given wavelength grid in wavelength space.")
        return None
    else:
        # determine left and right indices in wave grid
        left_index = np.argmin(np.abs(lambda_grid - lambda_0))
        right_index = np.argmin(np.abs(lambda_grid - (lambda_0+width)))
        # estimate final width
        final_width = lambda_grid[right_index] - lambda_grid[left_index]
        # Make sure filter width is within alloeable error
        if np.abs(final_width - width) <= width_err:
            # generate filter array
            mask = [ (el >= lambda_grid[left_index]) and (el <= lambda_grid[right_index]) for el in lambda_grid]
            filter_array = np.array([1.0 if i == True else 0.0 for i in mask])
            #print ("The desired filter width was achieved. Desired={}, Actual={}. Left Index = {}".format(width, final_width, left_index))
            return filter_array, lambda_grid[left_index], lambda_grid[right_index], final_width
        else: 
            # generate filter array even when final width is larger than allowable width error 
            # BUT print a warning 
            mask = [ (el >= lambda_grid[left_index]) and (el <= lambda_grid[right_index]) for el in lambda_grid]
            filter_array = np.array([1.0 if i == True else 0.0 for i in mask])
            # 
            print ("The desired filter width was not achieved. Desired={}, Actual={}. Left Wave = {}".format(width, final_width, lambda_grid[left_index]))
            return filter_array, lambda_grid[left_index], lambda_grid[right_index], final_width

            #return None

def generate_filter_fromc(lambda_grid, lambda_0, width):
    """ Generates a filter of a given width, with its center at lambda_0. 
    The top of the filter may be flat, or modelled as a sinusoid (future upgrade). 
    The filter is interpolated into the given wavelength grid. 
    Inputs:
        - lambda_grid: wavelength grid shared by telluric (flux) and stellar flux. In Angstroms. 
        - lambda_0: wavelength value user wants at center of filter. In Angstroms
        - width: define how wide you want the filter to be, in Angstroms. Width will span to both sides.
        User need not worry about checking wavelength grid.
    Outputs:
        - filter array fulfilling user specs as closely as possible, and interpolated into the lambda_grid.
    """
    # find the index value closest to lambda_0
    for index in range(len(lambda_grid)): # range started at index=90000 to make code more efficient
        if lambda_0 >= lambda_grid[index] and lambda_0 <= lambda_grid[index+1]:
            center_index = index
    # approximate \delta \lambda of wavelength grid near the left_index (really, could be anywhere)
    delta_lambda = lambda_grid[center_index] - lambda_grid[center_index - 1]
    # get right and left index
    left_index = center_index - int(round(width * (1.0/2) / delta_lambda))
    right_index = center_index + int(round(width * (1.0/2) / delta_lambda))

    filter_array = np.array([])
    for index in range(len(lambda_grid)):
        if index >= left_index and index <= right_index:
            filter_array = np.append(filter_array, 1.0)
        else:
            filter_array = np.append(filter_array, 0.0)

    return filter_array, left_index, right_index

def find_best_bandpass_simple(lambda_grid, tel_flx, bandwidth, n):
    """Simpler version of function find_best_bandpass. Allows user to find a bandpass 
    of desired bandwidth with the least telluric features throughout lambda grid. 
    This function will explore all of lambda grid. The more complex find_best_bandpass
    would be better for when you want to specify a range of wavelength to explore.
    As of Jan25/2019 that function is still buggy and needs some work. 
    Note: here bandpass and filter used interchangeably.
    Inputs:
        (0) lambda_grid: wavelength grid to search for best filter. 
        Units: nm prefereably but not necessarily. Just be consistent across the code. 
        (1) tel_flx: telluric flux grid. Must be already interpolated onto lambda_grid. 
        Preferable to have flux grid normalized, but doesn't have to be. 
        (2) bandwidth: width of bandpass filter. In units of (0).
        (3) n is the step size between left-most wavelengths to try, in index units.
    Returns: 
        Current: Numpy Array describing the filter transmission in wavelength grid lambda_grid.
        Could Modify to return: List with left-most bandpass wavelength, right_most bandpass wavelength, and bp width."""
    
    
    # find approx number of indices desired bandwidth spans 
    delta_lambda = np.mean(lambda_grid[1:] - lambda_grid[0:-1]) # get average delta_lambda across grid
    nindices = int(bandwidth/delta_lambda) 
    #print (nindices)
    
    # if desired filter width does not fit in wavelength chunk, issue warning
    if nindices >= len(lambda_grid):
        print ("The bandpass you specified is larger than the chunk of wavelength space you are exploring.")
        best_filter = None
        
    # if desired filter width does fit in wavelength chunk, continue to find it
    else:
        # define largest sum and best filter vars
        largest_sum = 0
        best_filter = np.array([])
        # sweep across the provided wavelength grid testing each wavelength as a left-most wavelength
        #for potential_lam_0 in lambda_grid:
        for ind in np.arange(0,len(lambda_grid),n):
            # if the filter is fully inside the wavelength grid 
            # (instead of chopping off at the right side of the grid)
            if lambda_grid[ind] + bandwidth < lambda_grid[-1]:
                # generate a filter candidate
                pot_filt, pot_leftl, pot_rightl, pot_width = generate_filter(lambda_grid, lambda_grid[ind], bandwidth)
                # multiply filter candidate by flux array and sum result. 
                # Since filter arr is simply zeros and ones, all that is not releavant gets multiplied by 0 
                # and the relevant stuff where the filter is transmissive
                # gets multiplied by one and then summed to get sense of total flux through filter.
                this_sum = np.sum(pot_filt*tel_flx)
                if this_sum > largest_sum:
                    largest_sum = this_sum
                    best_filter = pot_filt
                    best_left = pot_leftl
                    best_right = pot_rightl
                    best_width = pot_width
        return [best_filter, best_left, best_right, best_width]


def find_best_bandpass(lambda_grid, tel_flx, bandwidth, n, left_lam, right_lam):
    """ Allows user to find a bandpassof desired bandwidth with the least telluric 
    features between left_lam and right_lam (that is, a wavelength chunk within the wavelength grid)
    This function will explore all of lambda grid.
    Note: here bandpass and filter used interchangeably.
    Inputs:
        (0) lambda_grid: wavelength grid to search for best filter. 
        Units: nm prefereably but not necessarily. Just be consistent across the code. 
        (1) tel_flx: telluric flux grid. Must be already interpolated onto lambda_grid. 
        Preferable to have flux grid normalized, but doesn't have to be. 
        (2) bandwidth: width of bandpass filter. In units of (0).
        (3) n is the step size between left-most wavelengths to try, in index units.
        (4) left_lam, right_lam: right_most and left-most wavelengths of wavelength "chunk"
        to explore and find best filter. 
    Returns: 
        Current: Numpy Array describing the filter transmission in wavelength grid lambda_grid.
        Could Modify to return: List with left-most bandpass wavelength, right_most bandpass wavelength, and bp width."""
    
    
    # find approx number of indices desired bandwidth spans 
    delta_lambda = np.mean(lambda_grid[1:] - lambda_grid[0:-1]) # get average delta_lambda across grid
    nindices = int(bandwidth/delta_lambda) 
    #print (nindices)
    
    # if desired filter width does not fit in wavelength chunk, issue warning
    if nindices >= len(lambda_grid):
        print ("The bandpass you specified is larger than the chunk of wavelength space you are exploring.")
        return None
        
    # check that wavelength chunk of exploration within wavelength grid
    if left_lam < min(lambda_grid) or right_lam> max(lambda_grid):
        print ("Either the left or right wavelengths provided to describe the chunk of exploration are outside the wavelength grid range")
        return None
    
    # if desired filter width does fit in wavelength chunk, continue to find it
    else:
        # define largest sum and best filter vars
        largest_sum = 0
        best_filter = np.array([])
        # sweep across the provided wavelength grid testing each wavelength as a left-most wavelength
        #for potential_lam_0 in lambda_grid:
        for ind in np.arange(0,len(lambda_grid),n):
            # if the filter is fully inside the wavelength grid 
            # (instead of chopping off at the right side of the grid)
            if lambda_grid[ind] + bandwidth < lambda_grid[-1]:
                # if index is within desired wavelength chunk of exploration
                if lambda_grid[ind] >= left_lam and lambda_grid[ind] <= right_lam:
                # generate a filter candidate
                    pot_filt, pot_leftl, pot_rightl, pot_width = generate_filter(lambda_grid, lambda_grid[ind], bandwidth)
                    # multiply filter candidate by flux array and sum result. 
                    # Since filter arr is simply zeros and ones, all that is not releavant gets multiplied by 0 
                    # and the relevant stuff where the filter is transmissive
                    # gets multiplied by one and then summed to get sense of total flux through filter.
                    this_sum = np.sum(pot_filt*tel_flx)
                    if this_sum > largest_sum:
                        largest_sum = this_sum
                        best_filter = pot_filt
                        best_left = pot_leftl
                        best_right = pot_rightl
                        best_width = pot_width
        return [best_filter, best_left, best_right, best_width]

import math
from scipy.interpolate import interp1d

def round_up_to_even(f):
    return math.ceil(f / 2.) * 2
#print ('Feb 10/2020 - Added generate_sloped_filter function.')
def generate_sloped_filter(lambda_grid, ctrwv, width, widtherr, slope, peakT):
    """ 
    Description: Generates a filter with sloped sides. Note: you must be consistent with
    the units of ALL your inputs. 
    Inputs: 
    lambda_grid - wavelength grid to interpolated sloped filter onto. 
    ctrwv - user-defined central wavelength of the desired filter. 
    width - user-defined bandwidth of the desired filter. 
    widtherr - allowable difference between user-defined width and actual filter width. 
    slope - the slope allowed on each side of the filter, from 0% to 100% transmission.
    peakT - the peak transmission of the filter. 
    Outputs:
    filt_array - array of transmission values for a sloped filter interpolated onto lambda_grid. 
    true_width - final width of the filter. 
    """
    # Define left and right waves at 50% peak transmission
    leftwv_50 = ctrwv - (width/2)
    rightwv_50 = ctrwv + (width/2)
    # Define left and right waves at 100% peak transmission
    leftwv_100 = leftwv_50 + slope/2
    rightwv_100 = rightwv_50 - slope/2
    # Define left and right waves at 0% peak transmission
    leftwv_0 = leftwv_50 - slope/2
    rightwv_0 = rightwv_50 + slope/2
    # Define left and right waves at extremes of lambda grid
    minwv = lambda_grid[0]
    maxwv = lambda_grid[-1]
    # Make array of waves with above
    waves = np.array([minwv, leftwv_0, leftwv_50, leftwv_100, rightwv_100, rightwv_50, rightwv_0, maxwv])
    transmsns = np.array([0, 0, 0.5, 1.0, 1.0, 0.5, 0, 0])*peakT
    # Interpolate onto desired wavelength grid scale
    F = interp1d(waves, transmsns)
    filt_array = F(lambda_grid)
    # Check true width of filter from leftwv_50 to rightwv_50
    left_index = np.argmin(np.abs(lambda_grid - leftwv_50))
    right_index = np.argmin(np.abs(lambda_grid - rightwv_50))
    true_width = lambda_grid[right_index] - lambda_grid[left_index]
    if np.abs(true_width - width) >= widtherr:
        print ("The true width of the filter is different from the user-defined width by more than the user-defined width error.")
    return filt_array, true_width, leftwv_50, rightwv_50, leftwv_100, rightwv_100, leftwv_0, rightwv_0, left_index, right_index



#print ('Feb 14/2020 - Added generate_sloped_filter2 function.')
def generate_sloped_filter2(lambda_grid, ctrwv, width, widtherr, slope, peakT, OBbuffer):
    """ 
    Description: Generates a filter with sloped sides. Note: you must be consistent with
    the units of ALL your inputs. 
    Inputs: 
    lambda_grid - wavelength grid to interpolated sloped filter onto. 
    ctrwv - user-defined central wavelength of the desired filter. 
    width - user-defined bandwidth of the desired filter. 
    widtherr - allowable difference between user-defined width and actual filter width. 
    slope - the slope allowed on each side of the filter, from 0% to 100% transmission.
    peakT - the peak transmission of the filter. 
    OBbuffer - out of band buffer. Specifies how far to place out of band from filter.
    Outputs:
    filt_array - array of transmission values for a sloped filter interpolated onto lambda_grid. 
    true_width - final width of the filter. 
    """
    # Define left and right waves at 50% peak transmission
    leftwv_50, rightwv_50 = ctrwv - (width/2), ctrwv + (width/2)
    # Define left and right waves at 100% peak transmission
    leftwv_100, rightwv_100 = leftwv_50 + slope/2, rightwv_50 - slope/2    
    # Define left and right waves at 0% peak transmission assuming perfect filter
    leftwv_0, rightwv_0 = leftwv_50 - slope/2, rightwv_50 + slope/2
    # Define left and right waves to spec out of band transmission
    leftwv_0oB, rightwv_0oB = leftwv_0 - OBbuffer, rightwv_0 + OBbuffer
    # Define left and right waves at 10% peak transmission
    leftwv_10, rightwv_10 = leftwv_0 - (OBbuffer/2), rightwv_0 + (OBbuffer/2)   
    # Define left and right waves at extremes of lambda grid
    minwv, maxwv = lambda_grid[0], lambda_grid[-1]
    # Make array of waves with above
    waves = np.array([minwv, leftwv_0oB, leftwv_10, leftwv_50, leftwv_100, rightwv_100, rightwv_50, rightwv_10, rightwv_0oB, maxwv])
    transmsns = np.array([0.005, 0.005, 0.1, 0.5, 1.0, 1.0, 0.5, .1, .005, .005])*peakT
    #plt.scatter(waves, transmsns)
    #plt.xlim(8300,8900)
    #plt.show()
    # Interpolate onto desired wavelength grid scale
    F = interp1d(waves, transmsns, kind='linear')
    filt_array = F(lambda_grid)
    # Check true width of filter from leftwv_50 to rightwv_50
    left_index = np.argmin(np.abs(lambda_grid - leftwv_50))
    right_index = np.argmin(np.abs(lambda_grid - rightwv_50))
    true_width = lambda_grid[right_index] - lambda_grid[left_index]
    if np.abs(true_width - width) >= widtherr:
        print ("The true width of the filter is different from the user-defined width by more than the user-defined width error.")
    return filt_array, true_width, leftwv_50, rightwv_50, leftwv_100, rightwv_100, leftwv_0, rightwv_0, left_index, right_index


# YOU NEED TO COMMENT THE BELOW! tel aperture must be in cm, int time in seconds and altitude in km

def scintillation1p5(airmass, tel_aperture, int_time, altitude):
    ratio = (airmass ** (3/2) / (tel_aperture ** (2/3))) * (1 / (np.sqrt(2*int_time)))
    return 0.09 * ratio * np.exp(- altitude / 8) # Osborn et al. suggest this should be X 1.5

def scintillation1p75(airmass, tel_aperture, int_time, altitude):
    ratio = (airmass ** (1.75) / (tel_aperture ** (2/3))) * (1 / (np.sqrt(2*int_time)))
    return 0.09 * ratio * np.exp(- altitude / 8)

def scintillation2(airmass, tel_aperture, int_time, altitude):
    ratio = (airmass ** (2) / (tel_aperture ** (2/3))) * (1 / (np.sqrt(2*int_time)))
    return 0.09 * ratio * np.exp(- altitude / 8)


