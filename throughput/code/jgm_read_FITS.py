from astropy.io import fits

###############################################
# Data Retrieval 
###############################################

def read_fits(filename):
	'''Input: .fits HST CALSPEC data set. Flux callibrated 
	and corrected to heliocentric vacuum reference frame. 
	Output: wavelength (ang) and flux (erg/sec/cm2/sec) arrays. '''

	# header data unit 
	hdu = fits.open(filename)
	hdr = hdu[0].header

	# see summarized content of the opened FITS file 
	#hdu.info()
	#hdu.close()

	# Fish spectrum data out of hdu
	data = hdu[1].data
	wvs = data['WAVELENGTH']            # [ang]
	flux = data['FLUX']                 # [erg/s/cnm2/ang]
	return wvs, flux

