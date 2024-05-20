import numpy as np
import scipy
from scipy import interpolate
import ct_detect
def ct_calibrate(photons, material, sinogram, scale, correct_beam_hardening = False, mas=10000):

	""" ct_calibrate convert CT detections to linearised attenuation
	sinogram = ct_calibrate(photons, material, sinogram, scale) takes the CT detection sinogram
	in x (angles x samples) and returns a linear attenuation sinogram
	(angles x samples). photons is the source energy distribution, material is the
	material structure containing names, linear attenuation coefficients and
	energies in mev, and scale is the size of each pixel in x, in cm."""

	# Get dimensions and work out detection for just air of twice the side
	# length (has to be the same as in ct_scan.py)
	n = sinogram.shape[1]

	# perform calibration
	depth = 2 * n * scale
	calibration_scan = np.sum(photons * np.exp(-depth * material.coeff('Air')))
	sinogram = -np.log(sinogram / calibration_scan)

	if correct_beam_hardening == True:
		C = 1
		#Correct beam hardening
		#Gather Data
		t = np.linspace(0,scale*n,1000)
		calibration_scan = np.sum(photons * np.exp(-2 * n * scale * material.coeff('Air')))
		p = ct_detect.ct_detect(photons, material.coeff('Water'), t)
		p = -np.log(p / calibration_scan)

		
		# #Polyfit approach
		fit_coeffs = np.polyfit(p, t, 3) #cubic
		sinogram = np.polyval(fit_coeffs, sinogram)

		# #Interpolation approach 
		# sinogram = np.interp(sinogram, p_fit, t)

		sinogram = sinogram * C #should be unnecessary with conversion to HU
	
	return sinogram