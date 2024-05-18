import numpy as np
from attenuate import *
from ct_calibrate import *

def hu(p, material, reconstruction, scale):
	""" convert CT reconstruction output to Hounsfield Units
	calibrated = hu(p, material, reconstruction, scale) converts the reconstruction into Hounsfield
	Units, using the material coefficients, photon energy p and scale given."""

	# use water to calibrate
	n = reconstruction.shape[1]
	depth = 2 * n * scale
	water_photons = np.sum(p * np.exp(-depth * material.coeff('Water')))

	# put this through the same calibration process as the normal CT data

	calibration_scan = np.sum(p * np.exp(-depth * material.coeff('Air')))
	water_photons = -np.log(water_photons / calibration_scan) / depth

	# use result to convert to hounsfield units
	reconstruction = 1000 * ((reconstruction - water_photons) / water_photons)

	# limit minimum to -1024, which is normal for CT data.
	reconstruction = np.clip(reconstruction, -1024, 3072)

	return reconstruction