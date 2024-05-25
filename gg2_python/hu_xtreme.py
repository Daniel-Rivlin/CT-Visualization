import numpy as np
from attenuate import *
from ct_calibrate import *

def hu(Ymax, material, reconstruction, scale):
	""" convert CT reconstruction output to Hounsfield Units
	calibrated = hu(p, material, reconstruction, scale) converts the reconstruction into Hounsfield
	Units, using the material coefficients, photon energy Ymax and scale given."""

	# use water to calibrate
	n = reconstruction.shape[1]
	depth = 2 * n * scale
	# what to do with material???
	water_photons = np.sum(Ymax * np.exp(-depth * material.coeff('Water')))

	# put this through the same calibration process as the normal CT data
	water_photons = -np.log(water_photons / sum(Ymax)) / depth

	# use result to convert to hounsfield units
	reconstruction = 1000 * ((reconstruction - water_photons) / water_photons)

	# limit minimum to -1024, which is normal for CT data.
	reconstruction = np.clip(reconstruction, -1024, None)

	return reconstruction