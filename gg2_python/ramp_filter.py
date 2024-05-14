import math
import numpy as np
import numpy.matlib

def ramp_filter(sinogram, scale, alpha=0.001):
	""" Ram-Lak filter with raised-cosine for CT reconstruction

	fs = ramp_filter(sinogram, scale) filters the input in sinogram (angles x samples)
	using a Ram-Lak filter.

	fs = ramp_filter(sinogram, scale, alpha) can be used to modify the Ram-Lak filter by a
	cosine raised to the power given by alpha."""

	# get input dimensions
	angles = sinogram.shape[0]
	n = sinogram.shape[1]

	# set up filter to be at least twice as long as input
	m = np.ceil(np.log(2*n-1) / np.log(2))
	m = int(2 ** m)

	# apply filter to all angles
	print('Ramp filtering')

	# step 1: take the Fourier transform in r direction
	FT = np.fft.fft(sinogram, axis=1)

	# step 2: multiply each frequency by the appropriate coefficient in the Ram-Lak filter
	frequencies = np.linspace(0, 2*np.pi, n, endpoint=False)
	RL_multipliers = (abs(frequencies) / (2 * np.pi)) * np.power((np.cos(frequencies / 4)), alpha)
	RL_multipliers[0] = RL_multipliers[1]/6
	FT *= RL_multipliers

	# step 3: take the inverse Fourier transform in the r direction
	sinogram = np.real(np.fft.ifft(FT, axis=1)).astype(float)
	
	return sinogram