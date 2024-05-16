import math
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
def ramp_filter(sinogram, scale, alpha=0.1):
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

	# step 1: take the Fourier transform in r direction
	FT = np.fft.fft(sinogram, n=m, axis=1)

	# step 2: multiply each frequency by the appropriate coefficient in the Ram-Lak filter
	frequencies = np.fft.fftfreq(m, scale)
	RL_multipliers = (abs(frequencies)) * np.power(np.cos((frequencies / np.max(abs(frequencies))) * np.pi/2), alpha)
	for row in FT:
		row *= RL_multipliers

	# step 3: take the inverse Fourier transform in the r direction
	sinogram = np.real(np.fft.ifft(FT, axis=1)[:,:n])

	return sinogram