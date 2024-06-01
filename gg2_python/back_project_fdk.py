import numpy as np
import math
import scipy
from scipy import interpolate
import sys

def back_project_fdk(sinogram3d, R):

	"""back_project back-projection to reconstruct CT data
	back_project(sinogram) back-projects the filtered sinogram
	(angles x samples) to create the reconstructed data (samples x
	samples)"""

	# get input dimensions
	ns = sinogram3d.shape[1]
	angles = sinogram3d.shape[0]
	slices = sinogram3d.shape[2]
	n = int(math.floor((ns-1) // 1) + 1)

	# zero output and form input coordinates
	# these have centre in the middle of the image
	reconstruction = np.zeros((n, n, slices))
	# xi, yi, zi = np.meshgrid(np.arange(0,ns,1) - (ns/2) + 0.5, np.arange(0,ns,1) - (ns/2) + 0.5, np.arange(-slices/2 + 0.5,slices/2 + 0.5))
	xi = yi = np.arange(0,ns,1) - (ns/2) + 0.5
	zi = np.arange(-slices/2 + 0.5,slices/2 + 0.5)
	betas = np.linspace(0, 2*np.pi, angles, endpoint=False)
	f = np.zeros((len(x), len(y), len(z)))
	b = np.zeros((len(x), len(y), len(z), len(betas)))
	a = np.zeros((len(x), len(y), len(betas)))
	U = np.zeros((len(x), len(y), len(betas)))



	# back project over each angle in turn
	for x in range(xi):
			for y in range(yi):
				for z in range(zi):
					for beta in range(len(betas)):
						b[x,y,z,beta] = zi[z] * (R / (R + xi[x] * math.cos(betas[beta]) - yi[y] * math.sin(betas[beta])))
						a[x,y,beta] = R * (-xi[x] * math.sin(betas[beta]) + yi[y] * math.cos(betas[beta])) / (R + xi[x] * math.sin(betas[beta]) + yi[y] * math.cos(betas[beta]))
						U[x,y,beta] = R + xi[x] * math.cos(betas[beta]) + yi[y] * math.sin(betas[beta])
		


	# ensure any data outside the reconstructed circle is set to invalid
	reconstruction[np.where((xi ** 2 + yi ** 2) > (ns/2)**2)] = -1

	sys.stdout.write("\n")

	return reconstruction