import numpy as np
import matplotlib.pyplot as plt

def fdk(sinograms, scale, R):
    # get input dimensions
	ns = sinograms.shape[1]
	angles = sinograms.shape[0]
	scans = sinograms.shape[2]
	
    # initialize, radius, bounds and arrays of fan and z location indices
	R *= scale
	ymin = -ns*scale/2
	ymax = -ymin
	z = np.arange(-scans/2 + 0.5, scans/2 + 0.5)
	f = np.arange(-scans/2 + 0.5, scans/2 + 0.5)
	y = np.zeros((scans, scans))
	
    # find intersection locations of each fan on each slice
	for scan in range(scans):
		
		# deal with middle slice case
		if z[scan] == 0:
				y[scan] = -100
				y[scan, int(scans/2+0.5)] = 0
				
		else:
			for fan in range(scans):
					
					if f[fan] != 0:
						# intersection location calculation (y measured from centre of slice)
						y[scan, fan] = R * (z[scan] / f[fan] - 1) - ymax
						 
                        # check whether fan doesn't intersect
						if y[scan, fan] < ymin or y[scan, fan] > ymax:
							y[scan, fan] = -100
						
					# deal with middle fan case (only intersects central slice)
					else:
						y[scan, fan] = -100

    # initialize array of fan indices for each slice
	fan_select = np.zeros((scans, ns))
	
    # distances in millimetres at which slice is sampled
	sample_points = np.linspace(ymin, ymax, ns, endpoint=False)
	sample_points += ymax / ns

    # get the index of the fan whose intersection is closest to the sample points
	for scan in range(scans):
		for sample in range(ns):
			
			# calculate the difference array
			difference_array = np.absolute(y[scan] - sample_points[sample])
			
			# find the index of minimum element from the array
			fan_select[scan, sample] = difference_array.argmin()
	
    # initialize new stack of sinograms
	new_sinograms = np.zeros(sinograms.shape)
	
    # assign values of the original sinogram to new one, this time selecting most appropriate slice in stack for each sample
	# for scan in range(scans):
	# 	for sample in range(ns):
	# 		new_sinograms[:,sample,scan] = sinograms[:,sample,int(fan_select[scan,sample])]

	for slice in range(scans):
		sample_final = np.sum(sinograms[:,:,fan_select[slice]], axis=3)/ns
		new_sinograms[:,:,slice] = sample_final

	return new_sinograms