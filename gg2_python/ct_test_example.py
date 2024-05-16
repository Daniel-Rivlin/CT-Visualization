
# these are the imports you are likely to need
import numpy as np
from material import *
from source import *
from fake_source import *
from ct_phantom import *
from ct_lib import *
from scan_and_reconstruct import *
from create_dicom import *

# create object instances
material = Material()
source = Source()

def multi_circle_phantom(material_indexes):
	#Plot 3 circles per phantom
	a = [material_indexes[0], 0.2, 0.2, -0.3, 0.3, 0]
	b = [material_indexes[1], 0.2, 0.2, 0.3, 0.3, 0]
	c = [material_indexes[2], 0.2, 0.2, -0.3, -0.3, 0]
	# d = [material_indexes[3], 0.2, 0.2, 0.3, -0.3, 0]
	return phantom([a, b, c], 256)

# define each end-to-end test here, including comments
# these are just some examples to get you started
# all the output should be saved in a 'results' directory

def normalize_min_max(data):
	return (data - np.min(data)) / (np.max(data) - np.min(data))

def test_1():
	"""Flattended Geometry Test
	This test performs scan and resconst on a phantom, flattens the output
	to a binary output and then checks the geometry of the reconstructed image.
	This compares the scale and orientation of the image to the phantom
	"""

	# work out what the initial conditions should be
	p = ct_phantom(material.name, 256, 1)
	#s = source.photon('80kVp, 3mm Al')
	s = fake_source(source.mev, 0.08, method='ideal')
	y = scan_and_reconstruct(s, material, p, 0.01, 256)

	# how to check whether these results are actually correct?
	# normalize and flatten
	p_norm = normalize_min_max(p)
	y_norm = normalize_min_max(np.clip(y, 0, None))

	# find difference
	diff_norm = normalize_min_max(np.clip(p_norm - y_norm, 0, None))
	num_pixels = diff_norm.sum(axis=1).sum(axis=0)/256**2

	# save a quantitative result
	f = open('results/test_1_output.txt', mode='w')
	f.write('Fraction of pixels different is ' + str(num_pixels))
	f.close()

	# save relevant qualitative results (images)
	save_draw(p_norm, 'results', 'test_1_phantom_norm')
	save_draw(y_norm, 'results', 'test_1_image_norm')
	save_draw(diff_norm, 'results', 'test_1_diff_norm')

def test_2():
	""" Correlation Test
	This test finds the correlation between the exact attenuation
	values of a phantom at photon energy 0.1 MeV to the attenuation
	values of the reconstruction at the same energy. Values for
	correlation should be large (>0.85) to be considered matching.
	"""

	# generate a phantom image
	correlations = np.zeros(3)
	for i in range(3):
		phantom = ct_phantom(material.name, 256, i+3)

		# simulate scanning and reconstruction using an ideal
		# source emitting photons of energy 0.1 MeV
		s = fake_source(source.mev, 0.1, method='ideal')
		reconstruction = scan_and_reconstruct(s, material, phantom, 0.01, 256)
		phantom[np.where(phantom < 0)] = 0
		reconstruction[np.where(reconstruction < 0)] = 0

		# convert phantom image material indexes to attenuations
		# (at energy 0.1 MeV)
		for m in range(1, len(material.name)):
			if m in phantom:
				phantom[np.where(phantom == m)] = material.coeffs[m, 99]

		# find correlation between phantom and reconstruction
		phantom = phantom.flatten()
		reconstruction = reconstruction.flatten()
		correlations[i] = np.corrcoef(phantom, reconstruction)[0,1]

	f = open('results/test_2_output.txt', mode='w')
	f.write('The following coefficients correspond to phantoms and reconstructions\n with an ideal photon source emitting at a single photon energy of 1 MeV.\n\n')
	f.write('Correlation coefficient for single large hip replacement: ' + str(correlations[0]) + '\n')
	f.write('Correlation coefficient for bilateral hip replacement: ' + str(correlations[1]) + '\n')
	f.write('Correlation coefficient for sphere with three satellites: ' + str(correlations[2]))
	f.close()

def test_3(i=99):
	# Block attenuation values test
    # Sets of simulated material circles are scanned using an idealised source
	# The expected attenuation is calculated using the attenuation coefficient for that energy and compared to the known values
	selected_mev = material.mev[i] #0.1MeV
	s = fake_source(source.mev, selected_mev, method='ideal') #idealised source
	ms_errors = []
	f = open('results/test_3_output.txt', mode='w')
	save_draw(multi_circle_phantom([3, 2, 1]), 'results', 'Test3 Circles')
	X = multi_circle_phantom((3*i, 3*i+1, 3*i+2))
	for i in range(2): #testing first 6 materials in batches of 3
		X = multi_circle_phantom((3*i, 3*i+1, 3*i+2)) #generate phantom
		y = scan_and_reconstruct(s, material, X, 0.0001, 256, alpha = 3)

		ms_errors.append(np.mean((y[80:100, 80:100]-material.coeff(material.name[3*i])[99])**2)) #average over suitable image sections
		ms_errors.append(np.mean((y[80:100, 155:175]-material.coeff(material.name[3*i+1])[99])**2))
		ms_errors.append(np.mean((y[155:175, 80:100]-material.coeff(material.name[3*i+2])[99])**2))

	for i in range(6):
		f.writelines('Squared attenuation error for '+ material.name[i] + ' is ' + str(ms_errors[i]) + "\n")
	total_ms_error = np.mean(ms_errors)

	f.write('Mean squared error is  ' + str(total_ms_error) + "\n")

	if total_ms_error < 0.01:
		f.write("Acceptable value agreement from reconstruction \n")
		f.write("Pass \n")

# Run the various tests
print('Test 1')
test_1()
print('Test 2')
test_2()
print('Test 3')
test_3()
