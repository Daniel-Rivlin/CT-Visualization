
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
	#normalize and flatten
	p_norm = normalize_min_max(p)
	y_norm = normalize_min_max(np.clip(y, 0, None))

	#find difference
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
	# explain what this test is for

	# work out what the initial conditions should be
	p = ct_phantom(material.name, 256, 2)
	s = source.photon('80kVp, 1mm Al')
	y = scan_and_reconstruct(s, material, p, 0.01, 256)

	# save some meaningful results
	save_plot(y[128,:], 'results', 'test_2_plot')

	# how to check whether these results are actually correct?

def test_3():
	# explain what this test is for

	# work out what the initial conditions should be
	p = ct_phantom(material.name, 256, 1)
	s = fake_source(source.mev, 0.1, method='ideal')
	y = scan_and_reconstruct(s, material, p, 0.1, 256)

	# save some meaningful results
	f = open('results/test_3_output.txt', mode='w')
	f.write('Mean value is ' + str(np.mean(y[64:192, 64:192])))
	f.close()

	# how to check whether these results are actually correct?


# Run the various tests
print('Test 1')
test_1()
print('Test 2')
test_2()
print('Test 3')
test_3()
