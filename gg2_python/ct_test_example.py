
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

def test_1():
	# explain what this test is for

	# work out what the initial conditions should be
	p = ct_phantom(material.name, 256, 3)
	s = source.photon('100kVp, 3mm Al')
	y = scan_and_reconstruct(s, material, p, 0.01, 256)

	# save some meaningful results
	save_draw(y, 'results', 'test_1_image')
	save_draw(p, 'results', 'test_1_phantom')

	# how to check whether these results are actually correct?

def test_2():
	# This test finds the correlation between the exact 
	# attenuation values of a phantom at photon energy
	# 0.1 MeV to the attenuation values of the reconstruction
	# at the same energy. Values for correlation should be
	# large (>0.7).

	# generate a phantom image
	phantom = ct_phantom(material.name, 256, 3)

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
	correlation = np.corrcoef(phantom, reconstruction)
	print(correlation[0,1])

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
