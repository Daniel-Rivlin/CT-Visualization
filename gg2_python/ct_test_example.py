
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
	# explain what this test is for

	# work out what the initial conditions should be
	# p = ct_phantom(material.name, 256, 2)
	# s = source.photon('80kVp, 1mm Al')
	# y = scan_and_reconstruct(s, material, p, 0.01, 256)

	# # save some meaningful results
	# save_plot(y[128,:], 'results', 'test_2_plot')

	# check that attenuations at distinct energies match phantom

	# convert phantom into attenuations
	phantom = ct_phantom(material.name, 256, 1)
	s = fake_source(source.mev, 0.1, method='ideal')
	reconstruction = scan_and_reconstruct(s, material, phantom, 0.01, 256)

	phantom[np.where(phantom < 0)] = 0
	reconstruction[np.where(reconstruction < 0)] = 0
	for m in range(1, len(material.name)):
		if m in phantom:
			phantom[np.where(phantom == m)] = material.coeffs[m, 9]

	# generate a fake source with single energy
	draw(phantom)
	draw(reconstruction)

	# find correlation between phantom and reconstruction
	phantom = phantom.flatten()
	reconstruction = reconstruction.flatten()
	correlation = np.corrcoef(phantom, reconstruction)
	print(correlation[0,1])

def test_3(i=99):
	#Block attenuation values test
    #Sets of simulated material circles are scanned using an idealised source
	#The expected attenuation is calculated using the attenuation coefficient for that energy and compared to the known values
	selected_mev = material.mev[i] #0.1MeV
	s = fake_source(source.mev, selected_mev, method='ideal') #idealised source
	ms_errors = []
	f = open('results/test_3_output.txt', mode='w')
	save_draw(multi_circle_phantom([1,2,3]), 'results', 'Test3 Circles')
	X = multi_circle_phantom((3*i, 3*i+1, 3*i+2))
	for i in range(2): #testing first 6 materials in batches of 3
		X = multi_circle_phantom((3*i, 3*i+1, 3*i+2)) #generate phantom
		y = scan_and_reconstruct(s, material, X, 0.0001, 256, alpha = 3)

		ms_errors.append(np.mean((y[80:100, 80:100]-material.coeff(material.name[3*i])[99])**2)) #average over suitable image sections
		ms_errors.append(np.mean((y[80:100, 155:175]-material.coeff(material.name[3*i+1])[99])**2))
		ms_errors.append(np.mean((y[155:175, 80:100]-material.coeff(material.name[3*i+2])[99])**2))

	for i in range(3):
		f.writelines('Squared attenuation error for '+ material.name[i] + ' is ' + str(ms_errors[i]) + "\n")
	total_ms_error = np.mean(ms_errors)

	f.write('Mean squared error is  ' + str(total_ms_error) + "\n")

	if total_ms_error < 0.01:
		f.write("Acceptable value agreement from reconstruction \n")
		f.write("Pass \n")

# Run the various tests
# print('Test 1')
# test_1()
print('Test 2')
test_2()
# print('Test 3')
# test_3()
