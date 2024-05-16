from ct_detect import ct_detect
import numpy as np
from material import *
from source import *
from fake_source import *
from ct_phantom import *
from ct_lib import *
from scan_and_reconstruct import *
from create_dicom import *
import matplotlib.pyplot as plt
material = Material()
source = Source()

# #3.3.2
# water_coeff = material.coeff('Water')
# mev_values = material.mev
# # k1 = 1.539*(10**-5)
# # k2 = 4.375*(10**-9)
# # compton = k1 * 1000 / mev_values
# # photoelectric = k2 * 1000 / (mev_values**3)

# plt.loglog(mev_values, water_coeff)
# plt.axvline(x=0.016, color='r', linestyle='-')
# plt.axvline(x=0.05, color='r', linestyle='-')

# # plt.loglog(mev_values[8:], compton[8:])
# # plt.loglog(mev_values[0:50], photoelectric[0:50])
# plt.title('$\mu_{water}$ for different X-ray energies')
# plt.xlabel('Photon Energy (MeV)')
# plt.ylabel('$\mu_{water}$')


#3.3.3
# y1mm = ct_detect(source.photon('100kVp, 1mm Al'), material.coeff('Water'), np.arange(0, 10.1, 0.1), 1)
# y2mm = ct_detect(source.photon('100kVp, 2mm Al'), material.coeff('Water'), np.arange(0, 10.1, 0.1), 1)
# y3mm = ct_detect(source.photon('100kVp, 3mm Al'), material.coeff('Water'), np.arange(0, 10.1, 0.1), 1)
# y4mm = ct_detect(source.photon('100kVp, 4mm Al'), material.coeff('Water'), np.arange(0, 10.1, 0.1), 1)
# # Original
# plt.plot(np.arange(0, 10.1, 0.1) ,np.log(y1mm), label = "1mm Al")
# plt.plot(np.arange(0, 10.1, 0.1) ,np.log(y2mm), label = "2mm Al")
# plt.plot(np.arange(0, 10.1, 0.1) ,np.log(y3mm), label = "3mm Al")
# plt.plot(np.arange(0, 10.1, 0.1) ,np.log(y4mm), label = "4mm Al")
# plt.title('Al (100kVp) in water')
# plt.xlabel('Water Depth (mm)')
# plt.ylabel('Estimated photons at detector')
# plt.legend()

# # Fake Source
# source1 =fake_source(material.mev, 0.008, method = 'ideal')
# source2 =fake_source(material.mev, 0.04, method = 'ideal')
# source3 =fake_source(material.mev, 0.2, method = 'ideal')

# i1 = ct_detect(source1, material.coeff('Water'), np.arange(0, 10.1, 0.1), 1)
# i2 = ct_detect(source2, material.coeff('Water'), np.arange(0, 10.1, 0.1), 1)
# i3 = ct_detect(source3, material.coeff('Water'), np.arange(0, 10.1, 0.1), 1)

# # Input Graph
# # plt.loglog(material.mev, fake_source(material.mev, 0.008, method = 'ideal'), label="MVp = 0.008")
# # plt.loglog(material.mev, fake_source(material.mev, 0.04, method = 'ideal'), label="MVp = 0.04")
# # plt.loglog(material.mev, fake_source(material.mev, 0.2, method = 'ideal'), label="MVp = 0.2")

# # water_coeff = material.coeff('Water')
# # plt.loglog(material.mev, water_coeff, label = "$\mu_{water}$")
# # plt.xlabel('Photon Energy (MeV)')
# # plt.title('Ideal Inputs')
# # plt.legend()

# #Output Graph
# plt.plot(np.arange(0, 10.1, 0.1) ,np.log(i1), label = "MVp = 0.008")
# plt.plot(np.arange(0, 10.1, 0.1) ,np.log(i2), label = "MVp = 0.04")
# plt.plot(np.arange(0, 10.1, 0.1) ,np.log(i3), label = "MVp = 0.2")

# plt.title('Ideal sources in water')
# plt.xlabel('Water Depth (mm)')
# plt.ylabel('Estimated photons at detector')
# plt.legend()


##MultiMaterial
y_air = ct_detect(source.photon('100kVp, 1mm Al'), material.coeff('Air'), np.arange(0, 10.1, 0.1), 1)
y_water = ct_detect(source.photon('100kVp, 2mm Al'), material.coeff('Water'), np.arange(0, 10.1, 0.1), 1)
y_bone = ct_detect(source.photon('100kVp, 3mm Al'), material.coeff('Bone'), np.arange(0, 10.1, 0.1), 1)
y_titanium = ct_detect(source.photon('100kVp, 4mm Al'), material.coeff('Titanium'), np.arange(0, 10.1, 0.1), 1)

plt.plot(np.arange(0, 10.1, 0.1) ,np.log(y_air), label = "Air")
plt.plot(np.arange(0, 10.1, 0.1) ,np.log(y_water), label = "Water")
plt.plot(np.arange(0, 10.1, 0.1) ,np.log(y_bone), label = "Bone")
plt.plot(np.arange(0, 10.1, 0.1) ,np.log(y_titanium), label = "Titanium")
plt.title('Al (100kVp, 2mm) in different materials')
plt.xlabel('Depth (mm)')
plt.ylabel('Estimated photons at detector')
plt.legend()


plt.show()