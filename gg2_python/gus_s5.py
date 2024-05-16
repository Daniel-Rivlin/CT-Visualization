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



def plotter2(fig, ax, X, Y, angles, n=256):
    im0 = ax[0].imshow(X, cmap = "gray", origin='lower')
    ax[0].set_title("Phantom")
    ax[0].axis('off')

    im1 = ax[1].imshow(Y, cmap = "gray",origin='lower',aspect ='auto')
    ax[1].set_title("Sinogram")
    cbar1 = ax[1].figure.colorbar(im1, ax=ax[1],fraction=0.046, pad=0.04)
    cbar1.ax.set_ylabel("Total Attenuation", rotation = -90, va='bottom')
    ax[1].set_box_aspect(1)
    ax[1].set_yticks(np.linspace(0,angles,6), labels=np.round(np.linspace(0,angles,6)/angles*math.pi,3))
    ax[1].set_xticks([])
    ax[1].set_xlim([0,n])
    ax[1].set_ylim([0,angles])
    ax[1].set_ylabel('$\phi$ (rad)')

    fig.tight_layout()

def plotter3(fig, ax, X, Y, angles, n=256):
    im0 = ax[0].imshow(X, cmap = "gray", origin='lower')
    ax[0].set_title("Phantom")
    ax[0].axis('off')

    im1 = ax[1].imshow(Y, cmap = "gray",origin='lower',aspect ='auto', vmin = 0)
    ax[1].set_box_aspect(1)
    ax[1].set_title("Reconstruction")
    cbar1 = ax[1].figure.colorbar(im1, ax=ax[1],fraction=0.046, pad=0.04)
    cbar1.ax.set_ylabel("Attenuation", rotation = -90, va='bottom')


    fig.tight_layout()


n = 256
angles = 256
source2 = source.photon('100kVp, 2mm Al')
X = ct_phantom(material.name, 256, 1)
print(X)
# y = ct_scan(source2, material, X, 0.1, angles)
# # fig, ax = plt.subplots(1,2)
# # plotter2(fig, ax, X, y, angles)

# sinogram = ct_calibrate(source2, material, y, 0.1)
# # fig, ax = plt.subplots(1,2)
# # plotter2(fig, ax, X, cal_y, angles)

# angles = sinogram.shape[0]
# n = sinogram.shape[1]

# # set up filter to be at least twice as long as input
# m = np.ceil(np.log(2*n-1) / np.log(2))
# m = int(2 ** m)

# FT = np.fft.fft(sinogram, n=m, axis=0)
# print(FT.shape)
# frequencies = np.fft.fftfreq(m, scale) * 2 * np.pi
# print(frequencies.shape)

reconstruct_x = scan_and_reconstruct(fake_source(material.mev, 0.08), material, X, 0.1, angles, alpha = 3)
fig, ax = plt.subplots(1,2)
# actual_atten = np.zeros(X.shape)

# for i in range(X.shape[0]):
#     for j in range(X.shape[1]):
#         actual_atten[i, j] = np.sum(material.coeff(material.name[int(X[i,j])])*source2)

plotter3(fig, ax, X, reconstruct_x, angles)
print(np.shape(reconstruct_x))



# reconstruct_x = np.random.random([100, 100])
# m = 200
# ft = np.fft.fft(reconstruct_x[10])
# freq = np.fft.fftfreq(100)
# freq_m = np.fft.fftfreq(200)
# print(reconstruct_x.shape)
# print(ft.shape)
# print(freq)
# print(freq_m)
# n = 100
# m = np.ceil(np.log(2*n-1) / np.log(2))
# m = int(2 ** m)
# print(m)




plt.show()