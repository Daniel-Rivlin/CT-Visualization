import ct_detect
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

    im1 = ax[1].imshow(Y, cmap = "gray",origin='lower',aspect ='auto',vmin=0)
    ax[1].set_title("Reconstruction")
    cbar1 = ax[1].figure.colorbar(im1, ax=ax[1],fraction=0.046, pad=0.04)
    cbar1.ax.set_ylabel("Attenuation", rotation = -90, va='bottom')
    ax[1].set_box_aspect(1)


    fig.tight_layout()

def t_to_p(t):
    photons = source.photon('100kVp, 2mm Al')
    scale = 0.1
    n = 256
    calibration_scan = np.sum(photons * np.exp(-2 * n * scale * material.coeff('Air')))
    p = ct_detect.ct_detect(photons, material.coeff('Water'), t)
    p = -np.log(p / calibration_scan)
    return p

#Working out
# photons = source.photon('100kVp, 2mm Al')
# scale = 0.1
# n = 256
# t = np.linspace(0,scale*n,1000)
# print(photons.shape)
# print(material.coeff('Water').shape)
# print(t.shape)
# p = t_to_p(t)
# print(p)
# t_int = np.linspace(0,scale*n,10)
# p_int = t_to_p(t_int)

# p_values = np.linspace(0.1, np.max(p), 100)

# t1 = np.interp(p_values, p_int, t_int)
# t2 = np.polyval(np.polyfit(p, t, 1), p_values)
# t3 = np.polyval(np.polyfit(p, t, 2), p_values)
# t4 = np.polyval(np.polyfit(p, t, 3), p_values)

# fig,ax = plt.subplots(1)

# ax.plot(p, t, label = "True")

# ax.plot(p_values, t1, label = 'Interpolant, n = 30')
# ax.plot(p_values, t2, label = 'Linear Polyfit')
# ax.plot(p_values, t3, label = 'Quadratic Polyfit')
# ax.plot(p_values, t4, label = 'Cubic Polyfit')

# ax.set_xlabel('Attenuation ($mAs^{-1}$)')
# ax.set_ylabel('Depth (cm)')
# ax.set_title("Curve fits $t_w = f(p_w)$")
# ax.legend()
# plt.show()
# ##Errors
# fig,ax = plt.subplots(1)

# ax.plot(t1, p_values-t_to_p(t1), label = 'Interpolant, n = 30')
# ax.plot(t2, p_values-t_to_p(t2), label = 'Linear Polyfit')
# ax.plot(t3, p_values-t_to_p(t3), label = 'Quadratic Polyfit')
# ax.plot(t4, p_values-t_to_p(t4), label = 'Cubic Polyfit')
# ax.set_ylabel('Attenuation error ($mAs^{-1}$)')
# ax.set_xlabel('Depth')
# ax.set_title("Errors in Curve Fits")
# ax.legend()

###########

n = 256
angles = 256
source2 = source.photon('100kVp, 2mm Al')
X = ct_phantom(material.name, 256, 1, metal='Water')
# y = ct_scan(source2, material, X, 0.1, angles)
# fig, ax = plt.subplots(1,2)
# plotter2(fig, ax, X, y, angles)
# cal_y = ct_calibrate(source2, material, y, 0.1,correct_beam_hardening=True)
# fig, ax = plt.subplots(1,2)
# plotter2(fig, ax, X, cal_y, angles)


reconstruct_x = scan_and_reconstruct(source2, material, X, 0.1, angles, correct_beam_hardening=False)
fig, ax = plt.subplots(1,2)
plotter3(fig, ax, X, reconstruct_x, angles)
print(np.shape(reconstruct_x))

reconstruct_x = scan_and_reconstruct(source2, material, X, 0.1, angles, correct_beam_hardening=True)
fig, ax = plt.subplots(1,2)
plotter3(fig, ax, X, reconstruct_x, angles)
print(np.shape(reconstruct_x))

plt.show()