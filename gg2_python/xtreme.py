import numpy as np
import scipy
from scipy import ndimage
import math
import os
import sys
from ramp_filter import *
from back_project import *
from fdk import *
from create_dicom import *
from ct_lib import *

class Xtreme(object):
    def __init__(self, file):

        """   x = Xtreme( filename ) reads the Xtreme scanner .RSQ file
        specified by filename, and initialises the class with a header
        structure with information about that file. The structure contains
        the following fields:
 
        'scans' - number of scans (z-axis locations), including overlapping
                  scans
        'angles' - total number of angles (x-y plane rotations, at each z-axis
                   location)
        'samples' - number of samples (at each rotational angle, and in each
                      z-axis location)
        'skip_scans' - number of scans at each end of each z-fan
                         which overlap with the neighbouring z-fan
        'skip_angles' - number of rotational angles in a full rotation which
                          are additional to the required fan rotation
        'skip_samples' - number of samples in each line which are
                             invalid and not read in from the file
        'fan_scans' - number of z-axis scans in each z-fan group
        'fan_angles' - number of rotational angles which correspond to a
                         single (x-y plane) fan
        'recon_angles' - number of rotational angles in 180 degrees
        'dtheta' - angle between each measurement, in radians 
        'fan_theta' - x-y plane fan angle, in radians
        'radius' - distance from X-ray source to centre of rotation, expressed
                   in terms of numbers of samples
        'scale' - pixel size and z-increment, in mm
        'data_offset' - internal offset used for reading data from file
        'filename' - name of file"""

        self.okay = True

        if not os.path.isfile(file):
            self.okay = False
            print('File not found')

        if self.okay:
            f = open(file, 'rb')
            self.okay = f.readable()
            if not self.okay:
                print('File not readable')

        if self.okay:
            b = f.read(16)
            if b.decode() != 'CTDATA-HEADER_V1':
                self.okay = False
                print('File is not an Xtreme RSQ file')

        # now read in the data and form the output structure
        # 7: dimx_p
        # 8: dimy_p
        # 9: dimz_p
        # 14: slice_increment_um
        # 19: nr_of_samples
        # 20: nr_of_projections
        # 123: data_offset
        if self.okay:
            b = f.read(124*4)
            h = np.frombuffer(b, np.int32, 124)

            res = h[19]/h[7]
            left_samples = math.floor(36.0/res)
            right_samples = math.floor(17.0/res)
            skip_samples = left_samples + right_samples

            self.scans = h[9]  # number of scans
            self.angles = h[8] - 2 # number of angles in each total sweep
            self.samples = h[7] - skip_samples # number of sample measurements at each angle

            self.skip_scans = math.floor(18/res)
            self.skip_angles = 2*math.floor(12/res)
            self.skip_samples = skip_samples
            self.left_samples = left_samples
            self.right_samples = right_samples

            self.fan_scans = math.floor(h[20]/res)
            self.fan_angles = math.floor(138/res)
            self.recon_angles = (self.angles - self.skip_angles - self.fan_angles)
            self.dtheta = math.pi/float(self.recon_angles)
            self.fan_theta = self.dtheta * self.fan_angles

            self.radius = (float(self.samples)/2.0)/math.tan(float(self.fan_theta)/2.0)
            self.scale = 0.00104*float(h[14])

            self.data_offset = h[123]
            self.filename = file

            f.close()

    def get_rsq_scan(self, angle):

        """ [Y, Ymin, Ymax] = get_rsq_scan( A ) reads in angle A from the file.
        
        The returned data Y is effectively an X-ray at angle A, except that
        every slice is included, despite the raw data being split into z-fans of
        fan_scans size each. Hence Y is of size (scans x samples). Ymin
        are the recorded detections when there is no X-ray source, and Ymax are
        the recorded detections when there is no object in the scanner."""

        if not self.okay:
            print('File not opened correctly')
            return

        if (angle < 0) or (angle >= self.angles):
            print('Angle is not within range')
            return

        # open file and get to start of data
        f = open(self.filename, 'rb')
        f.seek((self.data_offset+1)*512,0)

        # loop through scans, selecting appropriate angle, and all calibration data
        Y = np.zeros([self.scans, self.samples])
        Ymin = np.zeros([self.scans, self.samples])
        Ymax = np.zeros([self.scans, self.samples])
        for scan in range(0, self.scans):
            f.seek(self.left_samples*2, 1)
            b = f.read(2*self.samples)
            Ymin[scan] = np.frombuffer(b, np.int16, self.samples)
            f.seek(self.skip_samples*2, 1)
            b = f.read(2*self.samples)
            Ymax[scan] = np.frombuffer(b, np.int16, self.samples)
            f.seek((self.samples+self.skip_samples)*angle*2, 1)
            f.seek(self.skip_samples*2, 1)
            b = f.read(2*self.samples)
            Y[scan] = np.frombuffer(b, np.int16, self.samples)
            f.seek(self.right_samples*2, 1)
            f.seek((self.samples+self.skip_samples)*(self.angles-angle-1)*2, 1)

        f.close()

        return Y, Ymin, Ymax

    def get_rsq_slice(self, scan):

        """ [Y, Ymin, Ymax] = get_rsq_slice( F ) reads in slice F from the file.

        The returned data Y is a fan-based sinogram of size (angles x 
        samples), Ymin are the recorded detections when there is no X-ray
        source, and Ymax are the recorded detections when there is no object in
        the scanner."""

        if not self.okay:
            print('File not opened correctly')
            return

        if (scan < 0) or (scan >= self.scans):
            print('Scan is not within range')
            return

        # open file and get to start of data
        f = open(self.filename, 'rb')
        f.seek((self.data_offset+1)*512,0)

        # Advance to requested slice
        f.seek((self.samples+self.skip_samples)*(self.angles+2)*2*scan, 1)

        # read in calibration data
        f.seek(self.left_samples*2, 1)
        b = f.read(self.samples*2)
        Ymin = np.frombuffer(b, np.int16, self.samples)
        f.seek(self.skip_samples*2, 1)
        b = f.read(self.samples*2)
        Ymax = np.frombuffer(b, np.int16, self.samples)

        # loop to read scan
        Y = np.zeros([self.angles, self.samples])
        for angle in range(0, self.angles):
            f.seek(self.skip_samples*2, 1)
            b = f.read(2*self.samples)
            Y[angle] = np.frombuffer(b, np.int16, self.samples)

        f.close()

        return Y, Ymin, Ymax

    def fan_to_parallel(self, X):

        """ Y = fan_to_parallel( X ) takes the raw sinogram in X (angles x
        samples) and converts this to an equivalent parallel-beam sinogram
        in Y (recon_angles x samples)."""

        # print('Fan to parallel sinogram')
        
        # calculate some required parameters
        angles = self.recon_angles
        samples = self.samples
        atheta = self.fan_theta*0.172# about 1/6 of the fan angle
        c = self.samples/2 - 0.5   # the centre sample

        # form output coordinates - y0 is zero-based at this point
        xo1, yo1 = np.meshgrid(np.arange(samples), np.arange(angles))
        #xo1 = np.zeros([angles, samples])
        #for index, value in np.ndenumerate(xo1):
        #    xo1[index] = index[1]
        #yo1 = np.zeros([angles, samples])
        #for index, value in np.ndenumerate(yo1):
        #    yo1[index] = index[0]
        yo = np.arcsin((xo1-c)/self.radius)

        # n1, n2 and n3 signify samples on one of three separate detector arrays,
        # each occupying one-third of the fan angle
        xo = np.zeros((angles, samples))
        #for index, y in np.ndenumerate(yo):
        #    if y>atheta:
        #        xo[index] = (xo1[index]-(5.0*samples/6.0))/np.cos(y-2.0*atheta) + c + samples/3.0
        #    elif y<-atheta:
        #        xo[index] = (xo1[index]-(samples/6.0))/np.cos(y+2.0*atheta) + c - samples/3.0
        #    else:
        #        xo[index] = (xo1[index]-(samples/2.0))/np.cos(y) + c
        index = yo>atheta
        xo[index] = (xo1[index]-(5.0*samples/6.0))/np.cos(yo[index]-2.0*atheta) + c + samples/3.0
        index = yo<-atheta
        xo[index] = (xo1[index]-(samples/6.0))/np.cos(yo[index]+2.0*atheta) + c - samples/3.0
        index = np.logical_and(yo<=atheta, yo>=-atheta)
        xo[index] = (xo1[index]-(samples/2.0))/np.cos(yo[index]) + c

        # adjust angle so it is not zero-based
        yo = yo/self.dtheta + yo1 + self.skip_angles/2.0 + self.fan_angles/2.0 - 0.5

        # actually perform the interpolation
        Y = scipy.ndimage.map_coordinates(X, [yo, xo], None, 1, 'constant', 0, False)

        return Y




    def reconstruct_all(self, file, method=None, alpha=None):
        
        """ reconstruct_all( FILENAME, ALPHA ) creates a series of DICOM
        files for the Xtreme RSQ data. FILENAME is the base file name for
        the data, and ALPHA is the power of the raised cosine function
        used to filter the data.
        
        reconstruct_all( FILENAME, ALPHA, METHOD ) can be used to
        specify how the data is reconstructed. Possible options are:
        'parallel' - reconstruct each slice separately using a fan to parallel
                           conversion
        'fdk' - approximate FDK algorithm for better reconstruction"""
                
        if alpha is None:
            alpha = 0.001

        if method is None:
            method = 'parallel'

        # set frame number and DICOM UIDs for saving to multiple frames
        z = 1
        seriesuid = pydicom.uid.generate_uid()
        studyuid = pydicom.uid.generate_uid()
        frameuid = pydicom.uid.generate_uid()
        time = datetime.datetime.now()
        sz = sp = self.scale

        # create grid of premultipliers for fdk
        # (accounts for fans travelling different distances through material)
        if method == 'fdk':
            R = self.radius
            b = np.arange(0.5 - (self.fan_scans/2), 0.5 + (self.fan_scans/2))
            a = np.arange(0.5 - (self.samples/2), 0.5 + (self.samples/2))
            b = np.tile(np.square(b), (self.samples, 1))
            b = b.T
            a = np.tile(np.square(a), (self.fan_scans, 1))
            multimat = R / np.sqrt(a + b + R*R)

        # main loop over each z-fan
        for fan in range(0, self.scans, self.fan_scans):
            if method == 'fdk':
                # correct reconstruction using FDK method, self.fan_scans scans at a time
                if fan + self.fan_scans > self.scans:
                    fan_total = self.scans - fan
                    b = np.arange(0.5 - (fan_total/2), 0.5 + (fan_total/2))
                    b = np.tile(np.square(b), (self.samples, 1))
                    b = b.T
                    a = np.tile(np.square(a), (fan_total, 1))
                    multimat = R / np.sqrt(a + b + R*R)
                else:
                    fan_total = self.fan_scans
                # 3d stack of sinograms
                comb3d = np.zeros((self.angles, self.samples, fan_total))
                
                print("\n retrieving data and filtering...")
                for thetaN in range(self.angles):

                    # get the current fan scan info
                    [Y, Ymin, Ymax] = self.get_rsq_scan(thetaN)

                    # account for background radiation
                    Y -= Ymin
                    calib_scan = Ymax - Ymin
                    
                    # take fan section
                    Y = Y[fan:fan+fan_total]
                    calib_scan = calib_scan[fan:fan+fan_total]

                    # premultiply
                    Y = np.multiply(Y, multimat)
                    calib_scan = np.multiply(calib_scan, multimat)

                    # calibrate
                    Y = np.clip(Y, 1, None)
                    calib_scan = np.clip(calib_scan, 1, None)
                    Y = -np.log(Y / calib_scan)

                    # Ram-Lak filter
                    Y = ramp_filter(Y, self.scale, alpha)

                    # Add to 3d array
                    comb3d[thetaN] = Y.T

                # make slices parallel in x-y planes
                new_angles = self.fan_to_parallel(comb3d[:,:,0]).shape[0]
                comb3dpara = np.zeros((new_angles, comb3d.shape[1], comb3d.shape[2]))
                for slice in range(comb3d.shape[2]):
                    comb3dpara[:,:,slice] = self.fan_to_parallel(comb3d[:,:,slice])
                
                # use fdk to select best slice for each sample in sinograms
                comb3dpara = fdk(comb3dpara, self.scale, self.radius)

                # remove overlap scans
                comb3dpara = comb3dpara[:,:,self.skip_scans:fan+fan_total-self.skip_scans]
                
                # reconstruct
                for slice in range(comb3dpara.shape[2]):
                    # back-projection
                    reconstruction = back_project(comb3dpara[:,:,slice])

                    # HU unit conversion
                    mu_w = 0.0234
                    reconstruction = ((reconstruction - mu_w) / mu_w) * 1000
                    reconstruction = np.clip(reconstruction, -1024, None)

                    # save as dicom file
                    create_dicom(reconstruction, filename=file, sp=sp, sz=sz, f=z, study_uid=studyuid, series_uid=seriesuid, frame_uid=pydicom.uid.generate_uid(), time=datetime.datetime.now(), storage_directory='DICOM a fdk')
                    print("file", z, "/ 605")
                    z = z + 1

            else:

                # default method should reconstruct each slice separately
                for scan in range(fan+self.skip_scans,fan+self.fan_scans-self.skip_scans):
                    if (scan<self.scans):

                        # get the current slice info
                        [Y, Ymin, Ymax] = self.get_rsq_slice(scan)

                        # account for background radiation
                        Y -= Ymin
                        calib_scan = Ymax - Ymin

                        # convert to parallel
                        calib_scan = np.tile(calib_scan, (Y.shape[0], 1))
                        Y = self.fan_to_parallel(Y)
                        calib_scan = self.fan_to_parallel(calib_scan)
                        calib_scan = calib_scan[0]

                        # calibrate
                        Y = np.clip(Y, 1, None)
                        calib_scan = np.clip(calib_scan, 1, None)
                        Y = -np.log(Y / calib_scan[None,:])

                        # Ram-Lak filter
                        Y = ramp_filter(Y, self.scale, alpha)

                        # back-projection
                        reconstruction = back_project(Y)

                        # HU unit conversion
                        mu_w = 0.0234
                        reconstruction = ((reconstruction - mu_w) / mu_w) * 1000
                        reconstruction = np.clip(reconstruction, -1024, None)

						# save as dicom file
                        create_dicom(reconstruction, filename=file, sp=sp, sz=sz, f=z, study_uid=studyuid, series_uid=seriesuid, frame_uid=pydicom.uid.generate_uid(), time=datetime.datetime.now(), storage_directory='DICOM b para')
                        print("file", z, "/ 581")
                        z = z + 1
        return

