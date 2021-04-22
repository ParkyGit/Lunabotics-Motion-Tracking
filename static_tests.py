# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 10:48:03 2021

@author: vargh
"""

import numpy as np
import matplotlib.pyplot as plt

from skimage import data
from scipy.ndimage import fourier_shift
import cv2

image = data.camera()
rows,cols = image.shape
shift = (-25, 15)
# The shift corresponds to the pixel offset relative to the reference image
M = np.float32([[1,0,15],[0,1,-25]])
offset_image = cv2.warpAffine(image,M,(cols,rows))
print("Known offset (y, x): {}".format(shift))

image = np.float32(image)
offset_image = np.float32(offset_image)

dispbuiltin = cv2.phaseCorrelate(image, offset_image)[0]
print("Calculated Offset (Built In) - (y,x): %f, %f"%(dispbuiltin[1], dispbuiltin[0]))

f1 = np.fft.fft2(image)
f2 = np.fft.fft2(offset_image)

cross_power_spect = np.multiply(f1 , np.conjugate(f2))/abs(np.multiply(f1, np.conjugate(f2)))

peakgraph = np.fft.ifft2(cross_power_spect)

detected_shift = np.where(peakgraph == np.amax(peakgraph))

if detected_shift[0][0] > cols//2:
    y_shift = detected_shift[0][0] - cols
else:
    y_shift = detected_shift[0][0]

if detected_shift[1][0] > rows//2:
    x_shift = detected_shift[1][0] - rows
else:
    x_shift = detected_shift[1][0]

print("Calculated Offset (Custom) - (y,x): %f, %f"%(y_shift, x_shift))
