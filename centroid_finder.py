"""
Created on Sun Dec 27 13:19:15 2020

@author: vargh
"""
# This one includes major downsampling

import cv2 as cv
import sys

from skimage.filters import threshold_otsu
from pandas import DataFrame, Series  
import numpy as np
from timeit import default_timer as timer
from skimage.registration import phase_cross_correlation
import turtle

img = cv.imread(cv.samples.findFile("centroid_test"))
if img is None:
    sys.exit("Could not read the image.")
cv.imshow("Display window", img)
k = cv.waitKey(0)

# convert image to grayscale image
gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# convert the grayscale image to binary image
ret,thresh = cv.threshold(gray_image,127,255,0)

# calculate moments of binary image
M = cv.moments(thresh)

# calculate x,y coordinate of center
cX = int(M["m10"] / M["m00"])
cY = int(M["m01"] / M["m00"])

# put text and highlight the center
cv.circle(img, (cX, cY), 5, (255, 255, 255), -1)
cv.putText(img, "centroid", (cX - 25, cY - 25),cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

# display the image
cv.imshow("Image", img)
cv.waitKey(0)


    
    



