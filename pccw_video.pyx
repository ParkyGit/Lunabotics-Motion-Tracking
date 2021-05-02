# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 10:45:18 2021

@author: vargh
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 13:19:15 2020

@author: vargh
"""
# This one includes major downsampling


import cv2

from skimage.filters import threshold_otsu
from pandas import DataFrame, Series
import numpy as np
from timeit import default_timer as timer
from skimage.registration import phase_cross_correlation
import turtle
import matplotlib.pyplot as plt

# Takes two frames, calculates the phase cross correlation between them and outputs displacement
def calcdisp(frame1, frame2):
    detected_shift = cv2.phaseCorrelate(frame2, frame1)# phase cross correlation algorithm
    if abs(detected_shift[0][0]) < 30 and abs(detected_shift[0][1]) < 30:
        tmp = [detected_shift[0][0], detected_shift[0][1]] # store x,y displacement
    else:
        tmp = [0,0]
    return tmp

vid = cv2.VideoCapture(0) # Starts video capture object

# Initializations
iteration = 0
prevFrame = 0
dstep = np.empty([1,2])
totD= np.array([[0,0]])
time = [0]
start = timer()

#gpuframe1 = cv2.cuda_GpuMat() (Could not get gpu acceleration to work)
#gpuframe2 = cv2.cuda_GpuMat()
currentLoc = turtle.Turtle() # Initializes turtle for visualization
cap = cv2.VideoCapture('C:/Users/vargh/Desktop/IMG_2509.MOV')

while(True):
    # Capture the video frame by frame

    ret, frame = cap.read() # get frame

    #frameShape = frame.shape #?

    im = cv2.resize(frame, None, fx=.5, fy=.5) # decimate quality of image by resizing

    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([36,140,140])
    upper_blue = np.array([86,255,255])

    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    res = cv2.bitwise_and(im,im, mask= mask)
    proc_im = cv2.cvtColor(res, cv2.COLOR_HSV2RGB)


    bw_img = cv2.cvtColor(np.float32(proc_im), cv2.COLOR_RGB2GRAY) # convert to grayscale
#    gpuframe1.upload(bw_img) (gpu acceleration, doesn't work)

    if iteration > 3: # waits till there are sufficient frames to calculate
        calcdisp(bw_img, prevFrame)
    if iteration > 15: # allows program to 'warm up'. In initial tests, initial measurements were not accurate
        tmp = calcdisp(bw_img, prevFrame) # raw displacement data
        dstep = np.vstack((dstep, tmp)) # stacks the displacement step data just recieved
        time = np.vstack((time, timer()-start)) # stacks the time data


        totD = np.vstack((totD, np.sum(dstep, axis=0))) # sums displacement steps to calculate total displacement

        # updates turtle
        currentLoc.sety(totD[iteration-15, 1]*.2)
        currentLoc.setx(totD[iteration-15, 0]*.2)

    prevFrame = bw_img # sets current frame as previous frame
#    gpuframe2.upload(prevFrame)
    iteration = iteration + 1 #increases iteration
    print(iteration,timer()-start) # prints time (for debugging/optimization purposes)
    cv2.imshow('frame2',res)
    if cv2.waitKey(50) & 0xFF == ord('q'):
        break

# After the loop release the capture object

vid.release()

# Destroy all the windows
