# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 11:04:46 2021

@author: vargh
"""


import cv2
from skimage.filters import threshold_otsu
from pandas import DataFrame, Series
import numpy as np
from timeit import default_timer as timer
from skimage.registration import phase_cross_correlation
import turtle
import matplotlib.pyplot as plt
import time as t

# Takes two frames, calculates the phase cross correlation between them and outputs displacement
def calcdisp(imshape, frame1, frame2):
    tmp = np.array([0,0])

    f1 = np.fft.fft2(frame1)
    f2 = np.fft.fft2(frame2)

    cross_power_spect = np.multiply(f1 , np.conjugate(f2))/abs(np.multiply(f1, np.conjugate(f2)))

    peakgraph = np.fft.ifft2(cross_power_spect)

    detected_shift = np.where(peakgraph == np.amax(peakgraph))

    if detected_shift[0][0] > imshape[0]//2:
        tmp[1] = detected_shift[0][0] - imshape[0]
    else:
        tmp[1] = detected_shift[0][0]

    if detected_shift[1][0] > imshape[1]//2:
        tmp[0] = detected_shift[1][0] - imshape[1]
    else:
        tmp[0] = detected_shift[1][0]


    if abs(tmp[0]) > 10:
        tmp[0] = 0
    if abs(tmp[1]) > 10:
        tmp[1] = 1


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
turtle.setup(width=300, height=300, startx=0, starty=0)
cap = cv2.VideoCapture('C:/Users/vargh/Desktop/IMG_2569.MOV')
t.sleep(2)

while(True):
    # Capture the video frame by frame

    ret, frame = cap.read() # get frame

    #frameShape = frame.shape #?

    im = cv2.resize(frame, None, fx=.25, fy=.25) # decimate quality of image by resizing

    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([36,140,140])
    upper_blue = np.array([86,255,255])

    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    res = cv2.bitwise_and(im,im, mask= mask)
    proc_im = cv2.cvtColor(res, cv2.COLOR_HSV2RGB)


    bw_img = cv2.cvtColor(np.float32(proc_im), cv2.COLOR_RGB2GRAY) # convert to grayscale

    imshape = np.array(bw_img.shape)
#    gpuframe1.upload(bw_img) (gpu acceleration, doesn't work)

    if iteration > 3 and iteration <= 15: # waits till there are sufficient frames to calculate
        calcdisp(imshape, bw_img, prevFrame)
    if iteration > 15: # allows program to 'warm up'. In initial tests, initial measurements were not accurate
        tmp = calcdisp(imshape, bw_img, prevFrame) # raw displacement data
        dstep = np.vstack((dstep, tmp)) # stacks the displacement step data just recieved
        time = np.vstack((time, timer()-start)) # stacks the time data


        totD = np.vstack((totD, np.sum(dstep, axis=0))) # sums displacement steps to calculate total displacement

        # updates turtle
        currentLoc.sety(totD[iteration-15, 1]*.35)
        currentLoc.setx(totD[iteration-15, 0]*.35)

    prevFrame = bw_img # sets current frame as previous frame
#    gpuframe2.upload(prevFrame)
    iteration = iteration + 1 #increases iteration
    print(iteration,timer()-start) # prints time (for debugging/optimization purposes)
    cv2.imshow('frame2',res)
    cv2.imshow('frame3', im)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the capture object

vid.release()

# Destroy all the windows
