# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 11:04:46 2021

@author: vargh
"""


import cv2
import numpy as np
from timeit import default_timer as timer

import turtle
import matplotlib.pyplot as plt
import time as t
from numba import njit

# Takes two frames, calculates the phase cross correlation between them and outputs displacement
@njit
def calcdisp(imshape, frame2, frame1):       
    center = imshape/2
    bound_box_x = int(center[0]/2)-1
    bound_box_y = int(center[1]/2)-1
    
    smaller_prev_frame = frame1[int(center[0] - bound_box_x): int(center[0] + bound_box_x), int(center[1] - bound_box_y): int(center[1] + bound_box_y)]
    peaks_init = np.where(smaller_prev_frame == np.amax(smaller_prev_frame))
    peaks_init = [peaks_init[1][0] + bound_box_y, peaks_init[0][0] + bound_box_x]
    
    search_pix = 10
    
    neighbor_search2 = frame2[int(peaks_init[1] - search_pix): int(peaks_init[1] + search_pix), int(peaks_init[0] - search_pix): int(peaks_init[0] + search_pix)]
    neighbor_search1 = frame1[int(peaks_init[1] - search_pix): int(peaks_init[1] + search_pix), int(peaks_init[0] - search_pix): int(peaks_init[0] + search_pix)]
    
    peaks2 = np.where(neighbor_search2 == np.amax(neighbor_search2))
    peaks1 = np.where(neighbor_search1 == np.amax(neighbor_search1))
    
    peaks2 = [peaks2[1][0], peaks2[0][0]]
    peaks1 = [peaks1[1][0], peaks1[0][0]]
    
    peaks2 = np.array(peaks2)
    peaks1 = np.array(peaks1)
    
    disp = peaks2 - peaks1
    
    tmp = list(disp)
    
    if abs(tmp[0]) > 10:
        tmp[0] = 0
    if abs(tmp[1]) > 10:
        tmp[1] = 0
        
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
cap = cv2.VideoCapture('C:/Users/vargh/Desktop/IMG_2509.MOV')

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
#    print(iteration,timer()-start) # prints time (for debugging/optimization purposes)
    cv2.imshow('frame2',res)
    cv2.imshow('frame3', im)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the capture object

vid.release()

# Destroy all the windows



    
    

