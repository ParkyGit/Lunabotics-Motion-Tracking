
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
import time as t


# Takes two frames, calculates the phase cross correlation between them and outputs displacement
def calcdisp(frame1, frame2):       
    frame_1DFT = cv2.dft(frame1) # phase cross correlation algorithm
    frame_2DFT = cv2.dft(frame2)
    return frame_1DFT, frame_2DFT

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
t.sleep(10)

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
   
    
    frame1_DFT, frame2_DFT = calcdisp(bw_img, prevFrame) # raw displacement data
    time = np.vstack((time, timer()-start)) # stacks the time data
        
        

        
        # updates turtle
        
    prevFrame = bw_img # sets current frame as previous frame
#    gpuframe2.upload(prevFrame)
    iteration = iteration + 1 #increases iteration 
    
    print(iteration,timer()-start) # prints time (for debugging/optimization purposes)
    cv2.imshow('frame',res)
    cv2.imshow('frame1', frame1_DFT)
    cv2.imshow('frame2', frame2_DFT)
    if cv2.waitKey(1000) & 0xFF == ord('q'):
        break

# After the loop release the capture object

vid.release()

# Destroy all the windows



    
    

