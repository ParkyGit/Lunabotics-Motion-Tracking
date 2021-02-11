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

def calcdisp(frame1, frame2):       
    detected_shift = cv2.phaseCorrelate(frame2, frame1)
    tmp = [detected_shift[0][0], detected_shift[0][1]] # store x,y displacement
    return tmp

vid = cv2.VideoCapture(0)

iteration = 0
prevFrame = 0
dstep = np.empty([1,2])
totD= np.array([[0,0]])
time = [0]
start = timer()

#gpuframe1 = cv2.cuda_GpuMat()
#gpuframe2 = cv2.cuda_GpuMat()
currentLoc = turtle.Turtle()


while(True):
    # Capture the video frame

    # by frame

    ret, frame = vid.read()
   
    frameShape = frame.shape #?
 

    # Display the resulting frame
    im = frame
    
    im = cv2.resize(im, None, fx=0.05, fy=0.05)
       
    bw_img = cv2.cvtColor(np.float32(im), cv2.COLOR_RGB2GRAY)
#    gpuframe1.upload(bw_img)
    
    if iteration > 3:
        calcdisp(bw_img, prevFrame)
    if iteration > 15:
        tmp = calcdisp(bw_img, prevFrame)
        dstep = np.vstack((dstep, tmp))
        time = np.vstack((time, timer()-start))
        
        
        totD = np.vstack((totD, np.sum(dstep, axis=0)))
        
        currentLoc.sety(totD[iteration-15, 1]*10)
        currentLoc.setx(totD[iteration-15, 0]*10)
        
    prevFrame = bw_img
#    gpuframe2.upload(prevFrame)
    iteration = iteration + 1
    print(iteration,timer()-start)
    
    

   
   

# After the loop release the cap object



vid.release()

# Destroy all the windows



    
    

