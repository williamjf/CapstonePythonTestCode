# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 18:31:04 2018

@author: William
"""

import cv2
import time
import numpy as np

def showPixelValue(event,x,y,flags,param):
    global frame, combinedResult, placeholder
    
    if event == cv2.EVENT_MOUSEMOVE:
        # get the value of pixel from the location of mouse in (x,y)
        bgr = frame[y,x]

        # Convert the BGR pixel into other colro formats
        ycb = cv2.cvtColor(np.uint8([[bgr]]),cv2.COLOR_BGR2YCrCb)[0][0]
        lab = cv2.cvtColor(np.uint8([[bgr]]),cv2.COLOR_BGR2Lab)[0][0]
        hsv = cv2.cvtColor(np.uint8([[bgr]]),cv2.COLOR_BGR2HSV)[0][0]
        
        # Create an empty placeholder for displaying the values
        placeholder = np.zeros((frame.shape[0],400,3),dtype=np.uint8)

        # fill the placeholder with the values of color spaces
        cv2.putText(placeholder, "BGR {}".format(bgr), (20, 70), cv2.FONT_HERSHEY_COMPLEX, .9, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(placeholder, "HSV {}".format(hsv), (20, 140), cv2.FONT_HERSHEY_COMPLEX, .9, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(placeholder, "YCrCb {}".format(ycb), (20, 210), cv2.FONT_HERSHEY_COMPLEX, .9, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(placeholder, "LAB {}".format(lab), (20, 280), cv2.FONT_HERSHEY_COMPLEX, .9, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(placeholder, "Frame: {}".format(current_frame), (20, 350), cv2.FONT_HERSHEY_COMPLEX, .9, (255,255,255), 1, cv2.LINE_AA)
        
        # Combine the two results to show side by side in a single image
        cv2.imshow('hsv',placeholder)
    elif event == cv2.EVENT_LBUTTONDOWN:
        frame = cv2.circle(frame,(x,y),5,(0,255,0))
        cv2.imshow("N for next, P for Previous",frame)
        
if __name__ is '__main__':
    global frame
    start_frame = 1000
    
    cap = cv2.VideoCapture('red_roomba_test_vid.mp4') #Setup video read
    cap.set(1, start_frame) #Set starting frame of video
    ret,frame = cap.read()
    cv2.imshow("N for next, P for Previous",frame)
    global current_frame
    current_frame = cap.get(1)
    
    #cv2.waitKey(0)
    cv2.setMouseCallback("N for next, P for Previous",showPixelValue)
    
    while(cap.isOpened()):
        key = cv2.waitKey(0) & 0xFF
        #cv2.destroyAllWindows()
        #Press p for previous frame, q to quit, any other key to continue
        if key == ord('n'):
            current_frame += 1
            ret,frame = cap.read()
            cv2.imshow("N for next, P for Previous",frame)
        elif key == ord('p'):
            current_frame = cap.get(1)
            current_frame -= 1
            cap.set(1, current_frame)
            ret, frame = cap.read()
            cv2.imshow("N for next, P for previous", frame)
            continue
        elif key == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()