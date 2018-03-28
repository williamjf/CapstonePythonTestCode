# -*- coding: utf-8 -*-
"""
Created on Thu Feb 01 18:43:14 2018

@author: William
"""
import numpy as np
import cv2
import time

start_frame = 1000
area_search_scalar = 1

font = cv2.FONT_HERSHEY_SIMPLEX #Setup font for printing text

cap = cv2.VideoCapture('red_roomba_test_vid.mp4') #Setup video read
cap.set(1, start_frame) #Set starting frame of video

#Setup video captures for bounding box videos
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter('output.mp4',fourcc, 30.0, (1280,720))

#fourcc2 = cv2.VideoWriter_fourcc(*'XVID')
#out2 = cv2.VideoWriter('output_closed.mp4',fourcc, 30.0, (1280,720))

#Setup kernals for morphological closing, long for shadow
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
kernel_black = cv2.getStructuringElement(cv2.MORPH_RECT,(21,3))
kernel_canny = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))

#read in the first frame to size image
ret, frame = cap.read()
#get image sizes
img_height,img_width = frame.shape[:2]

#Initialize FPS counters
last_time = time.time()
ret,last_frame = cap.read()
fps = 0

while(cap.isOpened()):
    robot_count = 0
    ret, frame = cap.read()
    #Reset frame position if the end of file is reached
    if (not ret): 
       cap.set(1, start_frame);
       continue
    #Show raw frame and make a copy for bounding rectangles
    #cv2.imshow('frame',frame)
    bounded = frame.copy();
    #Blur to reduce noise
    blurred = cv2.GaussianBlur(frame,(5,5),0)
    
    #Change color space to HSV for later filtering 
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    clone = hsv

    #Mask red portions of image
    red_masked = cv2.inRange(hsv,np.array([160,50,100],dtype = "uint8"),np.array([173,255,255],dtype = "uint8"))
    final_masked = red_masked
    
    #Morphological close to reduce noise from color mask, opening didnt seem to help much
    closing = cv2.morphologyEx(final_masked, cv2.MORPH_CLOSE, kernel, iterations = 1)
    #opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    
    #Edge detection to reduce work of findContours function,
    #closed after to remove 1 pixel holes in edge
    filtered = cv2.Canny(closing, 2, 5)  
    filtered = cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, kernel_canny, iterations = 1)  

    image, contours, hierarchy = cv2.findContours(filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #For each read contour
    for contour in contours:    
        area = cv2.contourArea(contour)
        #Ignore areas too small or too large
        if (area > 30 and area < 5000): #needs to be function of y position of countour
            #Get position and shape of each contour
            x, y, w, h = cv2.boundingRect(contour)
            #Rudimentary filter based on position in image,
            #area large near the base, small near the top
            if(area < y*3 and area > y/4):
                #Draw bounding rectangle
                bounded = cv2.rectangle(bounded,(x,y),(x+w,y+h), (255,0,0),1)
                #Setup parameters for cropped image into the next layer
                #Ensures no coordinate is out of image bounds
                min_y = y+h
                if min_y >= img_height:
                    min_y = y
                max_y = y+(area_search_scalar*4+1)*h
                if max_y > img_height:
                    max_y = img_height
                min_x = x-area_search_scalar*w
                if min_x < 1:
                    min_x = 1
                max_x = x+(area_search_scalar+1)*w
                if max_x > img_width:
                    max_x = img_width-1
                #print(min_y,max_y,min_x,max_x)
                
                #Create copy of cropped image in brg so it can be shown later
                brg_robot = frame[min_y:max_y,min_x:max_x]
                #Create hsv of cropped image for processing
                hsv_robot = clone[min_y:max_y,min_x:max_x]
                #cv2.destroyWindow('robot')
                #cv2.imshow("robot",brg_robot) 
                #key = cv2.waitKey(0)
                
                #Apply black mask and elongated closing to cropped image
                black_masked = cv2.inRange(hsv_robot,np.array([80,50,40],dtype = "uint8"),np.array([150,255,60],dtype = "uint8"))
                cv2.imshow('bk',black_masked);
                #cv2.waitKey(0);
                black_masked = cv2.morphologyEx(black_masked, cv2.MORPH_CLOSE,kernel_black,iterations = 1)

                #Find black contours
                image, b_contours, hierarchy = cv2.findContours(black_masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                #Limit number of black contours near red contours to reduce noise
                if(len(b_contours) < 3):
                    #For each black contour
                    for b_contour in b_contours:
                        #Filter out too small and too large areas
                        area = cv2.contourArea(b_contour)
                        if (area > 10 and area < 1000):
                            #Find and draw bounding rectangles of black areas
                            b_x, b_y, b_w, b_h = cv2.boundingRect(b_contour)
                            bounded = cv2.rectangle(bounded,(b_x+min_x,b_y+min_y),(b_x+min_x+b_w,b_y+min_y+b_h), (0,255,0),1)         
                            
                            #Filter to only dark areas below red areas
                            if(b_y+min_y > y & (10 < b_y+min_y - y < 200)):  
                                #Crop image to area between black and red contour
                                brg_robot = frame[y+h:b_y+min_y+b_h,x:x+w]
                                hsv_robot = clone[y+h:b_y+min_y+b_h,x:x+w]
                                #cv2.destroyWindow('robot2')
                                #cv2.imshow("robot2",brg_robot)
                                #key = cv2.waitKey(0)
                                
                                #Mask second cropped frame for white, close
                                white_masked = cv2.inRange(hsv_robot,np.array([60,10,150],dtype = "uint8"),np.array([120,150,255],dtype = "uint8"))
                                white_masked = cv2.morphologyEx(white_masked, cv2.MORPH_CLOSE,kernel,iterations = 2)
                                image, w_contours, hierarchy = cv2.findContours(white_masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                
                                #For each white contour
                                #This layer honestly could be better. Basically just checks if anything is their at all
                                for w_contour in w_contours:
                                    area = cv2.contourArea(w_contour)
                                    #Filter area sizes
                                    if (area > 10 and area < 1000):
                                        #Accumulate robot count
                                        robot_count += 1
                                        #Number white areas
                                        cv2.putText(bounded,str(robot_count),(x,y),font,0.5,(0,0,255),1,cv2.LINE_AA)
                                        w_x, w_y, w_w, w_h = cv2.boundingRect(w_contour)
                                    #ratio = w_w/w_h
                                     #   if(ratio > 0.5 or ratio < 3):
                                        bounded = cv2.rectangle(bounded,(x+w_x,y+h+w_y),(x+w_x+w_w,y+h+w_y+w_h), (0,0,255),5)  
                                        #key = cv2.waitKey(0)                                      
                                        #cv2.imshow('dst',brg_robot)
                                        #cv2.imshow("bounded",bounded)
                                        #cv2.imshow("bgr_robot",brg_robot)
                                        #cv2.waitKey(0)
                                        #cv2.destroyWindow("bgr_robot")
    
    #current_time = time.time()
    #fps = 1/(current_time - last_time)*.2+.8*fps
    #last_time = current_time
    #cv2.putText(bounded,str(int(fps)),(10,100), font, 4,(255,255,255),2,cv2.LINE_AA)
    cv2.putText(bounded,str(robot_count),(10,500), font, 4,(0,0,127*robot_count),2,cv2.LINE_AA)

    cv2.imshow('bounded',bounded)
    final = bounded
    #out.write(final)
    #out2.write(cv2.cvtColor(closing, cv2.COLOR_GRAY2BGR))

    #cv2.imshow('edgy',filtered)
    #cv2.imshow('frame',frame)
    #cv2.imshow('post_morph',opening)
    #cv2.imshow('closing',closing)        

    #Set waitKey(1) for continuous play
    key = cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #Press p for previous frame, q to quit, any other key to continue
    if key & 0xFF== ord('p'):
        current_frame = cap.get(1)
        cap.set(1, current_frame-2)
        continue
    elif key & 0xFF == ord('q'):
        break

cap.release()
#out.release()
#out2.release()
cv2.destroyAllWindows()
