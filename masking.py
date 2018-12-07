# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 11:58:09 2018

@author: Ashe
"""
import numpy as np
import cv2

def main():
    
    
    path = "C:\\Users\\Ashe\\Desktop\\Books\\Semester project\\min\\"
    imgpath = path + "k_mean.tiff"
    
    img = cv2.imread(imgpath, 1)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    sensitivity = 15;
    lower_green = np.array([55 - sensitivity, 40, 40])
    upper_green = np.array([55 + sensitivity, 255, 255])
    
    mask = cv2.inRange(hsv,  lower_green,  upper_green)
    mask_inv = cv2.bitwise_not(mask)
    
    res = cv2.bitwise_and(img, img, mask = mask_inv) 
    
    
    cv2.imwrite("C:\\Users\\Ashe\\Desktop\\Books\\Semester project\\min\\masked.tiff", res)
    
    cv2.imshow('fram', img)
    cv2.imshow('mask_inv', mask_inv)
    cv2.imshow('res', res)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
        
if __name__ == "__main__":
    main()
    

