# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 23:03:21 2018

@author: Ashe
"""

import numpy as np
import cv2


def main():
    
   path = "C:\\Users\\Ashe\\Desktop\\Books\\Semester project\\min\\"
   imgpath = path + "1.13.jpg"
   
   img = cv2.imread(imgpath,1)
   
   r = 512.0 / img.shape[1]
   dim = (512, int(img.shape[0] * r))
   
   img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
   #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

   
   z=img.reshape((-1,3))
   z=np.float32(z)
    
   criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    
   k=3
   ret, lebel, center = cv2.kmeans(z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
   center = np.uint8(center)
   res = center[lebel.flatten()]
    
   output = res.reshape((img.shape))

   cv2.imwrite("C:\\Users\\Ashe\\Desktop\\Books\\Semester project\\min\\k_mean.tiff", output)      
        
   for i in range(1):
        
        cv2.imshow('clusterd', output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
if __name__ == "__main__":
    main()
    
    
    
     
    
    