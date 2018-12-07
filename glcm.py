# -*- coding: utf-8 -*-

import numpy as np
import cv2
from skimage.feature import greycomatrix, greycoprops




def main():
    
    path = "C:\\Users\\Ashe\\Desktop\\Books\\Semester project\\min\\"
    imgpath = path + "masked.tiff"
    img = cv2.imread(imgpath, 1) 
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_arr = np.array(img_gray)
    #cv2.imshow("tray", img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
    gCoMat = greycomatrix(img_arr, [2], [0], 256,symmetric=True, normed=True)
    contrast = greycoprops(gCoMat, prop='contrast')
    dissimilarity = greycoprops(gCoMat, prop='dissimilarity')
    homogeneity = greycoprops(gCoMat, prop='homogeneity')
    energy = greycoprops(gCoMat, prop='energy')
    correlation = greycoprops(gCoMat, prop='correlation')
    #cluster_shade = greycoprops(gCoMat, prop='cluster shade')
    #cluster_prominence = greycoprops(gCoMat, prop=' cluster prominence')
    
    print('contrast ='+str(contrast[0][0]))
    print('dissimilarity ='+str(dissimilarity[0][0]))
    print('homogeneity ='+str(homogeneity[0][0]))
    print('energy ='+str(energy[0][0]))
    print('correlation ='+str(correlation[0][0]))

    
if __name__ == "__main__":
    main()
    