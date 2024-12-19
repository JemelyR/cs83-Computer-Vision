##
#JemelyR
#24W, cs83
#PA1
#
#hough lines
##

import numpy as np
import cv2  # For cv2.dilate function

def myHoughLines(H, nLines):
    # YOUR CODE HERE
    H = H.astype(np.float32)

    kernel_size = (3, 3)  # kernel size for dilation
    kernel = np.ones(kernel_size, np.float32)
    
    #supress nonmaximal cells
    dilated = cv2.dilate(H, kernel)

    H[H != dilated] = 0
#     H = np.where(dilated > 0, 0, dilated)

    #for rho and theta parameters
    rhos = np.zeros(nLines, dtype=int)
    thetas = np.zeros(nLines, dtype=int)

    #peaks
    for i in range(nLines):
        maX = np.argmax(H) #nLines maxes 
        #argmax ref: https://numpy.org/doc/stable/reference/generated/numpy.argmax.html]
        rhoMax, thetaMax = np.unravel_index(maX, H.shape)
        rhos[i] = rhoMax
        thetas[i] = thetaMax
        H[rhoMax, thetaMax] = 0

    return rhos, thetas


