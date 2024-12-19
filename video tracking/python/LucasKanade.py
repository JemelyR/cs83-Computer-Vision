###
#
#Jemely Robles
#cs83, 24W
#PA4
#LucasKanade.py
#
###

import numpy as np
from scipy.interpolate import RectBivariateSpline
#using equation 4 & 6:
#∆p∗ = H−1JT b,
#L = EX [T (x) − I (W (x; p))]2 

def LucasKanade(It, It1, rect):
    # Input: 
    #   It: template image
    #   It1: Current image
    #   rect: Current position of the object
    #   (top left, bot right coordinates: x1, y1, x2, y2)
    # Output:
    #   p: movement vector dx, dy
    
    # set up the threshold
    threshold = 0.01875
    maxIters = 100
    p = np.zeros(2)          
    x1,y1,x2,y2 = rect

    # put your implementation here
    x = np.arange(x1, x2+1)
    y = np.arange(y1, y2+1)
    X, Y = np.meshgrid(x, y)
    
    interlopeIt = RectBivariateSpline(np.arange(It.shape[0]), np.arange(It.shape[1]), It)
    interlopeIt1 = RectBivariateSpline(np.arange(It.shape[0]), np.arange(It.shape[1]), It1)

    rectH = int(round(x2-x1))
    rectW = int(round(y2-y1))

    gMatrix = np.zeros((rectW * rectH, 2))

    x = np.linspace(x1, x2, rectW)
    y = np.linspace(y1, y2, rectH)
    X, Y = np.meshgrid(x, y)

    error = interlopeIt.ev(Y, X)
    
    for i in range(maxIters):
    #estimates of p
        dx, dy = p
#         print("dx, dy", dx, dy)

        #warp It1 susing current estimates
        warped = interlopeIt1.ev(Y+dy, X+dx)

        errorImage = error - warped

        #gradiants
        gMatrix[:, 1] = interlopeIt1.ev(Y+dy, X+dx, dx=1).flatten()
        gMatrix[:, 0] = interlopeIt1.ev(Y+dy, X+dx, dy=1).flatten()

        #vectorize gradiants and error image
        gMatrix[:, 1] = gMatrix[:, 1].ravel()
        gMatrix[:, 0] = gMatrix[:, 0].ravel()
        b = errorImage.ravel()

        #make H matrix and solve for dp
        #∆p∗ = H−1JT b
        H = gMatrix.T @ gMatrix
        #solve A*dp = b
        dp, _, _, _ = np.linalg.lstsq(H, gMatrix.T @ b, rcond=None)  

        #update p with dp
        p += dp

        #convergence
        if np.linalg.norm(dp) < threshold:
            break

    return p


