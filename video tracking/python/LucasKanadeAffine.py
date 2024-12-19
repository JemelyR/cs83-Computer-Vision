###
#
#Jemely Robles
#cs83, 24W
#PA4
#LucasKanadeAffine.py
#
###

import numpy as np
from scipy.interpolate import RectBivariateSpline

def LucasKanadeAffine(It, It1, rect):
    # Input: 
    #   It: template image
    #   It1: Current image
    #   rect: Current position of the object
    #   (top left, bot right coordinates: x1, y1, x2, y2)
    # Output:
    #   M: the Affine warp matrix [2x3 numpy array]

    # set up the threshold
    threshold = 0.01875
    maxIters = 100
    p = np.zeros((6,1))
    x1,y1,x2,y2 = rect

    # put your implementation here   
    
    
    #interpolate
    interlopeIt = RectBivariateSpline(np.arange(It.shape[0]), np.arange(It.shape[1]), It)
    interlopeIt1 = RectBivariateSpline(np.arange(It.shape[0]), np.arange(It.shape[1]), It1)
    
    #template coords
    Y, X = np.mgrid[y1:y2+1, x1:x2+1]
    
    flatX = X.flatten()
    flatY = Y.flatten()
    Ones = np.ones_like(flatX)
    # Zeros = np.zeros_like(flatX)
    
    template = interlopeIt.ev(Y, X)

    for i in range(maxIters):
        #affine transformation M from p
        M = np.array([[1 + p[0, 0], p[1, 0], p[2, 0]], [p[3, 0], 1 + p[4, 0], p[5, 0]]])
        
        #put transform M on coordinates
        newCoords = M @ np.vstack((flatX, flatY, Ones))
        warped = interlopeIt1.ev(newCoords[1], newCoords[0])

        #error
        errorImg = template.flatten() - warped
        
        #curr image gradiants
        xGradiant = interlopeIt1.ev(newCoords[1], newCoords[0], dx=0, dy=1)
        yGradiant = interlopeIt1.ev(newCoords[1], newCoords[0], dx=1, dy=0)

        #jacobian
        J = np.vstack((flatX * xGradiant, flatY * xGradiant, xGradiant, flatX * yGradiant, flatY * yGradiant, yGradiant)).T
        
        #hessian
        H = J.T @ J
        
        dp, _, _, _ = np.linalg.lstsq(H, J.T @ errorImg, rcond=None)
        
        dp = dp.reshape(-1, 1)
        
        #update p with dp
        p += dp
        
        #convergence
        if np.linalg.norm(dp) < threshold:
            break
            
    M = np.array([[1 + p[0, 0], p[1, 0], p[2, 0]], [p[3, 0], 1 + p[4, 0], p[5, 0]]])
    
    return M