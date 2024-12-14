###
#
#Jemely Robles
#cs83, 24W
#PA4
#LucasKanade_Robust.py
#
###

import numpy as np
from scipy.interpolate import RectBivariateSpline

############################################
#extra credit 4.1
##################

def LucasKanadeRobust(It, It1, rect, p=np.zeros(2)):
    # Input: 
    #   It: template image
    #   It1: Current image
    #   rect: Current position of the object
    # Output:
    #   p: movement vector [dx, dy]

    c = 4.685
    threshold = 0.01875
    maxIters = 100
    x1, y1, x2, y2 = rect

    #grid for interpolation
    x = np.arange(x1, x2 + 1)
    y = np.arange(y1, y2 + 1)
    X, Y = np.meshgrid(x, y)

    #interpolation
    interpolateIt = RectBivariateSpline(np.arange(It.shape[0]), np.arange(It.shape[1]), It)
    interpolateIt1 = RectBivariateSpline(np.arange(It1.shape[0]), np.arange(It1.shape[1]), It1)

    for i in range(maxIters):
        #warp It1 to template 
        warped = interpolateIt1.ev(Y + p[1], X + p[0])

        #error
        error = interpolateIt.ev(Y, X) - warped

        #gradients
        xIt1 = interpolateIt1.ev(Y + p[1], X + p[0], dx=1)
        yIt1 = interpolateIt1.ev(Y + p[1], X + p[0], dy=1)

        #jacobian
        gMatrix = np.vstack((xIt1.ravel(), yIt1.ravel())).T

        #tukey weights
        weights = (1 - (error.flatten()**2 / c**2)**2)**2 * (np.abs(error.flatten()) <= c)

        gWeighted = gMatrix * weights[:, np.newaxis]
        errorWeighted = error.flatten() * weights

        #hessian
        H = gWeighted.T @ gWeighted

        dp = np.linalg.lstsq(H, gWeighted.T @ errorWeighted, rcond=None)[0]

        #update p
        p += dp

        #convergence
        if np.linalg.norm(dp) < threshold:
            break

    return p