###
#
#Jemely Robles
#cs83, 24W
#PA4
#InverseCompositionAffine.py
#
###

import numpy as np
from numpy.linalg import inv
from scipy.interpolate import RectBivariateSpline

def InverseCompositionAffine(It, It1, rect):
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
    #identity
    M = np.array([[1, 0, 0], [0, 1, 0]])  
    p = M.flatten()
    
    
    #interpolate
    interlopeIt = RectBivariateSpline(np.arange(It.shape[0]), np.arange(It.shape[1]), It)
    interlopeIt1 = RectBivariateSpline(np.arange(It.shape[0]), np.arange(It.shape[1]), It1)

    X, Y = np.meshgrid(np.arange(It.shape[1]), np.arange(It.shape[0]))
    
    xGradiant = interlopeIt.ev(Y, X, dy=1)
    yGradiant = interlopeIt.ev(Y, X, dx=1)

    #simplify computing
    J = np.stack([xGradiant * X, xGradiant * Y, xGradiant, yGradiant * X, yGradiant * Y, yGradiant], axis=-1)

    
    for i in range(maxIters):
        xTemplate = p[0] * X + p[1] * Y + p[2]
        yTemplate = p[3] * X + p[4] * Y + p[5]
        patch = np.all([x1 <= xTemplate, xTemplate < x2, y1 <= yTemplate, yTemplate < y2], axis=0)

        #flatten with patch
        flatJ = J.reshape(-1, 6) 
        flatPatch = patch.ravel()

        #index to flattened with flatPatch 
        xTemplate = xTemplate.flatten()[flatPatch]
        yTemplate = yTemplate.flatten()[flatPatch]
        jPatch = flatJ[flatPatch]

        H = np.linalg.inv(jPatch.T @ jPatch + np.eye(6) * 1e-5)
        warped = interlopeIt1.ev(yTemplate, xTemplate)
        error = (warped - It[patch]).flatten()
        warp = jPatch.T @ error
        dp = H @ warp

        M = np.vstack((np.array(p[:6]).reshape(2, 3), [0, 0, 1]))
        dM = np.vstack((np.array(dp[:6]).reshape(2, 3), [0, 0, 1]))
        dM[0, 0] += 1
        dM[1, 1] += 1
        M = M @ np.linalg.inv(dM)
        p = M[:2, :].ravel()

        if threshold > (np.linalg.norm(dp)) * (np.linalg.norm(dp)):
            break

    M = np.array([[p[0], p[1], p[2]], [p[3], p[4], p[5]]]).reshape(2, 3)

    return M
