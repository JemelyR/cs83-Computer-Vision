##
#JemelyR
#24W, cs83
#PA1
#
#hough tranform
##

import numpy as np

def myHoughTransform(Im, rhoRes, thetaRes):
    # YOUR CODE HERE
    imgH, imgW = Im.shape #Im height and width
    
    M = np.sqrt(imgH**2 + imgW**2)#needs to be large enought to accomidate possible lines
    rhoS = np.arange(0, M, rhoRes)
    thetaS = np.arange(0, 2*np.pi, thetaRes)
    
    #contains "votes"
    img_hough = np.zeros((len(rhoS), len(thetaS)))

    #edge indices
    edges = np.transpose(np.nonzero(Im))

    #rho for each edge index and theta value
    cos = np.cos(thetaS)
    sin = np.sin(thetaS)
    rho = np.rint(np.matmul(edges, np.array([sin, cos]))).astype(int)
    #rint ref:https://numpy.org/doc/stable/reference/generated/numpy.rint.html

    #hough space
    img_hough, rhoScale, thetaScale = np.histogram2d(rho.ravel(), np.tile(thetaS, rho.shape[0]),
                                           bins=[rhoS,thetaS], range=[[0, M],[0, 2*np.pi]])
    #histogram reference: https://numpy.org/doc/stable/reference/generated/numpy.histogram.html

    return img_hough, rhoScale, thetaScale

    
# plt.imshow(img_hough, cmap='ah', extent=[thetaScale.min(), thetaScale.max(), rhoScale.min(), rhoScale.max()],
#            aspect='auto', origin='lower')
# plt.xlabel('Theta (radians)')
# plt.ylabel('Rho (pixels)')
# plt.title('Hough Transform')
# plt.colorbar(label='Votes')
# plt.show()