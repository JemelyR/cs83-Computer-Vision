##
#Jemely Robles
#PA2, cs83, 24W
#planarH
##

import numpy as np
import cv2
import math
import random


def computeH(x1, x2):
    #Q3.6
    #Compute the homography between two sets of points
    N = x1.shape[0]
    A = np.zeros((2*N, 9))
    
    for i in range(N):
        x, y = x2[i]
        xPrime, yPrime = x1[i]
        A[2*i, :] = [-x, -y, -1, 0, 0, 0, xPrime*x, xPrime*y, xPrime]
        A[2*i+1, :] = [0, 0, 0, -x, -y, -1, yPrime*x, yPrime*y, yPrime]
        
    U, Sum, V = np.linalg.svd(A)
    H2to1 = V[-1].reshape((3, 3))
    
    return H2to1


def computeH_norm(x1, x2):
    #Q3.7
    #Compute the centroid of the points
    x1Centroid = np.mean(x1, axis=0)
    x2Centroid = np.mean(x2, axis=0)
    #Shift the origin of the points to the centroid
    x1Shift = x1 - x1Centroid
    x2Shift = x2 - x2Centroid
    #Normalize the points so that the largest distance from the origin is equal to sqrt(2)
    x1Distance = np.linalg.norm(x1Shift, axis=1)
    x2Distance = np.linalg.norm(x2Shift, axis=1)
    x1Scale = np.sqrt(2) / np.max(x1Distance)
    x2Scale = np.sqrt(2) / np.max(x2Distance)
    x1Norm = x1Shift * x1Scale
    x2Norm = x2Shift * x2Scale
    #Similarity transform 1
    T1 = np.array([[x1Scale, 0, -x1Scale * x1Centroid[0]],
                  [0, x1Scale, -x1Scale * x1Centroid[1]],
                  [0, 0, 1]])
    #Similarity transform 2
    T2 = np.array([[x2Scale, 0, -x2Scale * x2Centroid[0]],
                  [0, x2Scale, -x2Scale * x2Centroid[1]],
                  [0, 0, 1]])
    #Compute homography
    H = computeH(x1Norm, x2Norm)
    #Denormalization
    H2to1 = np.linalg.inv(T1) @ H @ T2 
    
    return H2to1

def computeH_ransac(x1, x2):
    # Q3.8
    # Compute the best fitting homography given a list of matching points
    bestH2to1 = None
    inliers = np.array([])

    x1 = x1[:, ::-1]
    x2 = x2[:, ::-1]
    maxCount = 0

    for i in range(1000):
        #indices
        random = np.random.permutation(len(x1))
        selected = random[:4]
        #grab the sample from the indices
        x1Sample = x1[selected, :]
        x2Sample = x2[selected, :]
        #use compute_norm
        H2to1 = computeH_norm(x1Sample, x2Sample)
        x2Homogeneous = np.concatenate([x2.T, np.ones((1, x2.shape[0]))], axis=0)
        #scoring the inliners 
        x2Score = np.dot(H2to1, x2Homogeneous)
        x2Score[2, x2Score[2] == 0] = np.finfo(float).eps
        x2Score /= x2Score[2]
        xsScore = x2Score[0] / x2Score[2]
        ysScore = x2Score[1] / x2Score[2]
        differences = (x1[:, 0] - xsScore)**2 + (x1[:, 1] - ysScore)**2
        distances = np.sqrt(differences)

        fit = np.where(distances < 10, 1, 0)
        #get best inliers
        if np.sum(fit) > maxCount:
            maxCount = np.sum(fit)
            inliers = fit
            bestH2to1 = H2to1
            
    points = np.flatnonzero(inliers)
    x1Points = x1[points]
    x2Points = x2[points]
    bestH2to1 = computeH_norm(x1Points, x2Points)

    return bestH2to1, inliers


def compositeH(H2to1, template, img):
    #Create a composite image after warping the template image on top
    #of the image using the homography
    #Note that the homography we compute is from the image to the template;
    #x_template = H2to1*x_photo
    #For warping the template to the image, we need to invert it.
    #Create mask of same size as template
    templateMask = np.zeros(img.shape[:2], np.uint8)  
    templateMask[:] = 1  
    #Warp mask by appropriate homography
    tempValues = (template.shape[1], template.shape[0])
    #Warp template by appropriate homography
    warpMask = cv2.warpPerspective(templateMask, H2to1, tempValues)
    warpImg = cv2.warpPerspective(img, H2to1, tempValues)
    #Use mask to combine the warped template and the image
    composite_img = np.zeros(template.shape, np.uint8)
    composite_img[warpMask == 1] = warpImg[warpMask == 1]
    composite_img[warpMask == 0] = template[warpMask == 0]

    return composite_img

print("done H")


