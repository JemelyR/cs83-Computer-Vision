###
#
#Jemely Robles
#cs83, 24W
#submission.py
#
###


"""
Homework 5
Submission Functions
"""

# import packages here
import numpy as np
import helper as hlp
import scipy.signal
from skimage import color


"""
Q3.1.1 Eight Point Algorithm
       [I] pts1, points in image 1 (Nx2 matrix)
           pts2, points in image 2 (Nx2 matrix)
           M, scalar value computed as max(H1,W1)
       [O] F, the fundamental matrix (3x3 matrix)
"""
def eight_point(pts1, pts2, M):
    #normalize points by M using T
    T = np.array([[1/M, 0, 0], [0, 1/M, 0], [0, 0, 1]])
    ones = np.ones((pts1.shape[0], 1))
    normP1 = np.hstack([pts1, ones]) @ T.T
    normP2 = np.hstack([pts2, ones]) @ T.T

    #make A matrix
    A = np.zeros((normP1.shape[0], 9))
    for i in range(normP1.shape[0]):
        x, y, _ = normP2[i, :]
        xPrime, yPrime, _ = normP1[i, :]
        A[i] = [x*xPrime, x*yPrime, x, y*xPrime, y*yPrime, y, xPrime, yPrime, 1]

    #singular value decomp of A
    U, Sum, V = np.linalg.svd(A)
    F = V[-1].reshape((3, 3))

    # rank2 constraint on F
    U2, Sum2, V2 = np.linalg.svd(F)
    Sum2[2] = 0  
    fPrime= U2 @ np.diag(Sum2) @ V2

    #refine f
    refineF = hlp.refineF(fPrime, normP1[:, :2], normP2[:, :2])

    # unscale F
    final = T.T @ refineF @ T


    return final


"""
Q3.1.2 Epipolar Correspondences
       [I] im1, image 1 (H1xW1 matrix)
           im2, image 2 (H2xW2 matrix)
           F, fundamental matrix from image 1 to image 2 (3x3 matrix)
           pts1, points in image 1 (Nx2 matrix)
       [O] pts2, points in image 2 (Nx2 matrix)
"""
def epipolar_correspondences(im1, im2, F, pts1):

    im1_gray = color.rgb2gray(im1)
    im2_gray = color.rgb2gray(im2)
    
    # window size
    w = 25
    half_w = w // 2
    
    #store matching poitns
    pts2 = []  

    for pt1 in pts1:
        pt1Homogenous = np.array([pt1[0], pt1[1], 1])
        
        #line in second image
        l = np.dot(F, pt1Homogenous)
        
        # line params
        a, b, c = l
        
        candidates = [(x, int(-(a*x + c) / b)) for x in range(half_w, im2.shape[1] - half_w, 1)]
    
        bestScore = np.inf
        bestPoint = None
        
        for x, y in candidates:
            #if window is in image bounds
            if y - half_w < 0 or y + half_w >= im2.shape[0]:
                continue
            
            # get patches to compare
            patch1 = im1_gray[int(pt1[1])-half_w:int(pt1[1])+half_w+1, int(pt1[0])-half_w:int(pt1[0])+half_w+1]
            patch2 = im2_gray[y-half_w:y+half_w+1, x-half_w:x+half_w+1]
            
            # euclidean distance
            score = np.linalg.norm(patch1 - patch2)
            
            # update best matches
            if score < bestScore:
                bestScore = score
                bestPoint = (x, y)

        pts2.append(bestPoint)
    
    return np.array(pts2)

"""
Q3.1.3 Essential Matrix
       [I] F, the fundamental matrix (3x3 matrix)
           K1, camera matrix 1 (3x3 matrix)
           K2, camera matrix 2 (3x3 matrix)
       [O] E, the essential matrix (3x3 matrix)
"""
def essential_matrix(F, K1, K2):
    # replace pass by your implementation
    E = K2.T @ F @ K1
    return E

"""
Q3.1.4 Triangulation
       [I] P1, camera projection matrix 1 (3x4 matrix)
           pts1, points in image 1 (Nx2 matrix)
           P2, camera projection matrix 2 (3x4 matrix)
           pts2, points in image 2 (Nx2 matrix)
       [O] pts3d, 3D points in space (Nx3 matrix)
"""

def triangulate(P1, pts1, P2, pts2):
    #pairs
    N = pts1.shape[0]  
    pts3DHomogenous = np.zeros((N, 4))  

    for i in range(N):
        #A matrix for each pair
        A = np.zeros((4, 4))
        A[0] = pts1[i, 0] * P1[2] - P1[0]
        A[1] = pts1[i, 1] * P1[2] - P1[1]
        A[2] = pts2[i, 0] * P2[2] - P2[0]
        A[3] = pts2[i, 1] * P2[2] - P2[1]

        #svd
        U, Sum, V = np.linalg.svd(A)
        x = V[-1] 

        #normalize
        pts3DHomogenous[i] = x / x[3] 

    #grab 3D points
    pts3d = pts3DHomogenous[:, :3]

    #reprojection error for pts1
    pts1ProjH = P1 @ pts3DHomogenous.T
    pts1Projected = (pts1ProjH[:2] / pts1ProjH[2]).T
    reprojErrorPts1 = np.linalg.norm(pts1 - pts1Projected, axis=1).mean()

    #reprojection error for pts2
    pts2ProjH = P2 @ pts3DHomogenous.T
    pts2Projected = (pts2ProjH[:2] / pts2ProjH[2]).T
    reprojErrorPts2 = np.linalg.norm(pts2 - pts2Projected, axis=1).mean()

    #reprojection error mean
    reprojErrorMean = (reprojErrorPts1 + reprojErrorPts2) / 2
    print("Mean reprojection error:", reprojErrorMean)

    return pts3d



"""
Q3.2.1 Image Rectification
       [I] K1 K2, camera matrices (3x3 matrix)
           R1 R2, rotation matrices (3x3 matrix)
           t1 t2, translation vectors (3x1 matrix)
       [O] M1 M2, rectification matrices (3x3 matrix)
           K1p K2p, rectified camera matrices (3x3 matrix)
           R1p R2p, rectified rotation matrices (3x3 matrix)
           t1p t2p, rectified translation vectors (3x1 matrix)
"""
def rectify_pair(K1, K2, R1, R2, t1, t2):
    c1 = -(np.linalg.inv(K1 @ R1) @ (K1 @ t1))
    c2 = -(np.linalg.inv(K2 @ R2) @ (K2 @ t2))

    r1 = ((c1 - c2) / np.linalg.norm(c1 - c2)).T
    r2 = np.cross(R1[2].reshape(3, 1).T, r1)
    r3 = np.cross(r2, r1)

    newAxis = np.array([r1.reshape(3,), r2.reshape(3,), r3.reshape(3,)])

    #new rotation matrices    
    R1p = newAxis
    R2p = newAxis
    # intrinsic params
    K1p = K2
    K2p = K2
    # new translation vectors
    t1p = -(newAxis @ c1)
    t2p = -(newAxis @ c2)

    #rectification matrices
    M1 = (K1p @ R1p) @ np.linalg.inv(K1 @ R1)
    M2 = (K2p @ R2p) @ np.linalg.inv(K2 @ R2)

    return M1, M2, K1p, K2p, R1p, R2p, t1p, t2p



"""
Q3.2.2 Disparity Map
       [I] im1, image 1 (H1xW1 matrix)
           im2, image 2 (H2xW2 matrix)
           maxDisp, scalar maximum disparity value
           win_size, scalar window size value
       [O] dispM, disparity map (H1xW1 matrix)
"""
def get_disparity(im1, im2, maxDisp, win_size):
    # replace pass by your implementation
    #grayScale
    if im1.ndim > 2:
        im1 = im1.mean(axis=2)
    if im2.ndim > 2:
        im2 = im2.mean(axis=2)

    #disparity inital
    dispMap = np.zeros_like(im1)

    #half window
    w = (win_size - 1) // 2

    #padding
    im1Padded = np.pad(im1, w, mode='constant', constant_values=0)
    im2Padded = np.pad(im2, w, mode='constant', constant_values=0)

    #go through each pixel
    for y in range(w, im1.shape[0] + w):
        for x in range(w, im1.shape[1] + w):
            minSsd = np.inf
            bestDisp = 0

            #search within disparity range
            for d in range(maxDisp + 1):
                #no out of bounds
                if x - d - w < 0:
                    continue

                #ssd
                im1Window = im1Padded[y-w:y+w+1, x-w:x+w+1]
                im2Window = im2Padded[y-w:y+w+1, x-d-w:x-d+w+1]
                ssd = np.sum((im1Window - im2Window) ** 2)

                #update
                if ssd < minSsd:
                    minSsd = ssd
                    bestDisp = d

            #best disparity goes to the disparity map
            dispMap[y-w, x-w] = bestDisp

    return dispMap


"""
Q3.2.3 Depth Map
       [I] dispM, disparity map (H1xW1 matrix)
           K1 K2, camera matrices (3x3 matrix)
           R1 R2, rotation matrices (3x3 matrix)
           t1 t2, translation vectors (3x1 matrix)
       [O] depthM, depth map (H1xW1 matrix)
"""
def get_depth(dispM, K1, K2, R1, R2, t1, t2):
    # replace pass by your implementation
    c1 = -np.linalg.inv(K1 @ R1) @ (K1 @ t1)
    c2 = -np.linalg.inv(K2 @ R2) @ (K2 @ t2)
    b = np.linalg.norm(c1 - c2)
    
    f = K1[0, 0]
    
    #depth map has same shape as disparity map
    depthMap = np.zeros_like(dispM, dtype=np.float32)
    
    mask = dispM > 0
    
    # get depth
    depthMap[mask] = b * f / dispM[mask]
    
    return depthMap


"""
Q3.3.1 Camera Matrix Estimation
       [I] x, 2D points (Nx2 matrix)
           X, 3D points (Nx3 matrix)
       [O] P, camera matrix (3x4 matrix)
"""


def estimate_pose(x, X):
    # replace pass by your implementation
    n = len(X)
    A = np.zeros((2 * n, 12))

    #fillinhg in S
    for i in range(n):
        A[2*i, :3] = X[i]
        A[2*i, 3] = 1
        A[2*i, 8:11] = -x[i, 0] * X[i]
        A[2*i, 11] = -x[i, 0]

        A[2*i + 1, 4:7] = X[i]
        A[2*i + 1, 7] = 1
        A[2*i + 1, 8:11] = -x[i, 1] * X[i]
        A[2*i + 1, 11] = -x[i, 1]

    #svd of A
    U, Sum, V = np.linalg.svd(A)
    P = V[-1].reshape((3, 4))

    return P

"""
Q3.3.2 Camera Parameter Estimation
       [I] P, camera matrix (3x4 matrix)
       [O] K, camera intrinsics (3x3 matrix)
           R, camera extrinsics rotation (3x3 matrix)
           t, camera extrinsics translation (3x1 matrix)
"""
def estimate_params(P):
    
    #svd give center
    U, Sum, V = np.linalg.svd(P)
    c = V[-1, :3] / V[-1, -1]  
    
    #instrinsic k and rotation r using QR decomp
    M = P[:, :3]  
    K, R = scipy.linalg.rq(M) 

    T = np.diag(np.sign(np.diag(K)))
    if np.linalg.det(R) < 0:
        #adjust
        R = -R
        K = -K
        
    K = np.dot(K, T)
    R = np.dot(T, R)

    #get translation
    t = -np.dot(R, c)

    return K, R, t