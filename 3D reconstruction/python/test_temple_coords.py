###
#
#Jemely Robles
#cs83, 24W
#test_temple_coords.py
#
###

import numpy as np
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
import submission as sub
import helper as hlp 


#images and corresp
data = np.load('../data/some_corresp.npz')
im1 = plt.imread('../data/im1.png')
im2 = plt.imread('../data/im2.png')
pts1 = data['pts1']
pts2 = data['pts2']

#run eight point
M = max(im1.shape[0], im1.shape[1])
F = sub.eight_point(pts1, pts2, M)
# print("F")
# print(F)

#load points and run epipolar corresp
templeCoords = np.load('../data/temple_coords.npz')
pts1Temple = templeCoords['pts1']
pts2Temple = sub.epipolar_correspondences(im1, im2, F, pts1Temple)

#load instrinsics and compute E
intrinsics = np.load('../data/intrinsics.npz')
K1 = intrinsics['K1']
K2 = intrinsics['K2']
E = sub.essential_matrix(F, K1, K2)
# print("E")
# print(E)

#cam projection
P2Options = hlp.camera2(E)

I = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
P1 = K1 @ I
P2 = np.zeros((3, 4))


#run triangulate
bestCount = 0
bestP3D = None
for i in range(P2Options.shape[2]):
    P2Temp = K2 @ P2Options[:, :, i]
    pts3D = sub.triangulate(P1, pts1Temple, P2Temp, pts2Temple)
    posDepth = np.sum(pts3D[:, 2] > 0)
    
    if posDepth > bestCount:
        bestCount = posDepth
        P2 = P2Temp
        bestP3D = pts3D

#plot points
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pts3D[:, 0], pts3D[:, 1], pts3D[:, 2])
plt.show()

M2 = np.linalg.inv(K2) @ P2
R1 = np.eye(3)
t1 = np.zeros([3,1])
R2 = M2[:,:3]
t2 = M2[:,3:4]
np.savez('../data/extrinsics.npz', R1=R1, t1=t1,R2 = R2, t2 = t2)