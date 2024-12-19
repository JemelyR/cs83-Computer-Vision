##
#Jemely Robles
#PA2, cs83, 24W
#HarryPotterize
##

import numpy as np
import cv2
import skimage.io 
import skimage.color
#Import necessary functions
from planarH import compositeH, computeH_ransac
from matchPics import matchPics

#Write script for Q3.9
#read images
cover = cv2.imread('../data/cv_cover.jpg')
desk = cv2.imread('../data/cv_desk.png')
harry = cv2.imread('../data/hp_cover.jpg')
#homography
matches, locs1, locs2 = matchPics(desk, cover)
points1 = locs1[matches[:, 0]]
points2 = locs2[matches[:, 1]]
H2to1, inliers = computeH_ransac(points1, points2)
#wrap to dimension of desk book
warped = cv2.warpPerspective(harry, H2to1, (desk.shape[1], desk.shape[0]))
#have to resize for it to fit
resized = cv2.resize(harry, (cover.shape[1], cover.shape[0]))
#compose
composite_img = compositeH(H2to1, desk, resized)

cv2.imwrite('../results/HarryPotter.jpg', composite_img)
print("Harry Potter Done")
