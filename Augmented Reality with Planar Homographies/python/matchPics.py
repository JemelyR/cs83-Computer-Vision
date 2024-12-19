##
#Jemely Robles
#PA2, cs83, 24W
#matchPics
##


import numpy as np
import cv2
import skimage.color
from helper import briefMatch
from helper import computeBrief
from helper import corner_detection
from helper import plotMatches

def matchPics(I1, I2):
    sigma = 0.15
    ratio = 0.65
    #I1, I2 : Images to match
    #Convert Images to GrayScale
    grayI1 = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)
    grayI2 = cv2.cvtColor(I2, cv2.COLOR_BGR2GRAY)
    #Detect Features in Both Images
    locsI1 = corner_detection(grayI1, sigma)
    locsI2 = corner_detection(grayI2, sigma)
    #Obtain descriptors for the computed feature locations
    desc1, locs1 = computeBrief(grayI1, locsI1)
    desc2, locs2 = computeBrief(grayI2, locsI2)
    #Match features using the descriptors
    matches = briefMatch(desc1, desc2, ratio)
#     plotMatches(I1, I2, matches, locs1, locs2)

    return matches, locs1, locs2

print("done")