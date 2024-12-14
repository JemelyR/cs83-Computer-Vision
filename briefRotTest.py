##
#Jemely Robles
#PA2, cs83, 24W
#briefRotTest
##


import numpy as np
import cv2
from matchPics import matchPics
from scipy import ndimage
import matplotlib.pyplot as plt
from helper import plotMatches
import os

#Q3.5
#Read the image and convert to grayscale, if necessary
path = '../data/cv_cover.jpg'
img = cv2.imread(path)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(img.shape)
degrees = []
matchCount = []
for i in range(36):
    #Rotate Image
    rotated = ndimage.rotate(img, angle=i*10)
    #Compute features, descriptors and Match features
    mathces, locs1, locs2 = matchPics(img, rotated)
    #Update histogram
    degrees.append(i*10)
    matchCount.append(len(mathces))
    if i == 9 : #90 degrees
        plotMatches(img, rotated, mathces, locs1, locs2)
    elif i == 20: #200 degrees
        plotMatches(img, rotated, mathces, locs1, locs2)
    elif i == 34: #340 degrees
        plotMatches(img, rotated, mathces, locs1, locs2)
#Display histogram
plt.bar(degrees, matchCount, width = 6)
plt.title('Histogram')
plt.show()