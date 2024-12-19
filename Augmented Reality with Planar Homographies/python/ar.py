##
#Jemely Robles
#PA2, cs83, 24W
#ar
##


import numpy as np
import cv2
from loadVid import loadVid
from matchPics import matchPics
from planarH import computeH_ransac, compositeH

movie = cv2.VideoCapture('../data/ar_source.mov')
book = cv2.VideoCapture('../data/book.mov')

frameW = int(book.get(3))
frameH = int(book.get(4))

# Use 'MJPG' for AVI format with Motion JPEG codec
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
# Change the output file extension to .avi
out = cv2.VideoWriter('../results/ar.avi', fourcc, 24, (frameW, frameH))

cover = cv2.imread('../data/cv_cover.jpg')

while True:
    retMovie, movieFrame = movie.read()
    retBook, bookFrame = book.read()
    #disregard
    if not retMovie or not retBook:
        break
    #read
    if cover is not None:
      #without black edges
        trimmed = movieFrame[50:movieFrame.shape[0]-50, :]
        #resize video
        newWidth = int(np.rint((cover.shape[0] / trimmed.shape[0]) * trimmed.shape[1]))
        newHeight = int(np.rint((cover.shape[0] / trimmed.shape[0]) * trimmed.shape[0]))
        scaled = cv2.resize(trimmed, (newWidth, newHeight))
        #slice middle of video
        ratio1 = int(np.rint(cover.shape[1] / 2))
        ratio2 = int(np.rint(scaled.shape[1] / 2))
        midSection = scaled[:newHeight, ratio2 - ratio1:ratio2 + ratio1, :]

        matches, locs1, locs2 = matchPics(bookFrame, cover)
        #put it all together
        if matches.shape[0] < 4:
            continue
        H2to1, inliers = computeH_ransac(np.take(locs1, matches[:, 0], axis=0), np.take(locs2, matches[:, 1], axis=0))
        composite_img = compositeH(H2to1, bookFrame, midSection)

        out.write(composite_img)

movie.release()
book.release()
out.release()

print("video done")