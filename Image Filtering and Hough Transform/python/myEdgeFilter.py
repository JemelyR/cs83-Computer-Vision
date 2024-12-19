##
#JemelyR
#24W, cs83
#PA1
#
#edge filter
##


import numpy as np
from scipy import signal    # For signal.gaussian function

from myImageFilter import myImageFilter

def myEdgeFilter(img0, sigma):
    # YOUR CODE HERE
    #kernel to smooth
    hSize = int(2 * np.ceil(3 * sigma) + 1)
    
    kernel = signal.gaussian(hSize, sigma).reshape(hSize, 1)#gaussian kernel
    kernel = np.dot(kernel, kernel.T)

    #smoothing with previous function
    smoothed = myImageFilter(img0, kernel)
    
    #sobel filters
    sobx = np.array([[-1, 0, 1], 
                        [-2, 0, 2], 
                        [-1, 0, 1]])
    
    soby = np.array([[1, 2, 1], 
                        [0, 0, 0], 
                        [-1, -2, -1]])

      
    #image gradient y&x
    imgx = myImageFilter(smoothed, sobx)
    imgy = myImageFilter(smoothed, soby)
    
    
    magnitude = np.sqrt(imgx**2 + imgy**2) #gradient magnitude
    
    edge_angle = np.arctan2(imgy, imgx) * 180 / np.pi #edge angle
    edge_angle[edge_angle < 0] += 180

    rounded_angle = np.round(edge_angle / 45) * 45
    
    #map 180 back to 0
    rounded_angle[rounded_angle == 180] = 0  

    img1 = np.zeros_like(magnitude)
    
    #supression
    for i in range(1, magnitude.shape[0] - 1):
        for j in range(1, magnitude.shape[1] - 1):
            # neighbouring pixels
            angle = rounded_angle[i, j] #gradient angle
            if angle == 0 or angle == 180:
                n1 = magnitude[i, j+1] #first neighbor
                n2 = magnitude[i, j-1] #second neighbor
            elif angle == 45:
                n1 = magnitude[i+1, j-1]
                n2 = magnitude[i-1, j+1]
            elif angle == 90:
                n1 = magnitude[i+1, j]
                n2 = magnitude[i-1, j]
            elif angle == 135:
                n1 = magnitude[i-1, j-1]
                n2 = magnitude[i+1, j+1]

            # Suppress non-maximum edges
            if magnitude[i, j] >= n1 and magnitude[i, j] >= n2:
                img1[i, j] = magnitude[i, j]

    return img1