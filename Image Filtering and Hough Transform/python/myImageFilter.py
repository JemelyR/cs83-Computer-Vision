##
#JemelyR
#24W, cs83
#PA1
#
#imagefilter
##

import numpy as np

def myImageFilter(img0, h):
    # YOUR CODE HERE
    print("running image filter")
#     print(h)
    imgH, imgW = img0.shape #height and width of image
    fH, fW = h.shape #height and width of filter

    #padding
    pH = fH // 2 #padding for height
    pW = fW // 2 #padding for width

    #padding image
    padded = np.pad(img0, ((pH, pH), (pW, pW)), mode='edge')

    img1 = np.zeros_like(img0)
    #reference for np.zeros_like: https://numpy.org/doc/stable/reference/generated/numpy.zeros_like.html
    

    #convolutiopn
    for i in range(fH):
        for j in range(fW):
            img1 += h[i, j] * padded[i:i + imgH, j:j + imgW]

    return img1


