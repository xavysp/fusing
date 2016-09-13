
#  This coding is only for test bloke of codes

import cv2
from matplotlib import pyplot as plt
import numpy as np
# import imageio
# import rawpy


# Computing Discrete Fourier Transform
def computeDft(inreal, inimag):
    assert len(inreal) == len(inimag)
    n = len(inreal)
    outreal = [0.0] * n
    outimag = [0.0] * n
    for k in range(n):  # For each output element
        sumreal = 0.0
        sumimag = 0.0
        for t in range(n):  # For each input element
            angle = 2 * np.pi * t * k / n
            sumreal += inreal[t] * np.cos(angle) + inimag[t] * np.sin(angle)
            sumimag += -inreal[t] * np.sin(angle) + inimag[t] * np.cos(angle)
        outreal[k] = sumreal
        outimag[k] = sumimag
    return (outreal, outimag)


# Computing Fourier Transform by myself
def myFt(img):
    grayImg = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    pi = np.pi
    row, col = np.shape(grayImg)
    F = np.zeros((row,col))
    s = np.complex(0)
    for i in range(row):
        for j in range(col):
            s = np.complex(0)
            for i1 in range(row):
                for j1 in range(col):
                    expo = np.exp(-(np.complex(1))*2*pi*(i*i1+j*j1)/(row*col))
                    s += grayImg[i1][j1] * expo
            F[i][j] = s
    cv2.imshow('Fourier Image', F)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def histogramEqualization(imgI, imgV):
    imgEq = cv2.cvtColor(imgI, cv2.COLOR_RGB2GRAY)
    equ = cv2.equalizeHist(imgEq)
    res = np.hstack((imgEq, equ))  # stacking images side-by-side
    cv2.imwrite('res.png', equ)
    # pyramid image
    imgVlPy = cv2.pyrDown(imgV, )
    # cv2.imwrite('res-v.png', imgVlPy)


# read data
img = cv2.imread('dataset/1826i.bmp')
imgV = cv2.imread('exam_rgb.jpg')
imgI = cv2.imread('exam_ir.bmp')
data1 = [1, 2, 3, 4]
data2 = [2, 4, 6, 8]

# print (computeDft(data1, data2))
#
# myFt(img)
histogramEqualization(imgI, imgV)

# opencv fourier transform #

# grayImg = cv2.imread('dataset/1826i.bmp', 0)
# fImg = np.fft.fft2(grayImg)
# fImgShift = np.fft.fftshift(fImg)
# magnitude_spectrum = 20*np.log(np.abs(fImgShift))
#
# cv2.imshow('Fourier transform image', magnitude_spectrum)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



