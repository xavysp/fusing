
#  This coding is only for test bloke of codes

import numpy as np
from scipy.fftpack import fft, ifft
import cv2


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
    pi = np.pi
    row, col, depth = np.shape(img)
    F = np.zeros((row,col,depth))
    s = np.complex(0)
    for d in range(depth):
        for i in range(row):
            for j in range(col):
                s = np.complex(0)
                for i1 in range(row):
                    for j1 in range(col):
                        s += img[i1][j1][d]* np.exp(-(np.complex(1))*2*pi*(i*i1+j*j1)/(row*col))
                F[i][j][d] = s
    cv2.imshow('Fourier Image', F)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




# read data
img = cv2.imread('dataset/1826i.bmp')

data1 = [1, 2, 3, 4]
data2 = [2, 4, 6, 8]

print (computeDft(data1, data2))

myFt(img)

