
# This code attempt make a multispectral image fusion based on wavelet
# transform

import cv2
import matplotlib
matplotlib.use('GTKAgg')  # IF you do't use this line cv2.imshow will give you an error
import numpy as np
import pywt
import matplotlib.pyplot as plt

def wtFusion(img1, img2):

    # generating Gaussian pyramid for visible image
    # gImgVl = img1.copy()
    # gpImgVl = [gImgVl]  # gpA
    # for i in xrange(6):
    #     gImgVl = cv2.pyrDown(gpImgVl[i])
    #     gpImgVl.append(gImgVl)
    # # generating Gaussian pyramid for nir image
    # gImgNir = img2.copy()
    # gpImgNir = [gImgNir]  # gpB
    # for i in xrange(6):
    #     gImgNir = cv2.pyrDown(gpImgNir[i])
    #     gpImgNir.append(gImgNir)
    wtImgVl = img1.copy()
    wtImgNir = img2.copy()







############################################
# Imput data #
imgVl = cv2.imread('1826v.bmp')
imgNir = cv2.imread('1826i.bmp')
print imgVl.shape, imgNir.shape
# cv2.imshow('image fused', imgVl)
# cv2.imshow('image natural', imgNir)
# cv2.waitKey(0)

#####################################
# Results #
wtFusion(imgNir, imgVl)
