
import cv2
import matplotlib
matplotlib.use('GTKAgg')  # IF you do't use this line cv2.imshow will give you an error
import numpy as np
# import time

# ni = [[[1, 2, 3], [1, 0, 2]],[[1, 2, 2, 4, 0, 1], [1, 2, 0, 1, 2, 2],
#                               [1, 1, 2, 2, 0, 1],[0, 0, 1, 1, 2, 1]]]
# vl =[[[1, 1, 1], [2, 2, 2]],[[1, 0, 1, 0, 1, 0], [1, 1, 1, 1, 1, 1],
#                              [2, 2, 2, 2, 2, 2], [0, 0, 0, 0, 0, 0]]]
#
# for lv, ln in zip(vl, ni):
#     rows, cols, dpt = lv.shape
#     print lv.shape
#     print rows, cols, dpt
#     # ls = np.hstack((lv[:, 0:cols/2], ln[:, cols/2:]))
#     ls = np.hstack((lv[:, 0:cols], ln[:, cols:]))
#     LS.append(ls)
imgVl = cv2.imread('1826v.bmp')
imgNir = cv2.imread('1826i.bmp')
gImgVl = imgVl.copy()
gpImgVl = [gImgVl]  # gpA
for i in xrange(6):
    gImgVl = cv2.pyrDown(gpImgVl[i])
    gpImgVl.append(gImgVl)
# generating Gaussian pyramid for nir image
gImgNir = imgNir.copy()
gpImgNir = [gImgNir]  # gpB
for i in xrange(6):
    gImgNir = cv2.pyrDown(gpImgNir[i])
    gpImgNir.append(gImgNir)
# generate laplacian pyramid for visible image
lpImgVl = [gpImgVl[5]]
for i in xrange(5, 0, -1):
    # GE = cv2.pyrUp(gpImgVl[i])
    size = (gpImgVl[i-1].shape[1], gpImgVl[i-1].shape[0])
    GE = cv2.pyrUp(gpImgVl[i], dstsize=size)
    L = cv2.subtract(gpImgVl[i-1], GE)
    lpImgVl.append(L)
# generate laplacian pyramid for nir image
lpImgNir = [gpImgNir[5]]
for i in xrange(5, 0, -1):
    size = (gpImgNir[i - 1].shape[1], gpImgNir[i - 1].shape[0])
    GE = cv2.pyrUp(gpImgNir[i], dstsize=size)
    L = cv2.subtract(gpImgNir[i-1], GE)
    lpImgNir.append(L)
# ****** Now add left and right halves of image in each level ******
# the function np.hstack blend and split two images
LS = []
ls = []

for lv, ln in zip(lpImgVl, lpImgNir):
    rows, cols, dpt = lv.shape
    ls = lv
    # rows2, cols2, dpt2 = ln.shape
    # print rows, cols, dpt
    # print rows2, cols2, dpt2
    # print lv, '  ', ln
    # time.sleep(10)
    # ls = np.hstack((lv[:, 0:cols], ln[:, cols:]))
    for k in xrange(dpt):
        for i in xrange(rows):
            for j in xrange(cols):
                if (lv[i][j][k]) < ln[i][j][k]:
                    ls[i][j][k] = ln[i][j][k]

    # print ls
    LS.append(ls)
ls_ = LS[0]
for i in xrange(1, 6):
    size = (LS[i].shape[1], LS[i].shape[0])
    ls_ = cv2.pyrUp(ls_, dstsize=size)
    ls_ = cv2.add(ls_, LS[i])
print np.shape(ls_)
cv2.imshow('image fused', ls_)
cv2.imshow('image natural', imgVl)
cv2.waitKey(0)
cv2.destroyAllWindows()
