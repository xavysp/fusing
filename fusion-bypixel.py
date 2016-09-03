# before to start you have to check
import cv2
import matplotlib
matplotlib.use('GTKAgg')  # IF you do't use this line cv2.imshow will give you an error
# from matplotlib import pyplot
import numpy as np
import pywt
import xlwt
# import time


# *1*  Image fusion using laplacian pyramid (low-pass filter)
def laplacian_pyramid_fusion(img1, img2):
    # generating Gaussian pyra mid for visible image
    gImgVl = img1.copy()
    gpImgVl = [gImgVl]  # gpA
    for i in xrange(6):
        gImgVl = cv2.pyrDown(gpImgVl[i])
        gpImgVl.append(gImgVl)
    # generating Gaussian pyramid for nir image
    gImgNir = img2.copy()
    gpImgNir = [gImgNir]  # gpB
    for i in xrange(6):
        gImgNir = cv2.pyrDown(gpImgNir[i])
        gpImgNir.append(gImgNir)
        # just for me
        cv2.imshow('image pyramid:', gpImgNir[i])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

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
    print lpImgNir[0].shape
    for i in xrange(5, 0, -1):
        size = (gpImgNir[i - 1].shape[1], gpImgNir[i - 1].shape[0])
        GE = cv2.pyrUp(gpImgNir[i], dstsize=size)
        L = cv2.subtract(gpImgNir[i-1], GE)
        print 'L: ', L.shape
        lpImgNir.append(L)
        # just for me
        cv2.imshow('Laplacian pyramid:', lpImgNir[5-i])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # ****** Now add left and right halves of image in each level ******
    # the function np.hstack blend and split two images
    LS = []
    o=0
    for lv, ln in zip(lpImgVl, lpImgNir):
        rows, cols, dpt = lv.shape
        ls = lv
        # ls = np.hstack((lv[:, 0:cols/2], ln[:, cols/2:]))
        # ls = np.hstack((lv[:, 0:cols], ln[:, cols:]))
        for k in xrange(dpt):
            for i in xrange(rows):
                for j in xrange(cols):
                    if (lv[i][j][k]) < ln[i][j][k]:
                        ls[i][j][k] = ln[i][j][k]
        LS.append(ls)
        # just for me
        cv2.imshow('Fused images:', LS[o])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        o += 1
    # now reconstruct I cannot understand how works this
    ls_ = LS[0]
    for i in xrange(1, 6):
        print 'this: ', LS[i].shape[1], LS[i].shape[0]
        size = (LS[i].shape[1], LS[i].shape[0])
        ls_ = cv2.pyrUp(ls_, dstsize=size)
        ls_ = cv2.add(ls_, LS[i])
    # image with direct connection each half
    real = np.hstack((img1[:, :cols], img2[:, cols:]))
    print np.shape(ls_)
    cv2.imshow('image fused', ls_)
    cv2.imshow('image natural', real)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# *2*  Function that fuse tho image with contrast pyramid fusion method
# The tittle of the article is : Merging thermal and visual images by a contrast pyramid
# SOme errors are bellow
def contrast_pyramid(img1, img2):
    # w = [[0.2, 0.5, 0.4, 0.5, 0.2], [0.5, 0.25, 0.2, 0.25, 0.5],
    #      [0.4, 0.2, 0.16, 0.2, 0.4], [0.5, 0.25, 0.2, 0.25, 0.5],
    #      [.02, 0.5, 0.4, 0.5, 0.2]]  # weighting function
    # Gaussian pyramid for visual image
    gImgVl= img1.copy()
    gpImgVl = [gImgVl]
    for i in xrange(6):

        gImgVl = cv2.pyrDown(gpImgVl[i])
        gpImgVl.append(gImgVl)
    gImgNir = img2.copy()
    gpImgNir = [gImgNir]
    # Gaussian pyramid for infrared image
    for i in xrange(6):
        gImgNir = cv2.pyrDown(gpImgNir[i])
        gpImgNir.append(gImgNir)
    # laplacion pyramid VIS
    lpImgVl = [gpImgVl[5]]
    for i in xrange(5, 0, -1):
        size = (gpImgVl[i - 1].shape[1], gpImgVl[i - 1].shape[0])
        GE = cv2.pyrUp(gpImgVl[i], dstsize=size)
        # L = cv2.subtract(gpImgVl[i - 1], GE)
        # the line bellow is the highlight of this fusion method
        # page 3 from Merging thermal and visual image by a contrast pyramid
        # article
        L = (GE / gpImgVl[i-1])
        lpImgVl.append(L)

    # laplacian pyramid NIR
    lpImgNir = [gpImgNir[5]]
    for i in xrange(5, 0, -1):
        size = (gpImgNir[i - 1].shape[1], gpImgNir[i - 1].shape[0])
        GE = cv2.pyrUp(gpImgNir[i], dstsize=size)
        # L = cv2.subtract(gpImgNir[i - 1], GE)
        L = GE / (gpImgNir[i - 1])
        lpImgNir.append(L)
    # Fusion algorithm
    LS = []
    o = 5
    for lv, ln in zip(lpImgVl, lpImgNir):
        rows, cols, dpt = lv.shape
        ls = lv
        # ls = np.hstack((lv[:, 0:cols/2], ln[:, cols/2:]))
        # ls = np.hstack((lv[:, 0:cols], ln[:, cols:]))
        for k in xrange(dpt):
            for i in xrange(rows):
                for j in xrange(cols):
                    if np.abs((lv[i][j][k])-1) > np.abs((ln[i][j][k])-1):
                        ls[i][j][k] = lv[i][j][k]
                    else:
                        ls[i][j][k] = ln[i][j][k]
        LS.append(ls)
        # cv2.imshow('only see', LS[o])
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # o += 1
    # for composite image
    ls_ = LS[0]
    for i in xrange(1, 6):
        print 'this: ', LS[i].shape[1], LS[i].shape[0]
        size = (LS[i].shape[1], LS[i].shape[0])
        ls_ = cv2.pyrUp(ls_, dstsize=size)
        # ls_ = cv2.multiply(LS[i], ls_)
        ls_ = cv2.add(ls_, LS[i])

    # ls_ = cv2.multiply(gpImgVl[0], ls_)
    print np.shape(ls_)
    cv2.imshow('image fused', ls_)
    cv2.imshow('image natural', img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # cv2.imwrite('fuse.jpg', ls_)


# *3**  Function that hierarchical based image fusion to fuse tho kind of images
# The article is: hierarchical image fusion
def hierarchical_fusion(img1, img2):
    pass


# *4** Function for image fusion based in wavelet transform
# You can find the algorithm in: Multisensor image fusion using the wavelet transform
def wavelet_transform_fusion(img1, img2):
    imgVl = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    ImgNir = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    imgVlLh = np.zeros((np.shape(imgVl)))
    row, col = np.shape(imgVlLh)
    for i in range(row):
        for j in range(0, col, 4):
            if j+4 <= col:
                imgVlLh[i][j] = imgVl[i][j]+imgVl[i][j+1]
                imgVlLh[i][j+1] = imgVl[i][j+2]+imgVl[i][j+3]
                imgVlLh[i][j+2] = imgVl[i][j] - imgVl[i][j + 1]
                imgVlLh[i][j + 3] = imgVl[i][j + 2] - imgVl[i][j + 3]

    cv2.imshow('Horizontal wavelet image', imgVlLh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# *5** Function that can fuse with pywt and image
def waveletTransformFunction(img1, img2):
    # whole of the wavelet families: ['haar', 'db', 'sym', 'coif', 'bior', 'rbio', 'dmey']
    imgVl = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    wImgV= pywt.dwt2(imgVl,'haar')
    cAv, (cHv, cVv, cDv) = wImgV
    print (imgVl.shape)
    imgIr = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    wImgIr = pywt.dwt2(imgNir, 'haar')
    cAi, (cHi, cVi, cDi) = wImgIr
    # cv2.imwrite('temporal.png', cAv)
    # cv2.imshow('cAv', cAv)
    # cv2.imshow('cHv', cHv)
    # cv2.imshow('cVv', cVv)
    # cv2.imshow('cDv', cDv)
    # cv2.imshow('imgVl', imgVl)
    # cAv1 = np.int16(cAv)
    # cHv1 = np.int16(cHv)
    # cVv1 = np.int16(cVv)
    # cDv1 = np.int16(cDv)
    # np.savetxt('texcAv.txt', cAv1)
    # np.savetxt('texcHv.txt', cHv)
    # print (cHv1[16:20][:18])
    # a short code for pyramid
    imgVlPy = []
    imgVlPy = cv2.pyrDown(imgVl,)
    #print (imgVlPy[16:20][:18])
    cv2.imshow('pyramid image', imgVlPy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    a = imgVlPy[0:19][0:17]
    print (a)
    print a.shape


############################################
# Imput data #
imgVl = cv2.imread('dataset/1826v.bmp')
imgNir = cv2.imread('dataset/1826i.bmp')
print imgVl.shape, imgNir.shape

#####################################
# Results #
print 'Please choose the fusion method which you want to test:'
print '1: for Laplacian pyramid based fusion'
print '2: for Contrast pyramid based fusion here is the error'
print ('3: For hierarchical based fusion')
print ('4: Wavelet transform based fusion ')
print ('5: Wavelet transform based fusion using Pywavelet')
a = input('Choose a number:')
if a == 1:
    laplacian_pyramid_fusion(imgVl, imgNir)
elif a == 2:
    contrast_pyramid(imgVl, imgNir)
elif a == 3:
    print('Sorry we are working')
elif a == 4:
    wavelet_transform_fusion(imgVl, imgNir)
elif a == 5:
    waveletTransformFunction(imgVl, imgNir)
