
#  This coding is only for test bloke of codes

import numpy as np
from scipy.fftpack import fft, ifft


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


data1 = [1, 2, 3, 4]
data2 = [2, 4, 6, 8]

print (computeDft(data1, data2))


