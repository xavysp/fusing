
#  This coding is only for test bloke of codes

import numpy as np
import cmath


def computeDff(input):
    n = len(input)
    output =[np.complex(0)]*n
    for k in range(n):
        s = np.complex(0)
        for t in range(n):
            exponential =np.exp(-(np.complex(2))*np.pi * t * k / n)
            s += input[t]*exponential
            # print s
        output[k] = s

    return output


dates = [1, 2, 3, 4]

print (computeDff(dates))
