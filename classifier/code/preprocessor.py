

import re
import string
import numpy as np
from numba import jit


def tokenize(url, seps=None, include_separators=False):

    """
    Split url by separators and return as list. Separators are included
    by default.
    """
    
    if seps == None:
        seps = string.punctuation
    else:
        seps = seps
    
    if include_separators:
        return list(filter(None, re.split("([" + seps + "])", url)))

    return list(filter(None, re.split("[" + seps + "]", url)))


def char_level_encoder(url, ndim=128, pad=True):
    if len(url) > ndim:
        url = url[:ndim]
    
    vect = list(map(ord, list(url)))

    if pad and len(vect) < ndim:
        vect += [0] * (ndim - len(vect))

    return np.array(vect)


@jit(nopython=True)
def char_onehot(char_array, unique_chars, n_unique):
    zlen = char_array.shape[0]
    ylen = char_array.shape[1]
    xlen = n_unique

    arr = np.zeros(shape=(zlen, ylen * xlen))

    for i in range(zlen):
        for j in range(ylen):
            for k in range(xlen):
                if char_array[i][j] == unique_chars[k]:
                    xidx = k
                    break
                    
            arr[i][ylen * j + xidx] = 1
            
    return arr

