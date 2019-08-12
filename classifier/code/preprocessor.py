

import re
import string
import numpy as np


def tokenize(url, include_separators=False):

    """
    Split url by separators and return as list. Separators are included
    by default.
    """

    seps = string.punctuation
    
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

