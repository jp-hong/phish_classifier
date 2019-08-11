

import re
import string
from urllib.parse import urlparse, quote
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


def char_level_encoder(url):
    if len(url) > 128:
        url = url[:128]
    
    vect = list(map(ord, list(url)))
    
    return np.array(vect)

