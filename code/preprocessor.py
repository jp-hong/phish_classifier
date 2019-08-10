

import re
import string
from urllib.parse import urlparse, quote
import numpy as np


def clean(url):

    """
    Return url after cleaning. Will remove 'http://', 'https://', 'www.',
    and any trailing '/'. Will also percent encode any non-ascii
    characters
    """

    p = "(https|http)?(://)?(www2?\.)?"
    temp = re.sub(p, "", url).rstrip("\/")
    return quote(temp, safe=string.printable)


def tokenize(url, include_separators=False):

    """
    Split url by separators and return as list. Separators are included
    by default.
    """

    seps = string.punctuation
    
    if include_separators:
        return list(filter(None, re.split(r"([" + seps + r"])", url)))

    return list(filter(None, re.split(r"[" + seps + r"]", url)))

def char_level_encoder(s):
    c_dict = {string.printable[i]:i for i in range(len(string.printable))}
    vect = np.array([0] * 128)
    
    for i in range(min(128, len(s))):
        vect[i] = c_dict[s[i]]
        
    return vect

def word_level_encoder(s):
    pass

