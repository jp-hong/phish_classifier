

import pickle
import numpy as np


def save(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

        
def load(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)

    return obj

