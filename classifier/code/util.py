

import pickle
import numpy as np


def save(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

        
def load(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)

    return obj


def print_scores(scores, label=None):
    if label:
        print("Accuracy: %0.2f (+/- %0.4f) [%s]" % (scores.mean(), 
            scores.std(), label))
    else:
        print("Accuracy: %0.2f (+/- %0.4f)" % (scores.mean(), scores.std()))

