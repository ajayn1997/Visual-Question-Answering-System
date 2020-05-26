import numpy as np
from keras.utils.np_utils import to_categorical
import json
import h5py
import os
from constants import *

def right_align(seq, lengths):
    v = np.zeros(np.shape(seq))
    N = np.shape(seq)[1]
    for i in range(np.shape(seq)[0]):
        v[i][N - lengths[i]:N] = seq[i][0: lengths[i]]
    return v
