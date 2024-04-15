import numpy as np
import copy
import matplotlib.pyplot as plt
import scipy
import h5py
from PIL import Image
from h5data import load_dataset
from LogisticRegression import LogReg
from scipy import ndimage

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes= load_dataset()