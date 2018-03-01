import os

import numpy as np

import keras
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Conv2D, BatchNormalization, Dropout, Flatten
from keras.layers import Activation, Reshape, Conv2DTranspose, UpSampling2D
from keras.optimizers import RMSprop

import pandas as pd 
import matplotlib
from matplotlib import pyplot as plt
matplotlib.interactive(True)

input_images = "./data/car.npy"
data = np.load(input_images)
data = data/255
img_w, img_h = 28, 28
data = np.reshape(data, [data.shape[0], img_w, img_h, 1])
print(data.shape)

plt.imshow(data[0,:,:,0], cmap="Greys")

input("Press Enter to exit")