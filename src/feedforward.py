from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K

import numpy as np

def get_model(img_height, img_width, classes):

    model = Sequential()
    model.add(Dense(1000, input_dim = img_width))
    model.add(Activation('relu'))
    model.add(Dense(1000))
    model.add(Activation('relu'))
    model.add(Dense(1000))
    model.add(Activation('relu'))
    model.add(Dense(classes))
    model.add(Activation('softmax'))

    return model
