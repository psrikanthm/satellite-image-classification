import numpy as np

import simple_arch
import vgg

from keras.optimizers import SGD

############ Constants ##########
TRAIN_X_CL = "../data/x_train_cl.npy"
TRAIN_Y_CL = "../data/y_train_cl.npy"
TEST_X = "../data/x_test.npy"

TRAIN_X_WR = "../data/x_train_weather.npy"
TRAIN_Y_WR = "../data/y_train_weather.npy"
#################################

XTRAIN_CL = np.load(TRAIN_X_CL)
YTRAIN_CL = np.load(TRAIN_Y_CL)

XTRAIN_WR = np.load(TRAIN_X_WR)
YTRAIN_WR = np.load(TRAIN_Y_WR)

XTEST = np.load(TEST_X)


_,img_width, img_height,_ = XTRAIN_CL.shape
_, classes = YTRAIN_CL.shape
model = vgg.get_model(img_width, img_height, classes)
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy')
model.fit(XTRAIN_CL, YTRAIN_CL, epochs=5, verbose = 1, validation_split = 0.1)
YTEST_CL = model.predict(XTEST)
np.save('../data/y_test_cl.npy', YTEST_CL)

_,img_width, img_height,_ = XTRAIN_WR.shape
_, classes = YTRAIN_WR.shape
model = vgg.get_model(img_width, img_height, classes)
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy')
model.fit(XTRAIN_WR, YTRAIN_WR, epochs=5, verbose = 1, validation_split = 0.1)
YTEST_WR = model.predict(XTEST)
np.save('../data/y_test_weather.npy', YTEST_WR)
