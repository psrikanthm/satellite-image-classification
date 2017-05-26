import numpy as np
import simple_arch

import csv

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

ytrue, yval, y1 = simple_arch.run(XTRAIN_CL, YTRAIN_CL, XTEST)
_, _, y2 = simple_arch.run(XTRAIN_WR, YTRAIN_WR, XTEST)

np.save('../data/y_test_cl.npy', y1)
np.save('../data/y_test_weather.npy', y2)
np.save('../data/ytrue_cl.npy', ytrue)
np.save('../data/yval_cl.npy', yval)
