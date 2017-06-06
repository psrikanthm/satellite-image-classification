import numpy as np
import process_y as py
import cv2
import os

from constants import *

(names, _) = py.get_data()

X = []
for name in names:
    print "Read Image TIF: ", name
    img = cv2.imread(TRAIN_TIF + name +'.tif', cv2.IMREAD_UNCHANGED)
    X.append(img)
X = np.array(X)
np.save(XTRAIN_TIF, X)

X = []
for name in names:
    print "Read Image JPG: ", name
    img = cv2.imread(TRAIN_JPG + name +'.jpg', cv2.IMREAD_UNCHANGED)
    X.append(img)
X = np.array(X)
np.save(XTRAIN_JPG, X)

X = []
filenames = []
for f in sorted(os.listdir(TEST_TIF)):
    filename = os.path.splitext(f)[0]
    print "Read Image TIF_TEST: ", filename
    img = cv2.imread(TEST_TIF + filename + '.tif', cv2.IMREAD_UNCHANGED)
    X.append(img)
    filenames.append(filename)
X = np.array(X)
np.save(XTEST_TIF, X)
np.save(XTEST_FILES, filenames)

X = []
for filename in filenames:
    print "Read Image JPG_TEST: ", filename
    img = cv2.imread(TEST_JPG + filename + '.jpg', cv2.IMREAD_UNCHANGED)
    X.append(img)
np.save(XTEST_JPG, X)
