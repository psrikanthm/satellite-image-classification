import numpy as np
import cv2
import os

###################################
#### Constants ###################

TEST_TIFF = "../data/test-tif-v2/"
TEST_JPG = "../data/test-jpg/"
##################################

X = []
filenames = []

for filename in sorted(os.listdir(TEST_TIFF)):
    print "Read Image: ", filename
    img = cv2.imread(TEST_TIFF + filename, cv2.IMREAD_UNCHANGED)
    X.append(img)
    filenames.append(filename)

X = np.array(X)

np.save('../data/x_test_tif.npy', X)
np.save('../data/test_filenames.npy', filenames)

X = []

for filename in sorted(os.listdir(TEST_JPG)):
    print "Read Image: ", filename
    img = cv2.imread(TEST_JPG + filename, cv2.IMREAD_UNCHANGED)
    X.append(img)

np.save('../data/x_test_jpg.npy', X)
