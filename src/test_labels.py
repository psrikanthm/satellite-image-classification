import numpy as np
import cv2
import os

###################################
#### Constants ###################

TRAIN_JPG = "../data/train-jpg"
TRAIN_TIFF = "../data/train-tif-v2"
TEST_TIFF = "../data/test-tif-v2"
##################################

X = []

for filename in sorted(os.listdir(TEST_TIFF)):
    print "Read Image: ", filename
    img = cv2.imread(TEST_TIFF + '/'+ filename)
    X.append(img)

X = np.array(X)

np.save('../data/x_test.npy', X)

print X.shape
