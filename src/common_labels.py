import numpy as np
import process_y as py
import cv2

###################################
#### Constants ###################

TRAIN_JPG = "../data/train-jpg"
TRAIN_TIFF = "../data/train-tif-v2"

##################################

common_labels = ["primary", "water", "habitation", "agriculture", "road", "cultivation", "bare_ground"]
Y = py.selected_columns(common_labels)
Y = Y[(Y.T != 0).any()]

X = []
for i in Y.index:
    print "Read Image: ", i
    img = cv2.imread(TRAIN_TIFF + '/train_'+ str(i) +'.tif')
    X.append(img)

X = np.array(X)

np.save('x_train.npy', X)
np.save('y_train.npy', Y)

print X.shape
print Y.shape
