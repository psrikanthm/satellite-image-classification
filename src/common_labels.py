import numpy as np
import process_y as py
import cv2

###################################
#### Constants ###################

TRAIN_JPG = "../data/train-jpg/"
TRAIN_TIFF = "../data/train-tif-v2/"

##################################

def encode2label(array_x):
    a = []
    for x in array_x:
        a.append(sum(1<<i for i, b in enumerate(x[::-1]) if b))
    
    a = np.array(a)
    import sklearn.preprocessing
    label_binarizer = sklearn.preprocessing.LabelBinarizer()
    label_binarizer.fit(range(max(a)+1))
    return label_binarizer.transform(a)

common_labels = ["primary", "water", "habitation", "agriculture", "road", "cultivation", "bare_ground"]
Y = py.selected_columns(common_labels)
Y = Y[(Y.T != 0).any()]

X = []
for i in Y.index:
    print "Read Image: ", i
    img = cv2.imread(TRAIN_TIFF + 'train_'+ str(i) +'.tif', cv2.IMREAD_UNCHANGED)
    X.append(img)

X = np.array(X)
Y = Y.as_matrix()
Y = encode2label(Y)

np.save('../data/x_train_tif_cl.npy', X)
np.save('../data/y_train_cl.npy', Y)


X = []
for i in Y.index:
    print "Read Image: ", i
    img = cv2.imread(TRAIN_JPG + 'train_'+ str(i) +'.jpg', cv2.IMREAD_UNCHANGED)
    X.append(img)

X = np.array(X)
np.save('../data/x_train_jpg_cl.npy', X)
