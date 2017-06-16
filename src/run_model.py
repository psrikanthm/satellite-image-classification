import numpy as np
import csv

import simple
import vgg
import inception
import resnet

from keras.optimizers import SGD

from keras import metrics
import keras.backend as K

from constants import *
from utils import *
import process_y as py

from sklearn.metrics import f1_score

def f1_score(y_true, y_pred):

    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

    # If there are no true samples, fix the F1 score at 0.
    if c3 == 0 || c2 == 0:
        return 0

    # How many selected items are relevant?
    precision = c1 / c2

    # How many relevant items are selected?
    recall = c1 / c3

    # Calculate f1_score
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score

def FScore2(y_true, y_pred):
    '''
    The F score, beta=2
    '''
    B2 = K.variable(4)
    OnePlusB2 = K.variable(5)
    pred = K.round(y_pred)
    tp = K.sum(K.cast(K.less(K.abs(pred - K.clip(y_true, .5, 1.)), 0.01), 'float32'), -1)
    fp = K.sum(K.cast(K.greater(pred - y_true, 0.1), 'float32'), -1)
    fn = K.sum(K.cast(K.less(pred - y_true, -0.1), 'float32'), -1)

    f2 = OnePlusB2 * tp / (OnePlusB2 * tp + B2 * fn + fp)

    return K.mean(f2)

def train(XTRAIN, YTRAIN, model_name = 'simple', epochs = 5, trainable_from = 'mixed9'):
    _, img_width, img_height, channels = XTRAIN.shape
    if YTRAIN.ndim == 1:
        classes = 1
        loss_fn = 'binary_crossentropy'
    else:
        _, classes = YTRAIN.shape
        loss_fn = 'categorical_crossentropy'

    if model_name == 'vgg':
        model = vgg.get_model(img_width, img_height, channels, classes)
    elif model_name == 'alexnet':
        pass
    elif model_name == 'inception':
        model = inception.get_model(img_width, img_height, channels, classes, trainable_from)
    elif model_name == 'resnet':
        model = resnet.get_model(img_width, img_height, channels, classes)
    else:
        model = simple.get_model(img_width, img_height, channels, classes)

    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss=loss_fn, metrics = [metrics.binary_accuracy, 'accuracy', FScore2, f1_score])
    model.fit(XTRAIN, YTRAIN, epochs=epochs, verbose=1, validation_split=0.2)
    return model

def predict(model, XTEST, PRED_FILE = 'tmp'):
    YTEST = model.predict(XTEST)
    if PRED_FILE != 'tmp':
        np.save(PRED_FILE, YTEST)
    return YTEST

def write_csv(filenames, pred_cl, pred_wr):
    y1 = softmax(pred_cl)
    y2 = softmax(pred_wr)

    w1 = encode2binary(np.argmax(y1, axis = 1))
    w2 = np.argmax(y2, axis = 1)

    ofile  = open(PRED_FILE, "wb")
    writer = csv.writer(ofile, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
    
    row = []
    row.append('image_name')
    row.append('tags')
    writer.writerow(row)
 
    for i in range(len(w2)):
        row = []
        row.append(filenames[i])

        tags =  weather_labels[w2[i]]
        if w2[i] != 0: 
            clabels = np.where(w1[i] == 1)[0]
            for label in clabels:
                tags += " " + common_labels[label]
        tags = tags.strip()
        row.append(tags)
        writer.writerow(row)
 
    ofile.close()

def run_combine():
    xtrain = np.load(XTRAIN_JPG)

    print "Training Weather Labels model"
    Y = py.Y[weather_labels]
    Y = Y[(Y.T != 0).any()]
    #Yt = encode2label(Y.values)
    model1 = train(xtrain[Y.index], Y.values, 'inception', 1, 'mixed9')

    xtest = np.load(XTEST_JPG)
    filenames = np.load(XTEST_FILES)
    Ypred1 = predict(model1, xtest, YTEST_CL)

    print "Training Common Labels model"
    Y = py.Y[common_labels]
    Y = Y[(Y.T != 0).any()]
    Yt = encode2label(Y.values)
    model2 = train(xtrain[Y.index], Yt, 'inception', 1, 'mixed8')
    Ypred2 = predict(model2, xtest, YTEST_WR)

    write_csv(filenames, Ypred2, Ypred1)

def run_ind():
    X = np.load(XTRAIN_JPG)
    xtest = np.load(XTEST_JPG)

    for label in common_labels:
        #a,b = py.data_by_label(label)
        
        #replace = False if len(b) > len(a) else True
      
        #xtrain = np.concatenate((X[a,:], X[np.random.choice(b,len(a),replace = replace),:]), axis = 0)
        #ytrain = np.concatenate((np.ones_like(a), np.zeros_like(a)), axis = 0)
        xtrain = X
        ytrain = py.Y[label].values
        model = train(xtrain, ytrain, 'inception', 5, 'mixed9')
        predict(model, xtest, "../data/ytest_cl_" + label + ".npy")
    
    #Y = Y[(Y.T != 0).any()]
    #model = train(xtrain[Y.index], Yt, 'inception', 1)

if __name__ == '__main__':
    #run_combine()
    run_ind()
