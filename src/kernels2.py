from __future__ import division

import numpy as np
import csv
import sys
from itertools import izip

import simple
import vgg
import inception
import resnet
import amazon
import feedforward
#import vgg16 as resnet

from keras.optimizers import SGD
from keras.optimizers import Adam

from keras import metrics
import keras.backend as K
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

from constants import *
from utils import *
import process_y as py

import pandas as pd

from sklearn.metrics import fbeta_score, f1_score, accuracy_score
#from sklearn.metrics import confusion_matrix
#from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split


def kernel1():
    
    X = normalize(np.load(XTRAIN_JPG))
    model_name = 'vgg16'
    
    print "Augmenting more weather labels data"
    Y = py.Y[weather_labels]
    Y = Y[(Y.T != 0).any()]

    Y1 = py.Y[["cloudy"]]
    Y1 = Y1[(Y1.T != 0).any()]

    Y2 = py.Y[["partly_cloudy"]]
    Y2 = Y2[(Y2.T != 0).any()]

    Y3 = py.Y[["haze"]]
    Y3 = Y3[(Y3.T != 0).any()]
    
    Y4 = py.Y[["clear"]]
    Y4 = Y4[(Y4.T != 0).any()]

    print "Loaded data"

    # delete big variables to clear memory
    datagen = ImageDataGenerator(
        #featurewise_center=True,
        #featurewise_std_normalization=True,
        #zca_whitening = True,
        rotation_range=180,
        width_shift_range=0.4,
        height_shift_range=0.4,
        shear_range = 1.5708,
        zoom_range = [0.5,2],
        channel_shift_range = 0.3,
        vertical_flip = True,
        horizontal_flip=True)

    #datagen.fit(X[Y1.index])

    no_batches = 500 
    epochs = 5
    batch_size = 128
    batches = 0

    loss_fn = 'categorical_crossentropy'
    _, img_width, img_height, channels = X[Y.index].shape 
    _, classes = Y.values.shape
    model = resnet.get_model(img_width, img_height, channels, classes, \
                            softmax = True, trainable_from = 'res4a')
    opt = Adam(lr=0.001)
    model.compile(optimizer=opt, loss=loss_fn, metrics = ['accuracy'])

    for (xbatch1, ybatch1), (xbatch2, ybatch2), (xbatch3, ybatch3), (xbatch4, ybatch4) \
        in izip(datagen.flow(X[Y1.index], Y.ix[Y1.index].values, batch_size=batch_size), \
            datagen.flow(X[Y2.index], Y.ix[Y2.index].values, batch_size=batch_size), \
            datagen.flow(X[Y3.index], Y.ix[Y3.index].values, batch_size=batch_size), \
            datagen.flow(X[Y4.index], Y.ix[Y4.index].values, batch_size=batch_size)):

        model.fit(np.concatenate((xbatch1, xbatch2, xbatch3, xbatch4), axis = 0), \
                np.concatenate((ybatch1, ybatch2, ybatch3, ybatch4), axis = 0), \
                verbose = 1, epochs = epochs)

        print "################ batch: ", batches
        batches += 1
        if batches >= no_batches:
        # we need to break the loop by hand because
        # the generator loops indefinitely
            break
  

    filepath="../weights/model1_jpg_"+ model + ".hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    epochs = 3
    model.fit(X,Y.values,verbose = 1,validation_split=0.1,epochs = epochs, callbacks = callbacks_list)
    model.save('../models/model1_jpg_'+ model +'.h5')
    print "Augmentation Complete"

def kernel2():
    
    X = normalize(np.load(XTRAIN_JPG))
    model_name = 'resnet'    

    # delete big variables to clear memory
    datagen = ImageDataGenerator(
        #featurewise_center=True,
        #featurewise_std_normalization=True,
        #zca_whitening = True,
        rotation_range=180,
        width_shift_range=0.4,
        height_shift_range=0.4,
        shear_range = 1.5708,
        zoom_range = [0.5,2],
        channel_shift_range = 0.3,
        vertical_flip = True,
        horizontal_flip=True)

    #datagen.fit(X[Y1.index])

    no_batches = 300 
    epochs = 5
    batch_size = 256 
    batches = 0

    print "Training Common Labels model"
    Y = py.Y[py.Y['cloudy'] == 0]

    Y = Y[common_labels] 
    Y = Y[(Y.T != 0).any()]

    Yx = {}

    for index in range(len(common_labels)):
        a = common_labels[1:index] + common_labels[index+1 :]
        Temp = Y[Y[common_labels[index]] == 1]
        Temp = Temp[a]
        Yx[index] = Temp[(Temp.T == 0).all()].index
        
    loss_fn = 'categorical_crossentropy'
    _, img_width, img_height, channels = X[Y.index].shape 
    _, classes = Y.values.shape
    model = resnet.get_model(img_width, img_height, channels, classes, \
                            softmax = False, trainable_from = 'add_9')
    opt = Adam(lr=0.001)
    model.compile(optimizer=opt, loss=loss_fn, metrics = ['accuracy'])

    for (xbatch1, ybatch1), (xbatch2, ybatch2), (xbatch3, ybatch3), (xbatch4, ybatch4), \
        (xbatch5, ybatch5), (xbatch6, ybatch6), (xbatch7, ybatch7) \
        in izip(datagen.flow(X[Yx[0]], Y.ix[Yx[0]].values, batch_size=batch_size), \
            datagen.flow(X[Yx[1]], Y.ix[Yx[1]].values, batch_size=batch_size), \
            datagen.flow(X[Yx[2]], Y.ix[Yx[2]].values, batch_size=batch_size), \
            datagen.flow(X[Yx[3]], Y.ix[Yx[3]].values, batch_size=batch_size), \
            datagen.flow(X[Yx[4]], Y.ix[Yx[4]].values, batch_size=batch_size), \
            datagen.flow(X[Yx[5]], Y.ix[Yx[5]].values, batch_size=batch_size), \
            datagen.flow(X[Yx[6]], Y.ix[Yx[6]].values, batch_size=batch_size)):

        model.fit(np.concatenate((xbatch1, xbatch2, xbatch3, xbatch4, xbatch5, xbatch6, xbatch7), axis = 0), \
                np.concatenate((ybatch1, ybatch2, ybatch3, ybatch4, ybatch5, ybatch6, ybatch7), axis = 0), \
                verbose = 1, epochs = epochs)

        print "################ batch: ", batches
        batches += 1
        if batches >= no_batches:
        # we need to break the loop by hand because
        # the generator loops indefinitely
            break

    filepath="../weights/model2_jpg_"+ model_name +".hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    epochs = 3
    model.fit(X[Y.index],Y.values,verbose = 1,validation_split=0.1,epochs = epochs, callbacks = callbacks_list)
    model.save('../models/model2_jpg_'+ model_name +'.h5')
    print "Augmentation Complete"

def kernel3():
   
    X = normalize(np.load(XTRAIN_JPG))
    model_name = 'vgg16'
    
    # delete big variables to clear memory
    datagen = ImageDataGenerator(
        #featurewise_center=True,
        #featurewise_std_normalization=True,
        #zca_whitening = True,
        rotation_range=180,
        width_shift_range=0.4,
        height_shift_range=0.4,
        shear_range = 1.5708,
        zoom_range = [0.5,2],
        channel_shift_range = 0.3,
        vertical_flip = True,
        horizontal_flip=True)

    #datagen.fit(X[Y1.index])

    no_batches = 550 
    epochs = 5
    batch_size = 32
    batches = 0

    print "Training Rare Labels model"
    Y = py.Y[py.Y['cloudy'] == 0]

    Y1 = Y[rare_labels] 
    Y = Y1[(Y1.T != 0).any()]

    Yx = {}
    _, classes = Y.values.shape

    for index in range(classes):
        a = rare_labels[:index] + rare_labels[index+1 :]
        Temp = Y[Y[rare_labels[index]] == 1]
        Temp = Temp[a]
        Yx[index] = Temp[(Temp.T == 0).all()].index
        
    loss_fn = 'categorical_crossentropy'
    _, img_width, img_height, channels = X[Y.index].shape 
    model = resnet.get_model(img_width, img_height, channels, classes, \
                            softmax = False, trainable_from = 'res4a')
    opt = Adam(lr=0.001)
    model.compile(optimizer=opt, loss=loss_fn, metrics = ['accuracy'])

    for (xbatch1, ybatch1), (xbatch2, ybatch2), (xbatch3, ybatch3), (xbatch4, ybatch4), \
        (xbatch5, ybatch5), (xbatch6, ybatch6)\
        in izip(datagen.flow(X[Yx[0]], Y.ix[Yx[0]].values, batch_size=batch_size), \
            datagen.flow(X[Yx[1]], Y.ix[Yx[1]].values, batch_size=batch_size), \
            datagen.flow(X[Yx[2]], Y.ix[Yx[2]].values, batch_size=batch_size), \
            datagen.flow(X[Yx[3]], Y.ix[Yx[3]].values, batch_size=batch_size), \
            datagen.flow(X[Yx[4]], Y.ix[Yx[4]].values, batch_size=batch_size), \
            datagen.flow(X[Yx[5]], Y.ix[Yx[5]].values, batch_size=batch_size)):

        model.fit(np.concatenate((xbatch1, xbatch2, xbatch3, xbatch4, xbatch5, xbatch6), axis = 0), \
                np.concatenate((ybatch1, ybatch2, ybatch3, ybatch4, ybatch5, ybatch6), axis = 0), \
                verbose = 1, epochs = epochs)

        print "################ batch: ", batches
        batches += 1
        if batches >= no_batches:
        # we need to break the loop by hand because
        # the generator loops indefinitely
            break

    filepath="../weights/model3_jpg_" + model_name + ".hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    epochs = 3
    model.fit(X[Y1.index],Y1.values,verbose = 1,validation_split=0.1,epochs = epochs, callbacks = callbacks_list)
    model.save('../models/model3_jpg_'+ model_name +'.h5')
    print "Augmentation Complete"

def kernel4():
    XTEST = normalize(np.load(XTEST_JPG))
    print "Loaded Test Files"

    model = load_model('../models/model1_jpg_inception.h5')
    YTEST1 = np.argmax(model.predict(XTEST), axis = 1)
    print "Predicted Weather Labels"

    model = load_model('../models/model2_jpg_inception.h5')
    thresholds = [0.90,0.46,0.52,0.26,0.46,0.42,0.14]
    YTEST2 = label_predictions(model.predict(XTEST), thresholds)
    print "Predicted Common Labels"

    model = load_model('../models/model3_jpg_own.h5')
    thresholds = [0.82, 0.94, 0.68, 0.65, 0.99, 0.65] 
    YTEST3 = label_predictions(model.predict(XTEST), thresholds)
    print "Predicted Rare Labels"

    filenames = np.load(XTEST_FILES) 
   
    write_csv(filenames, pred_cl = YTEST2, pred_wr = YTEST1, pred_rl = YTEST3, pred_rare=False)  

def kernel5(predict=True):
    Y = py.Y[py.Y['cloudy'] == 0]

    Y = Y[common_labels] 
    Y = Y[(Y.T != 0).any()]
    
    dist = lambda x: [len(x[x[i] == 1]) / len(x) for i in common_labels]
    distY = dist(Y)

    if predict:
        X = normalize(np.load(XTRAIN_JPG))
        model = load_model('../models/model2_jpg_inception.h5')
        Ypred = model.predict(X[Y.index])
        np.save('ypred.npy', Ypred)
    else:
        Ypred = np.load('ypred.npy')

    thresholds = [0.0] * len(common_labels)
    delta = 0.02
    tol = 0.002

    for i,v in enumerate(common_labels):
        temp = 1
        print
        print v

        while np.abs(distY[i] - temp) > tol and thresholds[i] < 1:
            s = hard_predictions(Ypred, thresholds)
            S = pd.DataFrame(data = s, columns = common_labels)
            temp = dist(S)[i]
            thresholds[i] += delta
            print thresholds[i],
    print
    print thresholds
    print distY
    print dist(S)
    
def kernel6(predict=True):
    Y = py.Y[py.Y['cloudy'] == 0]

    #Y = Y[rare_labels] 
    #Y = Y[(Y.T != 0).any()]
    
    dist = lambda x: [len(x[x[i] == 1]) / len(x) for i in rare_labels]
    distY = dist(Y)

    if predict:
        X = normalize(np.load(XTRAIN_JPG))
        model = load_model('../models/model3_jpg_own.h5')
        Ypred = model.predict(X[Y.index])
        np.save('ypred2.npy', Ypred)
    else:
        Ypred = np.load('ypred2.npy')

    thresholds = [0.0] * len(rare_labels)
    delta = 0.02
    tol = 0.002

    for i,v in enumerate(rare_labels):
        temp = 1
        print
        print v

        while np.abs(distY[i] - temp) > tol and thresholds[i] < 1:
            s = hard_predictions(Ypred, thresholds)
            S = pd.DataFrame(data = s, columns = rare_labels)
            temp = dist(S)[i]
            thresholds[i] += delta
            print thresholds[i],
    print
    print thresholds
    print distY
    print dist(S)

def kernel7():

    #X = normalize(np.load(XTRAIN_JPG))
    X = np.load(XTRAIN_JPG)
    #Xtest = normalize(np.load(XTEST_JPG))
    Xtest = np.load(XTEST_JPG)
    
    Y = py.Y.values

    #Xtrain, Xval, Ytrain, Yval = train_test_split(X[Y.index], Y.values, test_size = 0.2)
    print "Predicting common labels data"
    '''
    model = load_model('../models/model2_jpg_inception.h5')
    Ypred = model.predict(X)
    Ytest = model.predict(Xtest)
    np.save('../predictions/y2_jpg_inception.npy', Ypred)
    np.save('../predictions/y2test_jpg_inception.npy', Ytest)

    model = load_model('../models/model2_jpg_own.h5')
    Ypred = model.predict(X)
    Ytest = model.predict(Xtest)
    np.save('../predictions/y2_jpg_own.npy', Ypred)
    np.save('../predictions/y2test_jpg_own.npy', Ytest)
    
    model = load_model('../models/model1_jpg_inception.h5')
    Ypred = model.predict(X)
    Ytest = model.predict(Xtest)
    np.save('../predictions/y1_jpg_inception.npy', Ypred)
    np.save('../predictions/y1test_jpg_inception.npy', Ytest)
       
    model = load_model('../models/model1_jpg_own.h5')
    Ypred = model.predict(X)
    Ytest = model.predict(Xtest)
    np.save('../predictions/y1_jpg_own.npy', Ypred)
    np.save('../predictions/y1test_jpg_own.npy', Ytest)
    
    model = load_model('../models/model1_jpg_resnet.h5')
    Ypred = model.predict(X)
    Ytest = model.predict(Xtest)
    np.save('../predictions/y1_jpg_resnet.npy', Ypred)
    np.save('../predictions/y1test_jpg_resnet.npy', Ytest)

    model = load_model('../models/model3_jpg_own.h5')
    Ypred = model.predict(X)
    Ytest = model.predict(Xtest)
    np.save('../predictions/y3_jpg_own.npy', Ypred)
    np.save('../predictions/y3test_jpg_own.npy', Ytest)

    X = np.load(XTRAIN_TIF)
    Xtest = np.load(XTEST_TIF)

    model = load_model('../models/model1_tif_own.h5')
    Ypred = model.predict(X)
    Ytest = model.predict(Xtest)
    np.save('../predictions/y1_tif_own', Ypred)
    np.save('../predictions/y1test_tif_own.npy', Ytest)

    model = load_model('../models/model2_tif_own.h5')
    Ypred = model.predict(X)
    Ytest = model.predict(Xtest)
    np.save('../predictions/y2_tif_own.npy', Ypred)
    np.save('../predictions/y2test_tif_own.npy', Ytest)

    model = load_model('../models/model3_tif_own.h5')
    Ypred = model.predict(X)
    Ytest = model.predict(Xtest)
    np.save('../predictions/y3_tif_own.npy', Ypred)
    np.save('../predictions/y3test_tif_own.npy', Ytest)
    #Ypred = hard_predictions(model.predict(Xval), thresholds)
    #print fbeta_score(Yval, Ypred, 2, average = 'samples')
    '''

    print "Complete"

if __name__ == '__main__':
    print "Started"
    np.random.seed(42)
    if sys.argv[1] == "1":
        print "kernel1"
        kernel1()
    elif sys.argv[1] == "2":
        print "kernel2"
        kernel2()
    elif sys.argv[1] == "3":
        print "kernel3"
        kernel3()
    elif sys.argv[1] == "4":
        print "kernel4"
        kernel4()
    elif sys.argv[1] == "5":
        print "kernel5"
        kernel5()
    elif sys.argv[1] == "6":
        print "kernel6"
        kernel6()
    elif sys.argv[1] == "7":
        print "kernel7"
        kernel7()
    else:
        print "No Kernel to run"
