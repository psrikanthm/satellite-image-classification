import numpy as np
import csv

import simple
import vgg
import inception
import resnet
import amazon
import feedforward

from keras.optimizers import SGD
from keras.optimizers import Adam

from keras import metrics
import keras.backend as K
from keras.models import load_model
from keras.callbacks import ModelCheckpoint

from constants import *
from utils import *
import process_y as py

from sklearn.metrics import fbeta_score, f1_score, accuracy_score
from sklearn.utils import class_weight
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

def train(XTRAIN, YTRAIN, model_name = 'simple', epochs = 5, trainable_from = 'mixed9', class_weight = None, model_id = 'best'):

    if XTRAIN.ndim == 2:
        img_height, img_width = XTRAIN.shape
    elif XTRAIN.ndim == 3:    
        _, img_width, img_height = XTRAIN.shape
    elif XTRAIN.ndim == 4:
        _, img_width, img_height, channels = XTRAIN.shape
    else:
        return None

    if YTRAIN.ndim == 1:
        classes = 1
        loss_fn = 'binary_crossentropy'
    else:
        _, classes = YTRAIN.shape
        loss_fn = 'categorical_crossentropy'

    filepath="weights_" + model_id + ".hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    if model_name == 'vgg':
        model = vgg.get_model(img_width, img_height, channels, classes)
    elif model_name == 'alexnet':
        pass
    elif model_name == 'inception':
        model = inception.get_model(img_width, img_height, channels, classes, trainable_from)
    elif model_name == 'resnet':
        model = resnet.get_model(img_width, img_height, channels, classes)
    elif model_name == 'amazon':
        model = amazon.get_model(img_width, img_height, channels, classes)
    elif model_name == 'feedforward':
        model = feedforward.get_model(img_height, img_width, classes)
    else:
        model = simple.get_model(img_width, img_height, channels, classes)

    #opt = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    opt = Adam(lr=0.001)
    
    model.load_weights("weights_best.hdf5")
    model.compile(optimizer=opt, loss=loss_fn, metrics = ['accuracy'])
    model.fit(XTRAIN, YTRAIN, epochs=epochs, batch_size = 64, verbose=1, validation_split=0.2, callbacks=callbacks_list, class_weight = class_weight)
    return model

def predict(XTEST = [], PRED_FILE = 'tmp'):
    XTEST = np.load(XTEST_TIF)
    print "Loaded Test Files"

    model = load_model('model1.h5')
    YTEST1 = model.predict(XTEST)
    np.save(YTEST_WR, YTEST1)
    print "Predicted Weather Labels"

    model = load_model('model2.h5')
    thresholds = [0.2] * len(common_labels)
    YTEST2 = label_predictions(model.predict(XTEST), thresholds)
    np.save(YTEST_CL, YTEST2)
    print "Predicted Common Labels"

    model = load_model('model3.h5')
    thresholds = [0.2] * len(rare_labels)
    YTEST3 = label_predictions(model.predict(XTEST), thresholds)
    np.save(YTEST_RL, YTEST3)
    print "Predicted Rare Labels"

    filenames = np.load(XTEST_FILES) 
   
    write_csv(filenames, pred_cl = YTEST2, pred_wr = YTEST1, pred_rl = YTEST3)  

    if PRED_FILE != 'tmp':
        np.save(PRED_FILE, YTEST)
    return None

def predict2():
    #thresholds = [0.0432515 ,  0.9994697 ,  0.11011287,  0.01312997,  0.09748124, 0.04374343, 0.00665135]
    thresholds = np.mean(np.load('tmp.npy'), axis = 0)
    YTEST1 = np.load(YTEST_WR)
    YTEST2 = label_predictions(np.load(YTEST_CL), thresholds)
    YTEST3 = np.load(YTEST_RL)

    filenames = np.load(XTEST_FILES) 
    write_csv(filenames, pred_cl = YTEST2, pred_wr = YTEST1, pred_rl = YTEST3)  

def write_csv(filenames, pred_cl = [], pred_wr = [], pred_rl = []):
    #y1 = softmax(pred_cl)
    w1 = pred_cl
    y2 = softmax(pred_wr)
    w3 = pred_rl

    #w1 = encode2binary(np.argmax(y1, axis = 1))
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

        for label in w1[i]:
            tags += " " + common_labels[label]
        #for label in w3[i]:
        #    tags += " " + rare_labels[label]

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

def run_ind():
    X = np.load(XTRAIN_TIF)
    
    Y = py.Y[(py.Y['primary'] == 1) & (py.Y['clear'] == 1)]
    S = Y[Y[Y.columns.difference(['primary', 'clear'])] == 0]
    b = S.dropna(thresh= len(S.columns) - 2).index
    cl_dup = list(set(common_labels) - set(['primary']))

    for label in cl_dup:
        a,c = py.data_by_label(label)
        
        replace = False if len(b) > len(a) else True
      
        xtrain = np.concatenate((X[a,:], X[np.random.choice(b,len(a),replace = replace),:]), axis = 0)
        ytrain = np.concatenate((np.ones_like(a), np.zeros(len(a))), axis = 0)

        model = train(xtrain, ytrain, 'amazon', 10, model_id = label)
        #model.save('model_' + label + '.h5')
        ypred = model.predict(xtrain)
        np.save('pred1_' + label + '.npy', ypred)

        xval = np.concatenate((X[a,:], X[np.random.choice(c,len(a),replace = replace),:]), axis = 0)
        ypred = model.predict(xval)
        np.save('pred2_' + label + '.npy', ypred)
        print fbeta_score(ytrain, np.argmax(softmax(ypred), axis = 1), 2)

    a,b = py.data_by_label('primary')
    xtrain = X
    ytrain = py.Y['primary'].values
    model = train(xtrain, ytrain, 'amazon', 10, model_id = 'primary')
    #model.save('model_primary.h5')

def ind_combine():
    X = np.load(XTRAIN_TIF)
    #cl_dup = list(set(common_labels) - set(['primary']))
    cl_dup = common_labels
    Y = py.Y[cl_dup]
    Y = Y[(Y.T != 0).any()]
    
    ypred = []
    for label in common_labels:
        #model = load_model('model_' + label + '.h5')
        ypred.append(model.predict(X))
    ypred = np.array(ypred)
    
    #####
    #print ypred.shape
    np.save('ypred_final.npy', ypred)
    ####
       
    #ypred = np.load('ypred_final.npy') 
    ypred = np.transpose(ypred)[0]
    #np.delete(ypred, common_labels.index('primary'), axis = 1)
    ypred = ypred[Y.index]
    
    X_train, X_test, y_train, y_test = train_test_split(ypred, Y.values, test_size=0.2, random_state=42)
    model = MLPClassifier(hidden_layer_sizes=(6,6), verbose = True)
    model.fit(X_train, y_train)
    ypred = model.predict(X_test)

    #model = train(ypred, Y.values, 'feedforward', 5)
    #model.save('model_ff.h5')
    #thresholds = [0.2] * len(common_labels)
    #ypred = hard_predictions(model.predict(ypred), thresholds)
    np.save('tmp.npy', ypred)
    print accuracy_score(y_test, ypred)
    print fbeta_score(y_test, ypred, 2, average = 'samples')


def run_all():
    xtrain = np.load(XTRAIN_TIF)
    
    print "Training Weather Labels model"
    Y = py.Y[weather_labels]
    Y = Y[(Y.T != 0).any()]
    
    s = np.sum(Y.values, axis = 0)
    s = np.sum(s)/s
    s = s/np.sum(s)
    class_weight = {}
    for i,l in enumerate(weather_labels):
        class_weight[i] = s[i]

    model1 = train(xtrain[Y.index], Y.values, 'amazon', 5)
    model1.save('model1.h5')

    print "Training Common Labels model"
    Y = py.Y[py.Y['clear'] == 1]
    Y = Y[common_labels]
    Y = Y[(Y.T != 0).any()]
    s = np.sum(Y.values, axis = 0)
    s = np.sum(s)/s
    s = s/np.sum(s)

    class_weight = {}
    for i,l in enumerate(common_labels):
        class_weight[i] = s[i]

    model2 = train(xtrain[Y.index], Y.values, 'amazon', 15)
    model2.save('model2.h5')
    ytrue = Y.values
    ypred = model2.predict(xtrain[Y.index])   
    np.save('tmp.npy', ypred)

    print "Training Rare Labels model"
    Y = py.Y[rare_labels]
    Y = Y[(Y.T != 0).any()]
    s = np.sum(Y.values, axis = 0)
    s = np.sum(s)/s
    s = s/np.sum(s)
    class_weight = {}
    for i,l in enumerate(rare_labels):
        class_weight[i] = s[i]

    #model3 = train(xtrain[Y.index], Y.values, 'amazon', 5, 'mixed8', class_weight)    
    #ytrue = Y.values
    #ypred = model3.predict()
    #model3.save('model3.h5')
 

def run_all_part():
    Y = py.Y[common_labels]
    Y = Y[(Y.T != 0).any()]
    ytrue = Y.values

    ypred = np.load('tmp.npy')
    print ytrue.shape, ytrue[0,:]
    yp = hard_predictions(ypred, [0.2] * len(common_labels))
    print yp.shape, yp[0,:]
    print fbeta_score(ytrue, yp, 2, average='samples')
 
    for index, label in enumerate(common_labels):
        threshold = 0.05
        ytrue = Y[label].values
        predictions = ypred[:,index]

        while threshold < 0.7:
            ypred2 = []
            for pred in predictions:
                tmp = 1 if pred > threshold else 0
                ypred2.append(tmp)
            print "label: ", label, ", threshold: ", threshold, ", score: ", fbeta_score(ytrue, ypred2, 2)        
            threshold += 0.05

    #print "Training Rare Labels model"
    #Y = py.Y[rare_labels]
    #Y = Y[(Y.T != 0).any()]
    #model3 = train(xtrain[Y.index], Y.values, 'amazon', 1, 'mixed8')    
    #ytrue = Y.values
    #ypred = model3.predict()

if __name__ == '__main__':
    pass
    #run_combine()
    #run_ind()
    #ind_combine()
    #run_all()
    #run_all_part()
    #predict()
    #predict2()
