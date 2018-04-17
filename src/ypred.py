from __future__ import division

import numpy as np
from sklearn.metrics import accuracy_score, fbeta_score, confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd

from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt

import process_y as py
from constants import *
from utils import *

def avg_pred(arrs1,arrs2,arrs3):
    m1 = np.mean(arrs1, axis = 0)
    m2 = np.mean(arrs2, axis = 0)
    m3 = np.mean(arrs3, axis = 0)

    YTEST1 = np.argmax(m1, axis = 1)

    thresholds = [0.90,0.26,0.40,0.1,0.3,0.20,0.12]
    YTEST2 = label_predictions(m2, thresholds)

    thresholds = [0.86, 0.94, 0.60, 0.64, 0.60, 0.66] 
    YTEST3 = label_predictions(m3, thresholds)

    filenames = np.load(XTEST_FILES) 
   
    write_csv(filenames, pred_cl = YTEST2, pred_wr = YTEST1, pred_rl = YTEST3, pred_rare=True)  

s = py.Y[weather_labels]
s1 = np.load('../predictions/y1_jpg_inception.npy')
st1 = np.load('../predictions/y1test_jpg_inception.npy')
s2 = np.load('../predictions/y1_jpg_resnet.npy')
st2 = np.load('../predictions/y1test_jpg_resnet.npy')
s3 = np.load('../predictions/y1_jpg_own.npy')
st3 = np.load('../predictions/y1test_jpg_own.npy')
s4 = np.load('../predictions/y1_tif_own.npy')
st4 = np.load('../predictions/y1test_tif_own.npy')
t = py.Y[common_labels]
t1 = np.load('../predictions/y2_jpg_inception.npy')
tt1 = np.load('../predictions/y2test_jpg_inception.npy')
t2 = np.load('../predictions/y2_tif_own.npy')
tt2 = np.load('../predictions/y2test_tif_own.npy')
t3 = np.load('../predictions/y2_jpg_own.npy')
tt3 = np.load('../predictions/y2test_jpg_own.npy')
r = py.Y[rare_labels]
r1 = np.load('../predictions/y3_jpg_own.npy')
rt1 = np.load('../predictions/y3test_jpg_own.npy')
r2 = np.load('../predictions/y3_tif_own.npy')
rt2 = np.load('../predictions/y3test_tif_own.npy')

#avg_pred(arrs1,arrs2,arrs3)

def box_plot():
    label = 'habitation'
    index0 = t[t[label] == 0].index
    index1 = t[t[label] == 1].index

    data = [t1[index0], t1[index1]]
    plt.figure()
    plt.boxplot(data, meanline = True)
    plt.show()
    
#box_plot()

def ann_pred(X_train, X_test, y_train, y_test):
    model = MLPClassifier(hidden_layer_sizes=(100,50,20,10), alpha = 1, early_stopping = True)
    model.fit(X_train, y_train)
    ypred = model.predict(X_test)
    print "ann:", fbeta_score(y_test, ypred,2), confusion_matrix(y_test, ypred)
    return ypred

def tree_pred(X_train, X_test, y_train, y_test):
    model = XGBClassifier()
    model.fit(X_train, y_train)
    ypred = model.predict(X_test)
    print "tree:", fbeta_score(y_test, ypred, 2), confusion_matrix(y_test, ypred)
    return ypred

def logistic_pred(X_train, X_test, y_train, y_test,class_weight={0:1,1:2}):
    model = LogisticRegression(class_weight=class_weight)
    model.fit(X_train, y_train)
    ypred = model.predict(X_test)
    print "logistic:", fbeta_score(y_test, ypred, 2), confusion_matrix(y_test, ypred)
    return ypred

def linearsvm_pred(X_train, X_test, y_train, y_test,class_weight={0:1,1:2}):
    model = LinearSVC(class_weight=class_weight)
    model.fit(X_train, y_train)
    ypred = model.predict(X_test)
    print "linearsvm: ", fbeta_score(y_test, ypred, 2), confusion_matrix(y_test, ypred)
    return ypred

def svm_pred(X_train, X_test, y_train, class_weight=None,y_test=[]):
    if class_weight:
        model = SVC(kernel = 'rbf',class_weight=class_weight,tol=1e-4)
    else:
        model = SVC(kernel = 'rbf')
    model.fit(X_train, y_train)
    ypred = model.predict(X_test)
    if len(y_test) != 0:
        print "svm: ", fbeta_score(y_test, ypred, 2), confusion_matrix(y_test, ypred)
    return ypred

def threshold_pred(X_train, X_test, y_train, y_test):
    thresholds = [0.90,0.26,0.40,0.1,0.3,0.20,0.12]
    ypred = hard_predictions(X_test, thresholds)
    print 'threshold: ',
    print fbeta_score(y_test[:,0], ypred[:,0], 2)
    print fbeta_score(y_test[:,1], ypred[:,1], 2)
    print fbeta_score(y_test[:,2], ypred[:,2], 2)
    print fbeta_score(y_test[:,3], ypred[:,3], 2)
    print fbeta_score(y_test[:,4], ypred[:,4], 2)
    print fbeta_score(y_test[:,5], ypred[:,5], 2)
    print fbeta_score(y_test[:,6], ypred[:,6], 2)
    print fbeta_score(y_test, ypred, 2, average='samples')

def m1_predict(X_test):
    X_train = np.mean((s1,s2,s3,s4), axis = 0)
    y_train = s.values
    return svm_pred(X_train, X_test, y_train)

def m2_predict(X_test):
    X_train = np.mean((t1,t2,t3), axis = 0)
    y_train = t.values

    class_weight={0:2,1:1}
    ypred1 = svm_pred(X_train, X_test, y_train[:,0], class_weight=class_weight)

    class_weight={0:1,1:10}
    ypred1 = np.c_[ypred1, svm_pred(X_train, X_test, y_train[:,1],class_weight=class_weight)]

    class_weight={0:1,1:10}
    ypred1 = np.c_[ypred1, svm_pred(X_train, X_test, y_train[:,2],class_weight=class_weight)]

    gen_size = 7000
    G = np.c_[np.random.uniform(size=gen_size), \
          np.random.uniform(size=gen_size),
          np.random.uniform(size=gen_size),
          np.random.uniform(low=0.1, size=gen_size),
          np.random.uniform(size=gen_size),
          np.random.uniform(size=gen_size),
          np.random.uniform(size=gen_size)]
    class_weight={0:1,1:10}
    x = np.r_[X_train, G]
    y = np.r_[y_train[:,3], np.ones(gen_size)]
    ypred1 = np.c_[ypred1, svm_pred(x, X_test, y,class_weight=class_weight)]

    gen_size = 1000
    G = np.c_[np.random.uniform(size=gen_size), \
          np.random.uniform(size=gen_size),
          np.random.uniform(size=gen_size),
          np.random.uniform(size=gen_size),
          np.random.uniform(low = 0.3,size=gen_size),
          np.random.uniform(size=gen_size),
          np.random.uniform(size=gen_size)]
    class_weight={0:1,1:10}
    x = np.r_[X_train, G]
    y = np.r_[y_train[:,4], np.ones(gen_size)]
    ypred1 = np.c_[ypred1, svm_pred(x, X_test, y,class_weight=class_weight)]

    class_weight={0:1,1:10}
    ypred1 = np.c_[ypred1, svm_pred(X_train, X_test, y_train[:,5],class_weight=class_weight)]

    gen_size = 1000
    G = np.c_[np.random.uniform(size=gen_size), \
          np.random.uniform(size=gen_size),
          np.random.uniform(size=gen_size),
          np.random.uniform(size=gen_size),
          np.random.uniform(size=gen_size),
          np.random.uniform(size=gen_size),
          np.random.uniform(low=0.14,size=gen_size)]
    class_weight={0:1,1:10}
    x = np.r_[X_train, G]
    y = np.r_[y_train[:,6], np.ones(gen_size)]
    ypred1 = np.c_[ypred1, svm_pred(x, X_test, y,class_weight=class_weight)]

    print ypred1[2], ypred1[100], ypred1[1000]
    preds = []
    for y in ypred1:
        temp = [i for i,v in enumerate(y) if v == 1]
        preds.append(temp)
    print preds[2],preds[100], preds[1000]
    return preds

def m3_predict(X_test):
    X_train = np.mean((r1,r2), axis = 0)
    y_train = r.values

    thresholds = [0.86, 0.94, 0.60, 0.64, 0.60, 0.66] 
    class_weight={0:1,1:50}
    gen_size = 30000
    G = np.c_[np.random.uniform(low=0.86,size=gen_size), \
          np.random.uniform(size=gen_size),
          np.random.uniform(size=gen_size),
          np.random.uniform(size=gen_size),
          np.random.uniform(size=gen_size),
          np.random.uniform(size=gen_size)]
    x = np.r_[X_train, G]
    y = np.r_[y_train[:,0], np.ones(gen_size)]
    ypred1 = svm_pred(x, X_test, y,class_weight=class_weight)

    class_weight={0:1,1:50}
    gen_size = 30000
    G = np.c_[np.random.uniform(size=gen_size), \
          np.random.uniform(low=0.94,size=gen_size),
          np.random.uniform(size=gen_size),
          np.random.uniform(size=gen_size),
          np.random.uniform(size=gen_size),
          np.random.uniform(size=gen_size)]
    x = np.r_[X_train, G]
    y = np.r_[y_train[:,1], np.ones(gen_size)]
    ypred1 = np.c_[ypred1, svm_pred(x, X_test, y,class_weight=class_weight)]
    class_weight={0:1,1:50}
    gen_size = 30000
    G = np.c_[np.random.uniform(size=gen_size), \
          np.random.uniform(size=gen_size),
          np.random.uniform(low=0.60,size=gen_size),
          np.random.uniform(size=gen_size),
          np.random.uniform(size=gen_size),
          np.random.uniform(size=gen_size)]
    x = np.r_[X_train, G]
    y = np.r_[y_train[:,2], np.ones(gen_size)]
    ypred1 = np.c_[ypred1, svm_pred(x, X_test, y,class_weight=class_weight)]
    class_weight={0:1,1:50}
    gen_size = 30000
    G = np.c_[np.random.uniform(size=gen_size), \
          np.random.uniform(size=gen_size),
          np.random.uniform(size=gen_size),
          np.random.uniform(low=0.64, size=gen_size),
          np.random.uniform(size=gen_size),
          np.random.uniform(size=gen_size)]
    x = np.r_[X_train, G]
    y = np.r_[y_train[:,3], np.ones(gen_size)]
    ypred1 = np.c_[ypred1, svm_pred(x, X_test, y,class_weight=class_weight)]
    class_weight={0:1,1:50}
    gen_size = 30000
    G = np.c_[np.random.uniform(size=gen_size), \
          np.random.uniform(size=gen_size),
          np.random.uniform(size=gen_size),
          np.random.uniform(size=gen_size),
          np.random.uniform(low=0.60,size=gen_size),
          np.random.uniform(size=gen_size)]
    x = np.r_[X_train, G]
    y = np.r_[y_train[:,4], np.ones(gen_size)]
    ypred1 = np.c_[ypred1, svm_pred(x, X_test, y,class_weight=class_weight)]
    class_weight={0:1,1:50}
    gen_size = 30000
    G = np.c_[np.random.uniform(size=gen_size), \
          np.random.uniform(size=gen_size),
          np.random.uniform(size=gen_size),
          np.random.uniform(size=gen_size),
          np.random.uniform(size=gen_size),
          np.random.uniform(low=0.66,size=gen_size)]
    x = np.r_[X_train, G]
    y = np.r_[y_train[:,5], np.ones(gen_size)]
    ypred1 = np.c_[ypred1, svm_pred(x, X_test, y,class_weight=class_weight)]
    preds = []
    for y in ypred1:
        temp = [i for i,v in enumerate(y) if v == 1]
        preds.append(temp)
    return preds

def pred1():
    X1 = np.mean((st1,st2,st3,st4), axis = 0)
    X2 = np.mean((tt1,tt2,tt3), axis = 0)
    X3 = np.mean((rt1, rt2), axis = 0)

    YTEST1 = m1_predict(X1)

    YTEST2 = m2_predict(X2)

    YTEST3 = m3_predict(X3)

    filenames = np.load(XTEST_FILES) 
   
    write_csv(filenames, pred_cl = YTEST2, pred_wr = YTEST1, pred_rl = YTEST3, pred_rare=True)  

pred1()


#### Add some synthetic data to make primary 0 and cover all combinations of labels
#### Try 7 individual MLP classifiers ?
#### Try 7 individual Tree models, SVM, Logistic Regression ?

#X = np.c_[t1,t2,t3]
#Y = t.values
#X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
#print "ann1:",
#nn_pred(X_train, X_test, y_train, y_test)

#X = np.mean((t1,t2,t3), axis = 0)
X = np.mean((r1,r2), axis = 0)
Y = r.values


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

#threshold_pred(X_train, X_test, y_train, y_test)
print

def acc_pred(i):
    gen_size = 30000
    G = np.c_[np.random.uniform(low=0.9,size=gen_size), \
          np.random.uniform(size=gen_size),
          np.random.uniform(size=gen_size),
          np.random.uniform(size=gen_size),
          np.random.uniform(size=gen_size),
          np.random.uniform(size=gen_size)]
    x = np.r_[X_train, G]
    y = np.r_[y_train[:,i], np.ones(gen_size)]
    print rare_labels[i],
    class_weight={0:1,1:100}
    ypred1 = svm_pred(x, X_test, y, y_test[:,i], class_weight=class_weight)

acc_pred(1)
print
'''
print rare_labels[1],
class_weight={0:1,1:100}
ypred1 = np.c_[ypred1, svm_pred(X_train, X_test, y_train[:,1], y_test[:,1],class_weight=class_weight)]
print

print rare_labels[2],
class_weight={0:1,1:10}
ypred1 = np.c_[ypred1, svm_pred(X_train, X_test, y_train[:,2], y_test[:,2],class_weight=class_weight)]
print 

print rare_labels[3],
class_weight={0:1,1:10}
ypred1 = np.c_[ypred1, svm_pred(X_train, X_test, y_train[:,3], y_test[:,3], class_weight=class_weight)]
print

print rare_labels[4],
class_weight={0:1,1:10}
ypred1 = np.c_[ypred1, svm_pred(X_train, X_test, y_train[:,4], y_test[:,4],class_weight=class_weight)]
print

print rare_labels[5],
class_weight={0:1,1:10}
ypred1 = np.c_[ypred1, svm_pred(X_train, X_test, y_train[:,5], y_test[:,5],class_weight=class_weight)]
print

print common_labels[6],
class_weight={0:1,1:9}

class_weight={0:1,1:9}

#index = np.arange(len(y_train))
#index6a = np.random.choice(index[np.where(y_train[:,6] == 0)[0]], 7000, replace=False)
#index6b = np.random.choice(index[np.where(y_train[:,6] == 1)[0]], 3000, replace=True)

#x = np.r_[X_train[index6a], X_train[index6b]]
#y = np.r_[y_train[index6a], y_train[index6b]][:,6]

gen_size = 3000
G = np.c_[np.random.uniform(size=gen_size), \
          np.random.uniform(low=0.26,size=gen_size),
          np.random.uniform(size=gen_size),
          np.random.uniform(size=gen_size),
          np.random.uniform(size=gen_size),
          np.random.uniform(size=gen_size),
          np.random.uniform(size=gen_size)]

x = np.r_[X_train, G]
#x = X_train
    #thresholds = [0.90,0.26,0.40,0.1,0.3,0.20,0.12]
print y_train.shape
y = np.r_[y_train[:,1], np.ones(gen_size)]
#y = y_train[:,4]
svm_pred(x, X_test, y, y_test[:,1])


#ypred1 = np.c_[ypred1, svm_pred(x, X_test, y, y_test[:,6],class_weight=class_weight)]
print "cumulative: "

#print "svm: ", fbeta_score(y_test, ypred1, 2, average='samples')
'''

def kernel1(predict=False):
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
        Ypred = np.mean(arrs2, axis = 0)

    thresholds = [0.0] * len(common_labels)
    delta = 0.02
    tol = 0.002
    print distY

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
    
def kernel2(predict=False):
    Y = py.Y[py.Y['cloudy'] == 0]

    Y = Y[rare_labels] 
    #Y = Y[(Y.T != 0).any()]
    
    dist = lambda x: [len(x[x[i] == 1]) / len(x) for i in rare_labels]
    distY = dist(Y)

    if predict:
        X = normalize(np.load(XTRAIN_JPG))
        model = load_model('../models/model3_jpg_own.h5')
        Ypred = model.predict(X[Y.index])
        np.save('ypred2.npy', Ypred)
    else:
        Ypred = np.mean(arrs3, axis = 0)

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

#kernel1()
#kernel2()
