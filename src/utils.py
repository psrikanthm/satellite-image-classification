import numpy as np
import csv

from constants import *

def normalize(X):
    Temp = np.mean(X, axis = 0)
    X = X - Temp
    Temp = np.std(X, axis = 0)
    return X/Temp

def label_predictions(ypred, thresholds):
    predictions = []
    for prediction in ypred:
        labels = [i for i, value in enumerate(prediction) if value > thresholds[i]]
        predictions.append(labels)
    return np.array(predictions)

def write_csv(filenames, pred_cl = [], pred_wr = [], pred_rl = [], pred_rare = False):
    ofile  = open(PRED_FILE, "wb")
    writer = csv.writer(ofile, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
    
    row = []
    row.append('image_name')
    row.append('tags')
    writer.writerow(row)
 
    for i in range(len(pred_wr)):
        row = []
        row.append(filenames[i])

        tags =  weather_labels[pred_wr[i]]

        for label in pred_cl[i]:
            tags += " " + common_labels[label]

        if pred_rare:
            for label in pred_rl[i]:
                tags += " " + rare_labels[label]

        tags = tags.strip()
        row.append(tags)
        writer.writerow(row)
 
    ofile.close()

def hard_predictions(ypred, thresholds):
    predictions = []
    for prediction in ypred:
        labels = [1 if value > thresholds[i] else 0 for i, value in enumerate(prediction)]
        predictions.append(labels)
    return np.array(predictions)

def multilabel_to_onehot(array_x):
    a = []
    for x in array_x:
        a.append(sum(1<<i for i, b in enumerate(x[::-1]) if b))
   
    a = np.array(a)
    import sklearn.preprocessing
    label_binarizer = sklearn.preprocessing.LabelBinarizer()
    label_binarizer.fit(range(max(a)+1))
    return label_binarizer.transform(a)

def encode2binary(array_x):
    a = []
    for x in array_x:
        a.append([int(i) for i in bin(x)[2:]])
    return np.array(a)

def softmax(x):
    x = np.transpose(x)
    scoreMatExp = np.exp(np.asarray(x))
    s = scoreMatExp / scoreMatExp.sum(0)
    return np.transpose(s)
