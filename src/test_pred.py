import numpy as np
import csv
import os

TEST_TIFF = "../data/test-tif-v2"
TEST_FILES = "../data/test_filenames.npy"

common_labels = ["primary", "water", "habitation", "agriculture", "road", "cultivation", "bare_ground"]
weather_labels = ["cloudy", "partly_cloudy", "haze", "clear"]

y1 = np.load('../data/y_test_cl.npy')
y2 = np.load('../data/y_test_weather.npy')

names = np.load('../data/test_filenames.npy')

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

y1 = softmax(y1)
y2 = softmax(y2)

w1 = encode2binary(np.argmax(y1, axis = 1))
w2 = np.argmax(y2, axis = 1)

ofile  = open('../data/prediction.csv', "wb")
writer = csv.writer(ofile, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
    
row = []
row.append('image_name')
row.append('tags')
writer.writerow(row)
 
for i in range(len(w2)):
    row = []
    filename = os.path.splitext(names[i])[0]

    row.append(filename)

    tags =  weather_labels[w2[i]]
    
    if w2[i] != 0: 
        clabels = np.where(w1[i,:] == 1)[0]
        for label in clabels:
            if label == 1:
                print common_labels[label]
            tags += " " + common_labels[label]
    tags = tags.strip()
    row.append(tags)
    writer.writerow(row)
 
ofile.close()
