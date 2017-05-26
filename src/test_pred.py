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


def get_common_labels(y1[i]):
    return []

#ofile  = open('prediction.csv', "wb")
#writer = csv.writer(ofile, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
     
for i in range(len(y2)):
    row = []
    filename = os.path.splitext(names[i])[0]
    row.append(filename)
    wr_label = np.argmax(y2)[0]
    c_labels = ''
    c_classes = get_common_labels(y1[i])
    
    row.append(wr_label + " " + c_labels)
    #writer.writerow(row)
 
#ofile.close()
