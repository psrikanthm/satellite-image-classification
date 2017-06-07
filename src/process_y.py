import csv
import numpy as np
import pandas as pd
from constants import *

#import matplotlib.pyplot as plt

#Read the data and parse into a Matrix
ifile = open(YTRAIN_CSV, "rb")
rows = csv.reader(ifile)
next(rows)
#Y = np.zeros((sum(1 for row in rows),N_LABELS))

count = 0

ydata = []
filenames = []

for row in rows:
    count += 1
    ydata.append(row[1].split())
    filenames.append(row[0].strip())

labels = list(set([item for sublist in ydata for item in sublist]))

Y = np.zeros((len(ydata),len(labels)))

iteration = 0
for y in ydata:
    for x in y:
        Y[iteration, labels.index(x)] = 1
    iteration += 1

Y = pd.DataFrame(data = Y, columns = labels)
'''
print "############################# head"
print Y.head()
print "############################# tail"
print Y.tail()
print "############################# index"
print Y.index
print "############################# columns"
print Y.columns
print "############################# summary"
print Y.describe()
print "############################# shape"
print Y.shape
print "############################# correlation plot"
'''
def plot_corr(corr):
    plt.figure()
    plt.imshow(corr, cmap='RdYlGn', interpolation='none', aspect='auto')
    plt.colorbar()
    plt.grid(True, ls = '--', color = 'k')
    plt.xticks(range(len(corr)), corr.columns, rotation='vertical', fontsize = 8)
    plt.yticks(range(len(corr)), corr.columns, fontsize = 10);
    plt.suptitle('Label Correlations Heat Map', fontsize=15, fontweight='bold')
    plt.show()

#plot_corr(Y.corr())

## Returns indexes of rows containing this particular label and 
##                          indexes of rows not containing this label
## The list of indexes is returned as np.array in both the cases
def data_by_label(label):
    return Y[Y[label] == 1].index.values, Y[Y[label] == 0].index.values

## Returns shrinked data frame with only columns specified in labels
def selected_columns(labels):
    return Y[labels]

def get_data():
    return (filenames, Y)
