import csv
import numpy as np
import pandas as pd

## Constants
DATA_LOCATION = "../data/"
Y_CSV = "train.csv"
##

#Read the data and parse into a Matrix
ifile = open(DATA_LOCATION + Y_CSV, "rb")
rows = csv.reader(ifile)
next(rows)
#Y = np.zeros((sum(1 for row in rows),N_LABELS))

count = 0
ydata = []

for row in rows:
    count += 1
    ydata.append(row[1].split())

labels = list(set([item for sublist in ydata for item in sublist]))

d = []
for label in labels:
    d.append((label,'float64'))

Y = np.zeros((len(ydata),len(d)))

iteration = 0
for y in ydata:
    for x in y:
        Y[iteration, labels.index(x)] = 1
    iteration += 1

Y = pd.DataFrame(data = Y, columns = labels)

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
