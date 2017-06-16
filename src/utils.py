import numpy as np

def encode2label(array_x):
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
