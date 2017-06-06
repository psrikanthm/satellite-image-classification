
def encode2label(array_x):
    a = []
    for x in array_x:
        a.append(sum(1<<i for i, b in enumerate(x[::-1]) if b))
   
    a = np.array(a)
    import sklearn.preprocessing
    label_binarizer = sklearn.preprocessing.LabelBinarizer()
    label_binarizer.fit(range(max(a)+1))
    return label_binarizer.transform(a)
