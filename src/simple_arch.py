from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K

import numpy as np

###################################
#### Constants ###################


##################################

def run(X, Y, Xtest = None):
    _,img_width, img_height,_ = X.shape
    _, classes = Y.shape

#validation_data_dir = 'data/validation'
#nb_train_samples = 2000
#nb_validation_samples = 800
    epochs = 10
    batch_size = 16

#if K.image_data_format() == 'channels_first':
#    input_shape = (3, img_width, img_height)
#else:
    input_shape = (img_width, img_height, 3)

    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(classes))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator()

# this is the augmentation configuration we will use for testing:
# only rescaling


    model.fit(X, Y, epochs=10, verbose=1, validation_split=0.2, shuffle=True)

    #model.fit(X,Y,epochs=25)
    #a = model.predict(X)
#exp_scores = np.exp(a)
#probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
#ypred = np.argmax(probs, axis = 1)
#Y = np.argmax(Y, axis = 1)

#from sklearn.metrics import confusion_matrix, accuracy_score
#acc = accuracy_score(Y, ypred)
#print acc
    xval = X[:int(0.2 * len(X))]
    yval = model.predict(xval)
    ytrue = Y[:int(0.2 * len(X))]
    
    return (ytrue, yval, model.predict(Xtest))
