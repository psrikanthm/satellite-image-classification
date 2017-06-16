from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input


def get_model(img_width = 256, img_height = 256, channels = 3, classes = 4, trainable_from='mixed9'):

# create the base pre-trained model
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape = (img_width, img_height, channels))
  
    names = [l.name for l in base_model.layers]
    for layer in base_model.layers[:names.index(trainable_from) + 1]:
        layer.trainable = False

# add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)

# let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(classes, activation='softmax')(x)

# this is the model we will train
    model = Model(input=base_model.input, output=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
    
    return model

def print_layer_names():
    base_model = InceptionV3(weights='imagenet')
    return [l.name for l in base_model.layers]
