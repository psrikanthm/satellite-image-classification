from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Flatten
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input


def get_model(img_width = 256, img_height = 256, channels = 3, classes = 4, \
                trainable_from='mixed9', softmax = False):

    # create the base pre-trained model
    base_model = VGG19(weights='imagenet', include_top=False, pooling = 'avg', \
                    input_shape = (img_width, img_height, channels))

    x = base_model.output
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    
    if softmax:
        x = Dense(classes, activation='softmax')(x)
    else:
        x = Dense(classes, activation='sigmoid')(x)

    model = Model(input=base_model.input, output=x)
    return model

def print_layer_names():
    base_model = InceptionV3(weights='imagenet')
    return [l.name for l in base_model.layers]
