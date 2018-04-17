from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input


def get_model(img_width = 256, img_height = 256, channels = 3, classes = 4, \
                trainable_from='add_4', softmax = False):

# create the base pre-trained model
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape = (img_width, img_height, channels))

    names = [l.name for l in base_model.layers]
    print names
    for layer in base_model.layers[:names.index(trainable_from) + 1]:
        layer.trainable = False
# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional Resnet layers

# add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)

# let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)

    if softmax:
        predictions = Dense(classes, activation='softmax')(x)
    else:
        predictions = Dense(classes, activation='sigmoid')(x)
# this is the model we will train
    model = Model(input=base_model.input, output=predictions)

    return model
