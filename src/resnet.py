from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input


def get_model(img_width, img_height, channels, classes):

# create the base pre-trained model
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape = (img_width, img_height, channels))

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional Resnet layers
    for layer in base_model.layers:
        layer.trainable = False

# add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)

# let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(classes, activation='softmax')(x)

# this is the model we will train
    model = Model(input=base_model.input, output=predictions)

    return model
