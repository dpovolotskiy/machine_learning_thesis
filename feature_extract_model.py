import os
import shutil

from keras import Sequential, models, utils
from keras.layers import Input
from keras.layers.core import Flatten, Dense
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras import backend
from keras_applications.imagenet_utils import _obtain_input_shape

from utils import check_weights

LINK_FOR_DOWNLOAD_VGG16_WEIGHTS = "https://github.com/fchollet/deep-learning" \
                                  "-models/releases/download/v0.1/" \
                                  "vgg16_weights_tf_dim_ordering_tf_kernels.h5"


class VGG16(object):
    def __init__(self, path_to_weights=None, input_shape=None,
                 number_classes=1000):
        self.number_classes = number_classes
        self.path_to_weights = path_to_weights
        self.input_shape = input_shape
        self.model = Sequential()

    def get_model(self):
        input_shape = _obtain_input_shape(self.input_shape, default_size=224,
                                          min_size=32,
                                          data_format=backend.image_data_format(),
                                          require_flatten=True,
                                          weights="imagenet")
        img_input = Input(shape=input_shape)

        # Блок 1
        x = Conv2D(64, (3, 3), activation='relu', padding='same',
                   name='block1_conv1')(img_input)
        x = Conv2D(64, (3, 3), activation='relu', padding='same',
                   name='block1_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

        # Блок 2
        x = Conv2D(128, (3, 3), activation='relu', padding='same',
                   name='block2_conv1')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same',
                   name='block2_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

        # Блок 3
        x = Conv2D(256, (3, 3), activation='relu', padding='same',
                   name='block3_conv1')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same',
                   name='block3_conv2')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same',
                   name='block3_conv3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

        # Блок 4
        x = Conv2D(512, (3, 3), activation='relu', padding='same',
                   name='block4_conv1')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same',
                   name='block4_conv2')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same',
                   name='block4_conv3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

        # Блок 5
        x = Conv2D(512, (3, 3), activation='relu', padding='same',
                   name='block5_conv1')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same',
                   name='block5_conv2')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same',
                   name='block5_conv3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dense(4096, activation='relu', name='fc2')(x)
        x = Dense(1000, activation='softmax', name='predictions')(x)

        inputs = img_input

        self.model = models.Model(inputs, x, name="VGG16")

        if not (check_weights(self.path_to_weights)):
            directory_to_save_weights = self.path_to_weights[:-(
                len(os.path.basename(self.path_to_weights)) + 1)]
            self.model.load_weights(
                utils.get_file('vgg16_weights_tf_dim_ordering_tf_kernels.h5',
                               LINK_FOR_DOWNLOAD_VGG16_WEIGHTS,
                               cache_subdir='models',
                               file_hash='64373286793e3c8b2b4e3219cbf3544b',
                               cache_dir=directory_to_save_weights))
            shutil.move(
                directory_to_save_weights + r"\models" + r"\vgg16_weights_"
                                                         r"tf_dim_ordering_"
                                                         r"tf_kernels.h5",
                directory_to_save_weights)
            os.rmdir(directory_to_save_weights + r"\models")
        else:
            self.model.load_weights(self.path_to_weights)

        return self.model
