from keras import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD


class vgg19(object):
    def __init__(self, path_to_weights=None, image_rows=224, image_colons=224, channel=3, number_classes=10):
        self.image_rows = image_rows
        self.image_colons = image_colons
        self.channel = channel
        self.number_classes = number_classes
        self.path_to_weights = path_to_weights
        self.model = Sequential()

    def get_model(self):
        self.model.add(ZeroPadding2D((1, 1), input_shape=(self.channel, self.image_rows, self.image_colons)))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D((2, 2), strides=(2, 2), data_format="channels_first"))

        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Conv2D(128, (3, 3), activation='relu'))
        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Conv2D(128, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D((2, 2), strides=(2, 2), data_format="channels_first"))

        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Conv2D(256, (3, 3), activation='relu'))
        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Conv2D(256, (3, 3), activation='relu'))
        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Conv2D(256, (3, 3), activation='relu'))
        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Conv2D(256, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D((2, 2), strides=(2, 2), data_format="channels_first"))

        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Conv2D(512, (3, 3), activation='relu'))
        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Conv2D(512, (3, 3), activation='relu'))
        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Conv2D(512, (3, 3), activation='relu'))
        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Conv2D(512, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D((2, 2), strides=(2, 2), data_format="channels_first"))

        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Conv2D(512, (3, 3), activation='relu'))
        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Conv2D(512, (3, 3), activation='relu'))
        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Conv2D(512, (3, 3), activation='relu'))
        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Conv2D(512, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D((2, 2), strides=(2, 2), data_format="channels_first"))

        self.model.add(Flatten())
        self.model.add(Dense(4096, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(4096, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1000, activation='softmax'))

        if self.path_to_weights is not None:
            self.model.load_weights(self.path_to_weights)

        self.model.layers.pop()
        self.model.outputs = [self.model.layers[-1].output]
        self.model.layers[-1].outbound_nodes = []
        self.model.add(Dense(self.number_classes, activation='softmax'))

        sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)

        self.model.compile(optimizer=sgd, loss='categorical_crossentropy')

        return self.model
