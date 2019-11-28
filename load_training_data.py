import random

import cv2
import numpy as np

from keras.datasets import cifar10
from keras.utils import to_categorical

nb_train_samples = 3000 # 3000 training samples
nb_test_samples = 100 # 100 validation samples
num_classes = 10


def load_cifar10_data(img_rows, img_cols):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # limit the amount of the data
    # train data
    ind_train = random.sample(list(range(x_train.shape[0])), nb_train_samples)
    x_train = x_train[ind_train]
    y_train = y_train[ind_train]

    # test data
    ind_test = random.sample(list(range(x_test.shape[0])), nb_test_samples)
    x_test = x_test[ind_test]
    y_test = y_test[ind_test]

    def resize_data(data):
        data_upscaled = np.zeros((data.shape[0], 3, 224, 224))
        for i, img in enumerate(data):
            large_img = cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_CUBIC).transpose(2, 0, 1)
            data_upscaled[i] = large_img
        return data_upscaled

    x_train_resized = resize_data(x_train)
    x_test_resized = resize_data(x_test)

    y_train_hot_encoded = to_categorical(y_train)
    y_test_hot_encoded = to_categorical(y_test)

    return x_train_resized, y_train_hot_encoded, x_test_resized, y_test_hot_encoded