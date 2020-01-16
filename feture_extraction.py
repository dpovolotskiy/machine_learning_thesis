from os import listdir
from pickle import dump

from keras.models import Model
import keras

from utils import prepare_image_to_extracting_features


def extracting_features_from_image(image_dataset_directory):
    """
    функция используется для создания модели извлечения признаков из
    изображения, для удаления слоя классификации из модели и для извлечения
    признаков из указанного набора данных, который указывается с помощью
    параметра image_dataset_directory (str)
    """
    cnn_model = keras.applications.resnet.ResNet152(include_top=True,
                                                    weights='/home/dmitriy/'
                                                            'PycharmProjects/'
                                                            'machine_learning_'
                                                            'thesis/resnet152_'
                                                            'weights_tf_dim_'
                                                            'ordering_tf_'
                                                            'kernels.h5',
                                                    input_tensor=None,
                                                    input_shape=None,
                                                    pooling=None, classes=1000)

    cnn_model.layers.pop()
    cnn_model = Model(input=cnn_model.input,
                      outputs=cnn_model.layers[-1].output)

    print("Извлечение признаков из тренировочного набора данных начато! "
          "Это может занять несколько минут...\n")
    extracted_features = {}
    for image_name in listdir(image_dataset_directory):
        path_to_image = image_dataset_directory + r'/{}'.format(image_name)
        image = prepare_image_to_extracting_features(path_to_image)
        feature = cnn_model.predict(image)
        image_id = image_name.split(".")[0]
        extracted_features[image_id] = feature
        print("Извлечение завершено для изображения с именем {}".format(
            image_name))
    dump(extracted_features, open('features.pkl', 'wb'))
    print("Извлечение признаков завершено!")
