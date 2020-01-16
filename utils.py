import os


from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input
from pickle import load


def load_file(name):
    """
    функция используется для загрузки данных из файла указанного в параметре
    name
    """
    with open(name, "r") as data_file:
        data = data_file.read()
    return data


def check_weights(path):
    """
    функция используется для проверки существования файла по пути path
    """
    if os.path.exists(path):
        return True
    return False


def prepare_image_to_extracting_features(path_to_image):
    """
    функция использутется для подготовки изображения к извлечению признаков с
    помощью VGG16
    """
    result_image = load_img(path_to_image, target_size=(224, 224))
    result_image = img_to_array(result_image)
    result_image = result_image.reshape((1, result_image.shape[0],
                                         result_image.shape[1],
                                         result_image.shape[2]))

    result_image = preprocess_input(result_image)
    return result_image


def load_features_of_image(path, ids):
    """
    функция используется для загрузки файла с выделенными, при обучении модели,
    признаками изображений из файла
    """
    with open(path, "rb") as features_file:
        features = load(features_file)
        features = {image_id: features[image_id] for image_id in ids}
    return features
