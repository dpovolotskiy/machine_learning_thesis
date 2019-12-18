from os import listdir
from pickle import dump


from keras.models import Model


from feature_extract_model import VGG16
from utils import prepare_image_to_extracting_features


def extracting_features_from_image(image_dataset_directory):
    """
    функция используется для создания модели извлечения признаков из изображения,
    для удаления слоя классификации из модели и для извлечения признаков из указанного набора данных,
    который указывается с помощью параметра image_dataset_directory (str)
    """
    vgg = VGG16("/machine_learning_thesis/"
                "vgg16_weights_tf_dim_ordering_tf_kernels.h5")
    model = vgg.get_model()

    model.layers.pop()
    model = Model(input=model.input, outputs=model.layers[-1].output)


    print("Извлечение признаков из тренировочного набора данных начато! Это может занять несколько минут...\n")
    extracted_features = {}
    for image_name in listdir(image_dataset_directory):
        path_to_image = image_dataset_directory + r'/{}'.format(image_name)
        image = prepare_image_to_extracting_features(path_to_image)
        feature = model.predict(image)
        image_id = image_name.split(".")[0]
        extracted_features[image_id] = feature
        print("Извлечение завершено для изображения с именем {}".format(image_name))
    dump(extracted_features, open('features.pkl', 'wb'))
    print("Извлечение признаков завершено!")
