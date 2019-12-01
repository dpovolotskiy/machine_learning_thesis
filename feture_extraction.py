from os import listdir
from pickle import dump

from keras.applications.imagenet_utils import preprocess_input
from keras.models import Model
from keras_preprocessing.image import load_img, img_to_array

from feature_extract_model import VGG16


def extracting_features_from_image(image_dataset_directory):
    vgg = VGG16(r"F:\My Documents\SSU\diplom\vgg16_weights_tf_dim_ordering_tf_kernels.h5")
    model = vgg.get_model()

    model.layers.pop()
    model = Model(input=model.input, outputs=model.layers[-1].output)

    features = {}
    for image_name in listdir(image_dataset_directory):
        path_to_image = image_dataset_directory + r'\{}'.format(image_name)
        image = load_img(path_to_image, target_size=(224, 224))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        # может быть неправильная функция, посмотреть, если возникнут проблемы!!!
        image = preprocess_input(image)
        feature = model.predict(image, verbose=0)
        image_id = image_name.split(".")[0]
        features[image_id] = feature
        print('>%s' % image_name)
    dump(features, open('features.pkl', 'wb'))


directory = 'Flicker8k_Dataset'
extracting_features_from_image(directory)