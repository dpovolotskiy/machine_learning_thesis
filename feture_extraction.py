from os import listdir
from pickle import dump


from keras.models import Model


from feature_extract_model import VGG16
from utils import prepare_image_to_extracting_features


def extracting_features_from_image(image_dataset_directory):
    vgg = VGG16("/machine_learning_thesis/"
                "vgg16_weights_tf_dim_ordering_tf_kernels.h5")
    model = vgg.get_model()

    model.layers.pop()
    model = Model(input=model.input, outputs=model.layers[-1].output)


    print("Extracting features from training dataset was started. "
          "It may takes several minutes...")
    extracted_features = {}
    for image_name in listdir(image_dataset_directory):
        path_to_image = image_dataset_directory + r'/{}'.format(image_name)
        image = prepare_image_to_extracting_features(path_to_image)
        feature = model.predict(image)
        image_id = image_name.split(".")[0]
        extracted_features[image_id] = feature
        print("Extracting finished for image with name {}".format(image_name))
    dump(extracted_features, open('features.pkl', 'wb'))
    print("Extracting features was finished!")
