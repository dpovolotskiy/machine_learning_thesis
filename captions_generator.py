from keras import models
from numpy import argmax
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from pickle import load


from feature_extract_model import VGG16
from utils import prepare_image_to_extracting_features


def transform_to_word(integer, keras_tokenizer):
    """
    функция используется для преобразования числа в словов, с помощью tokenizer
    """
    for word, idx in keras_tokenizer.word_index.items():
        if idx == integer:
            return word
    return None


def generate_caption(model, keras_tokenizer, image, maximal_length):
    """
    функция используется для генерации описания изображения
    """
    in_text = "startseq"
    for _ in range(maximal_length):
        seq = keras_tokenizer.texts_to_sequences([in_text])[0]
        seq = pad_sequences([seq], maxlen=maximal_length)

        prediction = model.predict([image, seq], verbose=0)
        prediction = argmax(prediction)

        word = transform_to_word(prediction, keras_tokenizer)
        if word is None:
            break
        in_text += " " + word
        if word == "endseq":
            break
    return in_text


def clean_output_caption(predict):
    """
    функция используется для удаления из созданного описания начального и конечного флагов
    """
    spl_predict = predict.split()
    result = " ".join(spl_predict[1:-1])
    return result


def one_image_feature_extracting(path):
    """
    функция использутеся для извлечения признаков изображения в режиме
    создания описания
    """
    model = VGG16("/machine_learning_thesis/"
                  "vgg16_weights_tf_dim_"
                  "ordering_tf_kernels.h5").get_model()
    model.layers.pop()
    model = models.Model(input=model.input, outputs=model.layers[-1].output)
    image = prepare_image_to_extracting_features(path)
    feature = model.predict(image, verbose=0)
    return feature


def get_predict(path_to_image, path_to_model=None):
    """
    функция используется в качестве основной функции генерации текстового
    описания изображения, выполняет загрузку предобученной модели описания
    """
    print("Создание описания начато!")
    keras_tokenizer = load(open("keras_tokenizer.pkl", "rb"))
    with open("maximal_length.txt", "r") as max_len_file:
        maximal_length = int(max_len_file.read())
    if path_to_model is None:
        model = load_model("my_model.h5")
    else:
        model = load_model(path_to_model)
    image_features = one_image_feature_extracting(path_to_image)
    caption = generate_caption(model, keras_tokenizer, image_features,
                               maximal_length)
    caption = clean_output_caption(caption)
    print("Описание для указанного изображения {}: {}".format(path_to_image, caption))
    return caption


def get_predict_web(path_to_image, path_to_model=None):
    """
    функция используется в качестве основной функции генерации текстового
    описания изображения, выполняет загрузку предобученной модели описания
    для использования в веб версии интерфейса
    """
    print("Prediction was started!")
    keras_tokenizer = load(open("keras_tokenizer.pkl", "rb"))
    with open("maximal_length.txt", "r") as max_len_file:
        maximal_length = int(max_len_file.read())
    if path_to_model is None:
        model = load_model("my_model_9.h5")
    else:
        model = load_model(path_to_model)
    image_features = one_image_feature_extracting(path_to_image)
    caption = generate_caption(model, keras_tokenizer, image_features,
                               maximal_length)
    caption = clean_output_caption(caption)
    return caption
