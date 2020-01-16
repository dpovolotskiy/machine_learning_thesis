from keras_preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from numpy import array
from keras.callbacks import ModelCheckpoint
from pickle import dump

from utils import load_file, load_features_of_image
from caption_model import model_for_captions


def loading_of_photo_id(path):
    """
    функция используется для извлечения id изображений из файла переданного в
    параметре path
    """
    data = load_file(path)
    ids = []
    for row in data.split("\n"):
        if len(row) < 1:
            continue
        image_id = row.split(".")[0]
        ids.append(image_id)
    return set(ids)


def loading_cleaned_captions(path, ids):
    """
    функция используется для загрузки сохраненных, приведенных к единому виду
    описаний и формирования словаря, в котором каждому id изображения
    соответствует пять описаний на английском языке, с добавлением начальных
    и конечных токенов
    """
    data = load_file(path)
    clean_captions = {}
    for row in data.split("\n"):
        captions = row.split()
        image_id = captions[0]
        image_descriptions = captions[1:]
        if image_id in ids:
            if image_id not in clean_captions:
                clean_captions[image_id] = []
            caption = "startseq " + " ".join(image_descriptions) + " endseq"
            clean_captions[image_id].append(caption)
    return clean_captions


def captions_to_list(dict_of_captions):
    """
    функция используется для преобразования словаря, состоящего из id
    изображений и соответствующих им описаний, в список, без id изображений
    """
    list_of_captions = []
    for image_id in dict_of_captions.keys():
        [list_of_captions.append(caption) for caption in dict_of_captions[
            image_id]]
    return list_of_captions


def create_keras_tokenizer(captions):
    """
    функция используется для создания tokenizer, и его обучения на наборе
    описаний
    """
    list_of_captions = captions_to_list(captions)
    keras_tokenizer = Tokenizer()
    keras_tokenizer.fit_on_texts(list_of_captions)
    return keras_tokenizer


def calculate_max_caption_len(captions):
    """
    функция используется для определения максимально длинны описания
    """
    list_of_captions = captions_to_list(captions)
    max_len = max(len(caption.split()) for caption in list_of_captions)
    return max_len


def creating_of_sequences(keras_tokenizer, maximal_length, captions, images,
                          vocabulary_size):
    """
    функция используется для подготовки данных для подачи на вход при обучении
    модели стандартным образом
    """
    X1 = []
    X2 = []
    y = []
    for image_id, list_of_captions in captions.items():
        for caption in list_of_captions:
            sequence = keras_tokenizer.texts_to_sequences([caption])[0]
            for i in range(1, len(sequence)):
                input_sequence = sequence[:i]
                output_sequence = sequence[i]
                input_sequence = \
                    pad_sequences([input_sequence], maxlen=maximal_length)[0]
                output_sequence = \
                    to_categorical([output_sequence],
                                   num_classes=vocabulary_size)[
                        0]
                X1.append(images[image_id][0])
                X2.append(input_sequence)
                y.append(output_sequence)
    return array(X1), array(X2), array(y)


def lite_creating_of_sequences(keras_tokenizer, maximal_length, captions,
                               images, vocabulary_size):
    """
    функция используется для подготовки данных для подачи на вход при обучении
    модели облегчённым образом
    """
    X1 = []
    X2 = []
    y = []
    for caption in captions:
        sequence = keras_tokenizer.texts_to_sequences([caption])[0]
        for i in range(1, len(sequence)):
            input_sequence = sequence[:i]
            output_sequence = sequence[i]
            input_sequence = \
                pad_sequences([input_sequence], maxlen=maximal_length)[0]
            output_sequence = \
                to_categorical([output_sequence],
                               num_classes=vocabulary_size)[0]
            X1.append(images)
            X2.append(input_sequence)
            y.append(output_sequence)
    return array(X1), array(X2), array(y)


def start_fit_model(epochs=10):
    """
    функция используется для старта обчения модели описания изображений обычным
    методом, выполняет подготовку тренировочного и валидационного наборов
    данных, создает модель описания, и начинает обучения модели, сохраняя лучшую
    модель в файл "my_model.h5"
    """
    print("Начата загрузка тренировочного набора данных! "
          "Это может занять несколько минут...")
    path_to_training_dataset = "Flickr8k_text/Flickr_8k.trainImages.txt"
    train_ids = loading_of_photo_id(path_to_training_dataset)
    train_captions = loading_cleaned_captions("captions.txt", train_ids)
    train_image_features = load_features_of_image("features.pkl", train_ids)
    keras_tokenizer = create_keras_tokenizer(train_captions)
    dump(keras_tokenizer, open("keras_tokenizer.pkl", "wb"))
    vocabulary_size = len(keras_tokenizer.word_index) + 1
    maximal_length = calculate_max_caption_len(train_captions)
    with open("maximal_length.txt", "w") as max_len_file:
        max_len_file.write(str(maximal_length))
    X1train, X2train, ytrain = creating_of_sequences(keras_tokenizer,
                                                     maximal_length,
                                                     train_captions,
                                                     train_image_features,
                                                     vocabulary_size)
    print("Загрузка тренировочного набора данных завершена!")

    print("Начата загрузка набора данных для тестирования! "
          "Это может занять несколько минут...")
    path_to_testing_dataset = "Flickr8k_text/Flickr_8k.devImages.txt"
    test_ids = loading_of_photo_id(path_to_testing_dataset)
    test_captions = loading_cleaned_captions("captions.txt", test_ids)
    test_features = load_features_of_image("features.pkl", test_ids)
    X1test, X2test, ytest = creating_of_sequences(keras_tokenizer,
                                                  maximal_length,
                                                  test_captions,
                                                  test_features,
                                                  vocabulary_size)
    print("Загрузка набора данных для тестирования завершена!")

    model = model_for_captions(vocabulary_size, maximal_length)

    checkpoint = ModelCheckpoint("my_model.h5", monitor="val_loss", verbose=1,
                                 save_best_only=True, mode="min")
    model.fit([X1train, X2train], ytrain, epochs=epochs, verbose=2,
              callbacks=[checkpoint], validation_data=([X1test, X2test], ytest))


def data_fit_generator(captions, images, keras_tokenizer, maximal_length,
                       vocabulary_size):
    """
    функция используется для определения генератора обучающих данных для
    облегчённого метода обучения
    """
    while True:
        for image_id, captions in captions.items():
            image_feature = images[image_id][0]
            input_image, input_sequence, output_word = \
                lite_creating_of_sequences(keras_tokenizer,
                                           maximal_length,
                                           captions,
                                           image_feature,
                                           vocabulary_size)
            yield [[input_image, input_sequence], output_word]


def lite_train(training_captions, training_features, keras_tokenizer,
               maximal_length, vocabulary_size, epochs=10):
    """
    функция используется для создания модели описания при облегчённом методе
    обучения, запускает обучение модели, и сохраняет резьльтат после каждой
    эпохи в фалй "my_model_{номер эпохи}.h5"
    """
    model = model_for_captions(vocabulary_size, maximal_length)
    number_of_steps_per_epoch = len(training_captions)
    for i in range(epochs):
        gen = data_fit_generator(training_captions, training_features,
                                 keras_tokenizer, maximal_length,
                                 vocabulary_size)
        model.fit_generator(gen, epochs=1,
                            steps_per_epoch=number_of_steps_per_epoch,
                            verbose=1)
        model.save("my_model_" + str(i) + ".h5")


def start_lite_train(epochs):
    """
    функция используется для подготовки тренировочного набора данных для
    обучения модели в облегчённом режиме, и вызывает функцию обучения модели
    облегченным методом
    """
    print("Начата загрузка тренировочного набора данных! "
          "Это может занять несколько минут...")
    path_to_training_dataset = "Flickr8k_text/Flickr_8k.trainImages.txt"
    train_ids = loading_of_photo_id(path_to_training_dataset)
    train_captions = loading_cleaned_captions("captions.txt", train_ids)
    train_image_features = load_features_of_image("features.pkl", train_ids)
    keras_tokenizer = create_keras_tokenizer(train_captions)
    dump(keras_tokenizer, open("keras_tokenizer.pkl", "wb"))
    vocabulary_size = len(keras_tokenizer.word_index) + 1
    maximal_length = calculate_max_caption_len(train_captions)
    with open("maximal_length.txt", "w") as max_len_file:
        max_len_file.write(str(maximal_length))
    print("Загрузка тренировочного набора данных завершена!")
    lite_train(train_captions, train_image_features, keras_tokenizer,
               maximal_length, vocabulary_size, epochs=epochs)
