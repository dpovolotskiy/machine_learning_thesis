import os
import string

from keras.utils import to_categorical
from keras_preprocessing.sequence import pad_sequences
from numpy import array


def load_file(name):
    with open(name, "r") as data_file:
        data = data_file.read()
    return data


def extracting_captions(data):
    captions = {}
    for caption in data.split("\n"):
        image_descriptions = caption.split()
        if len(caption) < 2:
            continue
        image_id = image_descriptions[0].split(".")[0]
        image_description = " ".join(image_descriptions[1:])
        if image_id not in captions:
            captions[image_id] = []
        captions[image_id].append(image_description)
    return captions


def cleaning_captions(captions):
    translation_table = str.maketrans("", "", string.punctuation)
    for image_id, list_of_captions in captions.items():
        for i in range(len(list_of_captions)):
            caption = list_of_captions[i]
            caption = caption.split()
            caption = [word.lower() for word in caption]
            caption = [word.translate(translation_table) for word in caption]
            caption = [word for word in caption if len(word) > 1]
            caption = [word for word in caption if word.isalpha()]
            list_of_captions[i] = " ".join(caption)


def saving_ready_captions(captions, filename_to_save):
    lines = []
    for image_id, list_of_captions in captions.items():
        for caption in list_of_captions:
            lines.append(image_id + " " + caption)
    data = "\n".join(lines)
    with open(filename_to_save, "w") as saving_file:
        saving_file.write(data)


def creating_of_sequences(keras_tokenizer, maximal_length, captions, photos, vocabulary_size):
    x1 = []
    x2 = []
    y = []
    for image_id, list_of_captions in captions.items():
        for caption in list_of_captions:
            sequence = keras_tokenizer.text_to_sequences([caption])[0]
            for i in range(1, len(sequence)):
                input_sequence = sequence[:i]
                output_sequence = sequence[i]
                input_sequence = pad_sequences([input_sequence], maxlen=maximal_length)[0]
                output_sequence = to_categorical([output_sequence], num_classes=vocabulary_size)[0]
                x1.append(photos[image_id][0])
                x2.append(input_sequence)
                y.append(output_sequence)
    return array(x1), array(x2), array(y)


def check_weights(path):
    if os.path.exists(path):
        return True
    return False
