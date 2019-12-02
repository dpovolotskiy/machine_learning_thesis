from utils import creating_of_sequences


def train_generator(captions, images, keras_tokenizer, maximal_length, vocabulary_size):
    while True:
        for image_id, captions in captions.items():
            image = images[image_id][0]
            input_image, input_sequences, output_words = creating_of_sequences(keras_tokenizer, maximal_length, captions, image, vocabulary_size)
            yield [[input_image, input_sequences], output_words]
