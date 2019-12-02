from keras_preprocessing.text import Tokenizer
from tqdm import tqdm

from training_preparation import train_generator


def training(final_model, captions, extracted_features, maximal_length,
             vocabulary_size,
             epochs=10):
    number_of_steps = len(captions)
    for i in tqdm(range(epochs)):
        fitting_generator = train_generator(captions, extracted_features,
                                            Tokenizer, maximal_length,
                                            vocabulary_size)
        final_model.fit_generator(fitting_generator, epochs=1,
                                  steps_per_epoch=number_of_steps, verbose=1)
        final_model.save("model_epoch_" + str(i) + ".h5")
