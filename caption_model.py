from keras import Model
from keras.layers import Input, Dropout, Dense, Embedding
from keras.layers import LSTM
from keras.layers import add


def model_for_captions(vocabulary_size, maximal_length):
    input1 = Input(shape=(4096,))
    feature_extraction1 = Dropout(0.5)(input1)
    feature_extraction2 = Dense(256, activation="relu")(feature_extraction1)

    input2 = Input(shape=(maximal_length,))
    sequence_model1 = Embedding(vocabulary_size, 256)(input2)
    sequence_model2 = Dropout(0.5)(sequence_model1)
    sequence_model3 = LSTM(256)(sequence_model2)

    decode1 = add([feature_extraction2, sequence_model3])
    decode2 = Dense(256, activation="relu")(decode1)

    output = Dense(vocabulary_size, activation="softmax")(decode2)

    caption_model = Model(inputs=[input1, input2], outputs=output)

    caption_model.compile(loss="categorical_crossentropy", optimizer="adam")

    caption_model.summary()

    return caption_model
