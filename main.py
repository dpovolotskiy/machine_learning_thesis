from sklearn.metrics import log_loss

from VGG19 import vgg19
from load_training_data import load_cifar10_data

vgg = vgg19()
model = vgg.get_model()
img_rows, img_cols = 224, 224
channel = 3
num_classes = 10
batch_size = 16
nb_epoch = 1
X_train, Y_train, X_valid, Y_valid = load_cifar10_data(img_rows, img_cols)
model.summary()
model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, shuffle=True, verbose=1, validation_data=(X_valid,
                                                                                                              Y_valid))
predictions_valid = model.predict(X_valid, batch_size=batch_size, verbose=1)
score = log_loss(Y_valid, predictions_valid)
"""
model2 = keras.applications.vgg19.VGG19(include_top=True, weights='imagenet', input_tensor=None, input_shape=(224, 224, 3), pooling="maxpooling", classes=1000)
model2.summary()
"""