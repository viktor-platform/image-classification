"""Copyright (c) 2022 VIKTOR B.V.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
Software.

VIKTOR B.V. PROVIDES THIS SOFTWARE ON AN "AS IS" BASIS, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT
SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import json
import tensorflow as tf

# import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout
from tensorflow.keras.layers import GlobalMaxPooling2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model


def generate_cnn_model(epochs: int = 5):
    # Load in the data
    cifar10 = tf.keras.datasets.cifar10

    # Distribute it to train and test set
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Reduce pixel values / normalize
    x_train, x_test = x_train / 255.0, x_test / 255.0
    y_train, y_test = y_train.flatten(), y_test.flatten()

    AMOUNT_OF_CLASSES = len(set(y_train))

    # Build the model using the functional API
    # input layer
    i = Input(shape=x_train[0].shape)
    x = Conv2D(32, (3, 3), activation="relu", padding="same")(i)
    x = BatchNormalization()(x)
    x = Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    x = Flatten()(x)
    x = Dropout(0.2)(x)
    # Hidden layer
    x = Dense(1024, activation="relu")(x)
    x = Dropout(0.2)(x)
    # last hidden layer i.e.. output layer
    x = Dense(AMOUNT_OF_CLASSES, activation="softmax")(x)
    model = Model(i, x)
    # model description
    model.summary()

    # Compile
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # Fit
    r = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs)

    # The commented code below is used in the geeksforgeeks example, but is not necessary to use (it will
    # increase the duration)

    # Fit with data augmentation
    # Note: if you run this AFTER calling
    # the previous model.fit()
    # it will CONTINUE training where it left off
    # batch_size = 32
    # data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    #     width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
    #
    # train_generator = data_generator.flow(x_train, y_train, batch_size)
    # steps_per_epoch = x_train.shape[0] // batch_size
    #
    # r = model.fit(train_generator, validation_data=(x_test, y_test),
    #               steps_per_epoch=steps_per_epoch, epochs=epochs)

    # Plot accuracy per iteration (optional)
    # plt.plot(r.history['accuracy'], label='acc', color='red')
    # plt.plot(r.history['val_accuracy'], label='val_acc', color='green')
    # plt.legend()
    # plt.show()

    # save the model + history for later use in plots
    model.save(f"cnn_cifar10_model_{epochs}_epochs.h5")
    with open(f"cnn_cifar10_model_{epochs}_model_history.json", "w") as f:
        f.write(json.dumps(r.history))


# generate_cnn_model(epochs=2)
# generate_cnn_model(epochs=5)
# generate_cnn_model(epochs=10)
