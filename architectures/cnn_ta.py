from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

import architectures.helpers.constants as constants

hyperparameters = constants.hyperparameters["cnn_ta"]


def get_ct_model():
    print("Getting the CNN_TA model...")
    model = Sequential()
    model.add(Conv2D(32, hyperparameters["kernel_size"], activation='relu',
                     input_shape=hyperparameters["input_shape"]))
    model.add(Conv2D(64, hyperparameters["kernel_size"], activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(hyperparameters["first_dropout_rate"]))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(hyperparameters["second_dropout_rate"]))
    model.add(Dense(hyperparameters["num_classes"], activation='softmax'))
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=[
                      keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
                      keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy")
                    ])
    return model
