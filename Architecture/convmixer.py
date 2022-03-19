import numpy as np
import os
import tensorflow as tf
import tensorflow_addons as tfa
import time
import wandb

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, regularizers
from tensorflow import keras
from wandb.keras import WandbCallback

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

'''
CONVMIXER
Reference: (https://github.com/keras-team/keras-io/blob/master/examples/vision/convmixer.py)
'''


''' Hyperparameters
'''
learning_rate = 0.001
weight_decay = 0.0001
batch_size = 16
num_epochs = 60
filters_ = 768
depth = 32
kernel_size = 5
patch_size = 1


def initialize_wandb():
    wandb.init(project="test-project", entity="spdpvcnn",
            config={
                "model": "ConvMixer",
                "learning_rate": learning_rate,
                "epochs": num_epochs,
                "batch_size": batch_size,
                "weight_decay": weight_decay,
                "filters": filters_,
                "depth": depth,
                "kernel_size": kernel_size,
                "patch_size": patch_size,
                "threshold": 0.0038
            })



def load_dataset():
    imageList = np.load("../ETF/Images.npy")
    labelList = np.load("../ETF/Labels.npy")
    return imageList, labelList

def print_data_counts(labelList):
    labelDict = {
        0: "Buy",
        1: "Hold",
        2: "Sell"
    }
    unique, counts = np.unique(labelList, return_counts=True)
    for label, count in np.asarray((unique, counts)).T:
        print("{} count: {}".format(labelDict[int(label)], int(count)))


def prepare_dataset(imageList, labelList):
    x_train, x_test, y_train, y_test = train_test_split(imageList, labelList, test_size=0.1, random_state=41)
    val_split = 0.1

    val_indices = int(len(x_train) * val_split)
    new_x_train, new_y_train = x_train[val_indices:], y_train[val_indices:]
    x_val, y_val = x_train[:val_indices], y_train[:val_indices]

    print(f"Training data samples: {len(new_x_train)}")
    print(f"Validation data samples: {len(x_val)}")
    print(f"Test data samples: {len(x_test)}")

    return new_x_train, new_y_train, x_val, y_val, x_test, y_test


def make_datasets(images, labels, is_train=False):
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    if is_train:
        dataset = dataset.shuffle(batch_size * 10)
    dataset = dataset.batch(batch_size)
    if is_train:
        dataset = dataset.map(
            lambda x, y: (data_augmentation(x), y), num_parallel_calls=auto
        )
    return dataset.prefetch(auto)

def get_finalized_datasets(new_x_train, new_y_train, x_val, y_val, x_test, y_test):
    train_dataset = make_datasets(new_x_train, new_y_train, is_train=True)
    val_dataset = make_datasets(x_val, y_val)
    test_dataset = make_datasets(x_test, y_test)
    return train_dataset, val_dataset, test_dataset


def load_saved_model(path):
    return keras.models.load_model(path, custom_objects={'MyOptimizer': tfa.optimizers.AdamW})


def activation_block(x):
    x = layers.Activation("gelu")(x)
    return layers.BatchNormalization()(x)


def conv_stem(x, filters: int, patch_size: int):
    x = layers.Conv2D(filters, kernel_size=patch_size, strides=patch_size)(x)
    #, kernel_regularizer=regularizers.l2(1e-2)
    return activation_block(x)


def conv_mixer_block(x, filters: int, kernel_size: int):
    # Depthwise convolution.
    x0 = x
    x = layers.DepthwiseConv2D(kernel_size=kernel_size, padding="same")(x)
    x = layers.Add()([activation_block(x), x0])  # Residual.

    # Pointwise convolution.
    x = layers.Conv2D(filters, kernel_size=1)(x)
    #, kernel_regularizer=regularizers.l2(1e-2)
    x = activation_block(x)

    return x


def get_conv_mixer_model(
        image_size=11, filters=filters_, depth=depth, kernel_size=kernel_size, patch_size=patch_size, num_classes=3
):
    """ConvMixer-256/8: https://openreview.net/pdf?id=TVHS5Y4dNvM.
    The hyperparameter values are taken from the paper.
    """
    inputs = keras.Input((image_size, image_size, 1))
    # x = layers.Rescaling(scale=1.0 / 255)(inputs)

    # Extract patch embeddings.
    x = conv_stem(inputs, filters, patch_size)

    # ConvMixer blocks.
    for _ in range(depth):
        x = conv_mixer_block(x, filters, kernel_size)

    # Classification block.
    x = layers.GlobalAvgPool2D()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs, outputs)


class CmPrinter(tf.keras.callbacks.Callback):
    def __init__(self, test_dataset) -> None:
        super().__init__()
        self.test_dataset = test_dataset


    def on_epoch_end(self, epoch, logs={}):
        predictions = self.model.predict(self.test_dataset)
        classes = np.argmax(predictions, axis=1)
        print(confusion_matrix(y_test, classes))# Compute and store recall for each class here.


def compile_model_optimizer(model):
    optimizer = tfa.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )
    # scheduled_lrs

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def run_experiment(model, test_dataset):
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=num_epochs,
        callbacks=[WandbCallback(), CmPrinter(test_dataset)],
    )

    t = time.time()
    export_path_keras = "../SavedModels/{}-{}x{}-k{}p{}.h5".format(int(t), filters_, depth, kernel_size, patch_size)
    model.save(export_path_keras)
    _, accuracy = model.evaluate(test_dataset)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")

    return history, model



if __name__ == "__main__":
    initialize_wandb()
    imageList, labelList = load_dataset()
    print_data_counts(labelList)

    new_x_train, new_y_train, x_val, y_val, x_test, y_test = prepare_dataset(imageList, labelList)

    image_size = 11
    auto = tf.data.AUTOTUNE

    data_augmentation = keras.Sequential(
        [layers.RandomCrop(image_size, image_size), layers.RandomFlip("horizontal"), ],
        name="data_augmentation",
    )

    train_dataset, val_dataset, test_dataset = get_finalized_datasets(new_x_train, new_y_train, x_val, y_val, x_test, y_test)


    conv_mixer_model = get_conv_mixer_model()                                                  # If you want to load a saved model and train it
    conv_mixer_model = compile_model_optimizer(conv_mixer_model)                               # Comment these two lines and uncomment the line below
    # conv_mixer_model = load_saved_model('C:/Users/Tuna/Desktop/Saved Models/1647585460.h5')  

    history, conv_mixer_model = run_experiment(conv_mixer_model, test_dataset)

    predictions = conv_mixer_model.predict(test_dataset)
    classes = np.argmax(predictions, axis=1)

    cr = classification_report(y_test, classes)
    print(cr)
    f1 = f1_score(y_test, classes, average='micro')
    print(f1)
