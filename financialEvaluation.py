import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_addons as tfa

batch_size = 128
auto = tf.data.AUTOTUNE

def load_dataset():
    imageList = np.load("ETF/Images.npy")
    labelList = np.load("ETF/Labels.npy")
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
    return dataset.prefetch(auto)

def get_finalized_datasets(new_x_train, new_y_train, x_val, y_val, x_test, y_test):
    train_dataset = make_datasets(new_x_train, new_y_train, is_train=True)
    val_dataset = make_datasets(x_val, y_val)
    test_dataset = make_datasets(x_test, y_test)
    return train_dataset, val_dataset, test_dataset

def load_saved_model(path):
    return keras.models.load_model(path, custom_objects={'MyOptimizer': tfa.optimizers.AdamW})

model = load_saved_model("SavedModels/test/1650031082-256x8-k5p3.h5")

imageList, labelList = load_dataset()
print_data_counts(labelList)

new_x_train, new_y_train, x_val, y_val, x_test, y_test = prepare_dataset(imageList, labelList)

train_dataset, val_dataset, test_dataset = get_finalized_datasets(new_x_train, new_y_train, x_val, y_val, x_test, y_test)

predictions = model.predict(test_dataset)
classes = np.argmax(predictions, axis=1)

def print_data_counts(labelList):
    labelDict = {
        0: "Buy",
        1: "Hold",
        2: "Sell"
    }
    unique, counts = np.unique(labelList, return_counts=True)
    for label, count in np.asarray((unique, counts)).T:
        print("{} count: {}".format(labelDict[int(label)], int(count)))

print_data_counts(classes)
