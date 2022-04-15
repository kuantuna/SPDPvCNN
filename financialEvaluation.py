import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_addons as tfa


class Wallet:
    def __init__(self, base_currency: str, stock: str, initial_money: float):
        self.base_currency: str = base_currency
        self.stock: str = stock
        self.values: dict = {base_currency: initial_money, stock: 0}

    def buy(self, stock_price: float):
        if self.values[self.base_currency] == 0:
            return
        self.values[self.stock] = self.values[self.base_currency] / \
            stock_price - 1
        self.values[self.base_currency] = 0
        print("Bought {} {}".format(self.values[self.stock], self.stock))

    def sell(self, stock_price: float):
        if self.values[self.stock] == 0:
            return
        self.values[self.base_currency] = (
            self.values[self.stock] - 1) * stock_price
        self.values[self.stock] = 0
        print("Sold {} {}".format(self.values[self.stock], self.stock))

    def print_values(self):
        print(self.values)


wallet = Wallet("USD", "S&P500", 10000)


batch_size = 128
auto = tf.data.AUTOTUNE


def load_dataset():
    imageList = np.load("ETF/Images(BIG).npy")
    labelList = np.load("ETF/Labels0038.npy")
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


# def prepare_dataset(imageList, labelList):
#     x_train, x_test, y_train, y_test = train_test_split(
#         imageList, labelList, test_size=0.1, random_state=41)
#     val_split = 0.1

#     val_indices = int(len(x_train) * val_split)
#     new_x_train, new_y_train = x_train[val_indices:], y_train[val_indices:]
#     x_val, y_val = x_train[:val_indices], y_train[:val_indices]

#     print(f"Training data samples: {len(new_x_train)}")
#     print(f"Validation data samples: {len(x_val)}")
#     print(f"Test data samples: {len(x_test)}")

#     return new_x_train, new_y_train, x_val, y_val, x_test, y_test


# def make_datasets(images, labels, is_train=False):
#     dataset = tf.data.Dataset.from_tensor_slices((images, labels))
#     if is_train:
#         dataset = dataset.shuffle(batch_size * 10)
#     dataset = dataset.batch(batch_size)
#     return dataset.prefetch(auto)


# def get_finalized_datasets(new_x_train, new_y_train, x_val, y_val, x_test, y_test):
#     train_dataset = make_datasets(new_x_train, new_y_train, is_train=True)
#     val_dataset = make_datasets(x_val, y_val)
#     test_dataset = make_datasets(x_test, y_test)
#     return train_dataset, val_dataset, test_dataset


def load_saved_model(path):
    return keras.models.load_model(path, custom_objects={'MyOptimizer': tfa.optimizers.AdamW})


model = load_saved_model("SavedModels/1604/1650059581-256x8-k7p5.h5")

imageList, labelList = load_dataset()
print_data_counts(labelList)

dataset = tf.data.Dataset.from_tensor_slices((imageList, labelList))
dataset = dataset.batch(batch_size)
dataset = dataset.prefetch(auto)

predictions = model.predict(dataset)
classes = np.argmax(predictions, axis=1)

wallet.print_values()

for cl in classes:
    if cl == 0:
        wallet.buy(1)
    elif cl == 1:
        pass
    elif cl == 2:
        wallet.sell(1)

wallet.print_values()
