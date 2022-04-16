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
        # print("Bought {} {}".format(self.values[self.stock], self.stock))

    def sell(self, stock_price: float):
        if self.values[self.stock] == 0:
            return
        self.values[self.base_currency] = (
            self.values[self.stock] - 1) * stock_price
        self.values[self.stock] = 0
        # print("Sold {} {}".format(self.values[self.stock], self.stock))

    def print_values(self):
        print(self.values)


batch_size = 128
auto = tf.data.AUTOTUNE


def load_dataset():
    imageList = np.load("ETF/New/Images.npy")
    labelList = np.load("ETF/New/Labels00038.npy")
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


def load_saved_model(path):
    return keras.models.load_model(path, custom_objects={'MyOptimizer': tfa.optimizers.AdamW})


prices = []
etfList = ['XLF', 'XLU', 'QQQ', 'SPY', 'XLP', 'EWZ', 'EWH', 'XLY', 'XLE']
for etf in etfList:
    prices.append(np.load(f"ETF/New/{etf}.npy"))


def make_dataset(imageList, labelList):
    comp_dataset = []
    start = 0
    end = 5010
    for _ in etfList:
        img = imageList[start:end]
        lbl = labelList[start:end]
        start = end
        end += 5010
        dataset = tf.data.Dataset.from_tensor_slices((img, lbl))
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(auto)
        comp_dataset.append(dataset)
    return comp_dataset


model = load_saved_model("SavedModels/1604/1650059581-256x8-k7p5.h5")

imageList, labelList = load_dataset()
print_data_counts(labelList)
datasets = make_dataset(imageList, labelList)

for dataset, etf, price in zip(datasets, etfList, prices):
    predictions = model.predict(dataset)
    classes = np.argmax(predictions, axis=1)

    wallet = Wallet("USD", etf, 10000)
    wallet.print_values()

    for cl, p in zip(classes, price):
        if cl == 0:
            wallet.buy(p)
        elif cl == 1:
            pass
        elif cl == 2:
            wallet.sell(p)

    wallet.print_values()
