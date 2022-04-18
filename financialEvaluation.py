import numpy as np
from tensorflow import keras
import tensorflow as tf
import tensorflow_addons as tfa


class Wallet:
    def __init__(self, base_currency_name: str, stock_name: str, initial_money: float):
        self.base_currency_name: str = base_currency_name
        self.stock_name: str = stock_name
        self.info: dict = {base_currency_name: initial_money, stock_name: 0,
                           "buy_count": 0, "hold_count": 0, "sell_count": 0}
        #self.transactions: list = []

    def buy(self, stock_price: float, date: str):
        if self.info[self.base_currency_name] == 0:
            return
        self.info["buy_count"] += 1
        stock = (self.info[self.base_currency_name] - 1) / \
            stock_price
        # print(
        #     f"Bought {self.stock_name}: {round(stock, 2)} | USD: 0 | price: {round(stock_price, 2)} | date: {date}")
        self.info[self.stock_name] = stock
        self.info[self.base_currency_name] = 0

    def hold(self):
        self.info["hold_count"] += 1
        return

    def sell(self, stock_price: float, date: str):
        if self.info[self.stock_name] == 0:
            return
        self.info["sell_count"] += 1
        base = self.info[self.stock_name] * stock_price - 1
        # print(
        #     f"Sold   {self.stock_name}: 0 | USD: {round(base, 2)} | price: {round(stock_price, 2)} | date: {date}")
        self.info[self.base_currency_name] = base
        self.info[self.stock_name] = 0

    def print_values(self):
        print(self.info)


batch_size: int = 128
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


def make_dataset(imageList, labelList):
    datasets = []
    # keeps the images and labels for every stock one by one (datasets[0] == images & labels for etfList[0])
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
        datasets.append(dataset)
    return datasets


"""Loading the necessary stuff"""
model = load_saved_model("SavedModels/1604/1650059581-256x8-k7p5.h5")


listOfDates: list[np.ndarray] = []
listOfPrices: list[np.ndarray] = []
# keeps the prices for every stock one by one (listOfPrices[0] == prices for etfList[0])
etfList: list[str] = ['XLF', 'XLU', 'QQQ',
                      'SPY', 'XLP', 'EWZ', 'EWH', 'XLY', 'XLE']
for etf in etfList:
    listOfDates.append(np.load(f"ETF/New/date-{etf}.npy"))
    listOfPrices.append(np.load(f"ETF/New/{etf}.npy"))


imageList, labelList = load_dataset()
print_data_counts(labelList)
datasets = make_dataset(imageList, labelList)

listOfSignals = []
for dataset in datasets:
    predictions = model.predict(dataset)
    listOfSignals.append(np.argmax(predictions, axis=1))


"""Main algorithm"""
for signals, etf, price, dates in zip(listOfSignals, etfList, listOfPrices, listOfDates):
    wallet = Wallet("USD", etf, 10000)
    wallet.print_values()
    for signal, price, date in zip(signals, price, dates):
        if signal == 0:
            wallet.buy(price, date)
        elif signal == 1:
            wallet.hold()
        elif signal == 2:
            wallet.sell(price, date)
    wallet.print_values()
    print("\n")
