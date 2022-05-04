from tensorflow import keras
from tensorflow.keras import layers, regularizers
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

import time

THRESHOLD = "01"
MODEL_PATH = "1650312655-256x8-k7p5e"

learning_rate = 0.001
weight_decay = 0.0001
batch_size = 128
num_epochs = 500
filters_ = 256
depth = 8
kernel_size = 7
patch_size = 5

image_size = 67
auto = tf.data.AUTOTUNE

TOTAL_STEPS = int((50000 / batch_size) * num_epochs)
WARMUP_STEPS = 10000
INIT_LR = 0.01
WAMRUP_LR = 0.002


class WarmUpCosine(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(
        self, learning_rate_base, total_steps, warmup_learning_rate, warmup_steps
    ):
        super(WarmUpCosine, self).__init__()

        self.learning_rate_base = learning_rate_base
        self.total_steps = total_steps
        self.warmup_learning_rate = warmup_learning_rate
        self.warmup_steps = warmup_steps
        self.pi = tf.constant(np.pi)

    def __call__(self, step):
        if self.total_steps < self.warmup_steps:
            raise ValueError(
                "Total_steps must be larger or equal to warmup_steps.")
        learning_rate = (
            0.5
            * self.learning_rate_base
            * (
                1
                + tf.cos(
                    self.pi
                    * (tf.cast(step, tf.float32) - self.warmup_steps)
                    / float(self.total_steps - self.warmup_steps)
                )
            )
        )

        if self.warmup_steps > 0:
            if self.learning_rate_base < self.warmup_learning_rate:
                raise ValueError(
                    "Learning_rate_base must be larger or equal to "
                    "warmup_learning_rate."
                )
            slope = (
                self.learning_rate_base - self.warmup_learning_rate
            ) / self.warmup_steps
            warmup_rate = slope * \
                tf.cast(step, tf.float32) + self.warmup_learning_rate
            learning_rate = tf.where(
                step < self.warmup_steps, warmup_rate, learning_rate
            )
        return tf.where(
            step > self.total_steps, 0.0, learning_rate, name="learning_rate"
        )


scheduled_lrs = WarmUpCosine(
    learning_rate_base=INIT_LR,
    total_steps=TOTAL_STEPS,
    warmup_learning_rate=WAMRUP_LR,
    warmup_steps=WARMUP_STEPS,
)

t = time.time()
epoch_counter = 1

''' ConvMixer Implementation
'''


def activation_block(x):
    x = layers.Activation("gelu")(x)
    return layers.BatchNormalization()(x)


def conv_stem(x, filters: int, patch_size: int):
    x = layers.Conv2D(filters, kernel_size=patch_size,
                      strides=patch_size, kernel_regularizer=regularizers.l2(1e-2))(x)
    return activation_block(x)


def conv_mixer_block(x, filters: int, kernel_size: int):
    # Depthwise convolution.
    x0 = x
    x = layers.DepthwiseConv2D(kernel_size=kernel_size, padding="same")(x)
    x = layers.Add()([activation_block(x), x0])  # Residual.

    # Pointwise convolution.
    x = layers.Conv2D(filters, kernel_size=1,
                      kernel_regularizer=regularizers.l2(1e-2))(x)
    x = activation_block(x)

    return x


def get_conv_mixer_model(
        image_size=image_size, filters=filters_, depth=depth, kernel_size=kernel_size, patch_size=patch_size, num_classes=3
):
    """ConvMixer-256/8: https://openreview.net/pdf?id=TVHS5Y4dNvM.
    The hyperparameter values are taken from the paper.
    """
    inputs = keras.Input((image_size, image_size, 1))

    # Extract patch embeddings.
    x = conv_stem(inputs, filters, patch_size)

    # ConvMixer blocks.
    for _ in range(depth):
        x = conv_mixer_block(x, filters, kernel_size)

    # Classification block.
    x = layers.GlobalAvgPool2D()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs, outputs)


''' Compiling, Training and Evaluating
'''


def compile_model_optimizer(model):
    optimizer = tfa.optimizers.AdamW(
        learning_rate=scheduled_lrs, weight_decay=weight_decay
    )

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


class Wallet:
    def __init__(self, base_currency_name: str, stock_name: str, initial_money: float):
        self.base_currency_name: str = base_currency_name
        self.stock_name: str = stock_name
        self.initial_money: float = initial_money
        self.info: dict = {base_currency_name: initial_money, stock_name: 0, f"v_{base_currency_name}": initial_money, f"v_{stock_name}": 0,
                           "buy_count": 0, "hold_count": 0, "sell_count": 0}
        self.profit_percentage: float = 0
        #self.transactions: list = []

    def buy(self, stock_price: float, date: str):
        if self.info[self.base_currency_name] == 0:
            return
        self.info["buy_count"] += 1
        v_base = (self.info[self.base_currency_name] - 1)
        stock = v_base / stock_price
        # print(
        #     f"Bought {self.stock_name}: {round(stock, 2)} | USD: 0 | price: {round(stock_price, 2)} | date: {date}")
        self.info[self.stock_name] = stock
        self.info[f"v_{self.stock_name}"] = stock
        self.info[self.base_currency_name] = 0
        self.info[f"v_{self.base_currency_name}"] = v_base
        self.profit_percentage = v_base / self.initial_money - 1

    def hold(self):
        self.info["hold_count"] += 1
        return

    def sell(self, stock_price: float, date: str):
        if self.info[self.stock_name] == 0:
            return
        self.info["sell_count"] += 1
        base = self.info[self.stock_name] * stock_price - 1
        v_stock = base / stock_price
        # print(
        #     f"Sold   {self.stock_name}: 0 | USD: {round(base, 2)} | price: {round(stock_price, 2)} | date: {date}")
        self.info[self.base_currency_name] = base
        self.info[f"v_{self.base_currency_name}"] = base
        self.info[self.stock_name] = 0
        self.info[f"v_{self.stock_name}"] = v_stock
        self.profit_percentage = base / self.initial_money - 1

    def print_values(self):
        # if(self.profit_percentage > 0):
        print(self.info)
        print(f"Profit percentage: {self.profit_percentage}")


batch_size: int = 128
auto = tf.data.AUTOTUNE


def load_dataset():
    x_test = []
    y_test = []
    for etf in etfList:
        x_test.append(np.load(f"ETF/{THRESHOLD}/TestData/x_{etf}.npy"))
        y_test.append(np.load(f"ETF/{THRESHOLD}/TestData/y_{etf}.npy"))
    return x_test, y_test


# def print_data_counts(labelList):
#     labelDict = {
#         0: "Buy",
#         1: "Hold",
#         2: "Sell"
#     }
#     unique, counts = np.unique(labelList, return_counts=True)
#     for label, count in np.asarray((unique, counts)).T:
#         print("{} count: {}".format(labelDict[int(label)], int(count)))


def load_saved_model(path):
    return keras.models.load_model(path, custom_objects={'MyOptimizer': tfa.optimizers.AdamW})


def make_dataset(x_test, y_test):
    datasets = []
    # keeps the images and labels for every stock one by one (datasets[0] == images & labels for etfList[0])
    for xt, yt in zip(x_test, y_test):
        dataset = tf.data.Dataset.from_tensor_slices((xt, yt))
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(auto)
        datasets.append(dataset)
    return datasets


"""Loading the necessary stuff"""

listOfDates: list[np.ndarray] = []
listOfPrices: list[np.ndarray] = []
# keeps the prices for every stock one by one (listOfPrices[0] == prices for etfList[0])
etfList: list[str] = ['XLF', 'XLU', 'QQQ',
                      'SPY', 'XLP', 'EWZ', 'EWH', 'XLY', 'XLE']
for etf in etfList:
    listOfDates.append(
        np.load(f"ETF/{THRESHOLD}/Date/TestDate/{etf}.npy", allow_pickle=True))
    listOfPrices.append(
        np.load(f"ETF/{THRESHOLD}/Price/TestPrice/{etf}.npy", allow_pickle=True))


x_test, y_test = load_dataset()
# print_data_counts(labelList)
datasets = make_dataset(x_test, y_test)

profit_ranking = []

for i in [178, 486, 16, 462, 368, 403, 325, 383, 394, 461, 389, 297]:
    model = get_conv_mixer_model()
    model = compile_model_optimizer(model)
    model.load_weights(
        f"SavedModels/{THRESHOLD}/{MODEL_PATH}{i}.h5")
    listOfSignals = []
    for dataset in datasets:
        predictions = model.predict(dataset)
        listOfSignals.append(np.argmax(predictions, axis=1))

    print(f"MODEL{i}")
    """Main algorithm"""
    profits = []
    for signals, etf, price, dates in zip(listOfSignals, etfList, listOfPrices, listOfDates):
        wallet = Wallet("USD", etf, 10000)
        for signal, price, date in zip(signals, price, dates):
            if signal == 0:
                wallet.buy(price, date)
            elif signal == 1:
                wallet.hold()
            elif signal == 2:
                wallet.sell(price, date)
        wallet.print_values()
        # print("\n")
        profits.append(wallet.profit_percentage)
    mpp = np.mean(profits)
    print(f"Model profit percentage: {mpp}\n")
    profit_ranking.append({"mpp": mpp, "model": i})

sorted_pr = sorted(profit_ranking, key=lambda d: d['mpp'], reverse=True)
print(sorted_pr)
"""create a list of model values from sorted_pr"""
model_values = [d['model'] for d in sorted_pr]
print(model_values)
