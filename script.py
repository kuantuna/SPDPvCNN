import yfinance as yf
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from talib import RSI, WMA, EMA, SMA
from talib import ROC, CMO, CCI, PPO
from talib import TEMA, WILLR, MACD
from PIL import Image

from tensorflow.keras import layers
from tensorflow import keras
import tensorflow_addons as tfa
import tensorflow as tf
from sklearn.model_selection import train_test_split

'''
DEFINING SOME VARIABLES
'''
startDate = '2001-10-11'
endDate = '2021-11-11'
axes = ['Date', 'Value']
headers = ['RSI', 'WMA', 'EMA', 'SMA', 'ROC', 'CMO', 'CCI', 'PPO', 'TEMA', 'WILLR', 'MACD']
threshold = 0.003
# coins = ['SPY']


'''
DOWNLOADING THE DATA
'''
# DataFrame, size=(n_days, 6), col_names=["Open", "High", "Low", "Close", "Adj Close", "Volume"]
spy_data = yf.download('SPY', start = startDate, end = endDate)


'''
CALCULATING THE INDICATOR VALUES
'''
# DataFrame, size=(n_days, 2), col_names=["Date", "Value"]
rsi = RSI(spy_data["Close"], timeperiod=14).to_frame().reset_index().set_axis(axes, axis=1)
wma = WMA(spy_data["Close"], timeperiod=30).to_frame().reset_index().set_axis(axes, axis=1)
ema = EMA(spy_data["Close"], timeperiod=30).to_frame().reset_index().set_axis(axes, axis=1)
sma = SMA(spy_data["Close"], timeperiod=30).to_frame().reset_index().set_axis(axes, axis=1)
roc = ROC(spy_data["Close"], timeperiod=10).to_frame().reset_index().set_axis(axes, axis=1)
cmo = CMO(spy_data["Close"], timeperiod=14).to_frame().reset_index().set_axis(axes, axis=1)
cci = CCI(spy_data["High"], spy_data["Low"], spy_data["Close"], timeperiod=14).to_frame().reset_index().set_axis(axes, axis=1)
ppo = PPO(spy_data["Close"], fastperiod=12, slowperiod=26, matype=0).to_frame().reset_index().set_axis(axes, axis=1)
tema  = TEMA(spy_data["Close"], timeperiod=30).to_frame().reset_index().set_axis(axes, axis=1)
willr = WILLR(spy_data["High"], spy_data["Low"], spy_data["Close"], timeperiod=14).to_frame().reset_index().set_axis(axes, axis=1)
macd, macdsignal, macdhist = MACD(spy_data["Close"], fastperiod=12, slowperiod=26, signalperiod=9)
macd = macd.to_frame().reset_index().set_axis(axes, axis=1)


'''
PREPROCESSING INDICATOR DATA
'''
# List of (indicators) DataFrames, size=n_indicators
indicators = [rsi, wma, ema, sma, roc, cmo, cci, ppo, tema, willr, macd]

# Number of indicators (int)
nIndicators = len(indicators)

# Calculating the most number of null values in an indicator DataFrame's "Value" column
maxNullVal = -1
for indicator in indicators:
    if(indicator['Value'].isnull().sum() > maxNullVal):
        maxNullVal = indicator['Value'].isnull().sum()

# List of (indicators "Value" column) DataFrames, size=n_indicators
indicatorValues = []
for indicator in indicators:
    indicatorValues.append(indicator['Value'].iloc[maxNullVal:]) # Getting rid of null values
    
# DataFrame, size=(n_days, n_indicators, col_names=headers)
indicatorValuesMatrix = pd.concat(indicatorValues, axis=1, keys = headers)
indicatorCorr = indicatorValuesMatrix.corr(method = 'pearson')


'''
CREATING THE IMAGES
'''
# NumPy Array, size=(n_days, n_indicators, n_indicators, 1)
imageList = []
nDays = len(indicatorValues[0])
for idx in range(nDays-nIndicators):
    # List, size=n_indicators, contains imageRows of size (n_indicators, 1)
    image = []
    for indicatorValue in indicatorValues:
        # NumPy Array, size=(n_indicators, 1)
        imageRow = indicatorValue[idx:idx+nIndicators][..., np.newaxis]
        image.append(imageRow)
    imageList.append(np.array(image))
imageList = np.array(imageList)


'''
CREATING THE LABELS
'''
# Pandas Series, size=n_days-(maxNullVal+nIndicators-1) -> Check this, size is imageList+1, might be a bug.
spy_data_close = spy_data[maxNullVal+nIndicators-1:]["Close"]

# NumPy Array, size=(n_days, 1)
labelList = []
# Buy : 0
# Hold: 1
# Sell: 2 
for i in range(len(spy_data_close)-1):
    closePriceDifference = spy_data_close.iloc[i+1] - spy_data_close.iloc[i]
    thresholdPrice = threshold * spy_data_close.iloc[i]
    # If the price has increased
    if(closePriceDifference > 0):
        # but not enough to pass the threshold
        if(closePriceDifference <= thresholdPrice):
            labelList.append(np.array([1.0])) # HOLD
        # enough to pass the threshold
        else:
            labelList.append(np.array([0.0])) # BUY
    # If the price has decreased
    elif(closePriceDifference < 0):
        # but not so much to pass the thresshold
        if(abs(closePriceDifference) <= thresholdPrice):
            labelList.append(np.array([1.0])) # HOLD
        # so much to pass the threshold
        else:
            labelList.append(np.array([2.0])) # SELL
    # If the price hasn't changed
    else:
        labelList.append(np.array([1.0])) # HOLD
labelList = np.array(labelList)


'''
IMPLEMENTING THE CONVMIXER
Reference: (https://github.com/keras-team/keras-io/blob/master/examples/vision/convmixer.py)
'''
learning_rate = 0.001
weight_decay = 0.0001
batch_size = 128
num_epochs = 10


x_train, x_test, y_train, y_test = train_test_split(imageList, labelList, test_size=0.1, random_state=100)
val_split = 0.1

val_indices = int(len(x_train) * val_split)
new_x_train, new_y_train = x_train[val_indices:], y_train[val_indices:]
x_val, y_val = x_train[:val_indices], y_train[:val_indices]

print(f"Training data samples: {len(new_x_train)}")
print(f"Validation data samples: {len(x_val)}")
print(f"Test data samples: {len(x_test)}")


image_size = 11
auto = tf.data.AUTOTUNE

data_augmentation = keras.Sequential(
    [layers.RandomCrop(image_size, image_size), layers.RandomFlip("horizontal"),],
    name="data_augmentation",
)

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

train_dataset = make_datasets(new_x_train, new_y_train, is_train=True)
val_dataset = make_datasets(x_val, y_val)
test_dataset = make_datasets(x_test, y_test)


def activation_block(x):
    x = layers.Activation("gelu")(x)
    return layers.BatchNormalization()(x)


def conv_stem(x, filters: int, patch_size: int):
    x = layers.Conv2D(filters, kernel_size=patch_size, strides=patch_size)(x)
    return activation_block(x)


def conv_mixer_block(x, filters: int, kernel_size: int):
    # Depthwise convolution.
    x0 = x
    x = layers.DepthwiseConv2D(kernel_size=kernel_size, padding="same")(x)
    x = layers.Add()([activation_block(x), x0])  # Residual.

    # Pointwise convolution.
    x = layers.Conv2D(filters, kernel_size=1)(x)
    x = activation_block(x)

    return x


def get_conv_mixer_256_8(
    image_size=11, filters=256, depth=8, kernel_size=3, patch_size=2, num_classes=3
):
    """ConvMixer-256/8: https://openreview.net/pdf?id=TVHS5Y4dNvM.
    The hyperparameter values are taken from the paper.
    """
    inputs = keras.Input((image_size, image_size, 1))
    x = layers.Rescaling(scale=1.0 / 255)(inputs)

    # Extract patch embeddings.
    x = conv_stem(x, filters, patch_size)

    # ConvMixer blocks.
    for _ in range(depth):
        x = conv_mixer_block(x, filters, kernel_size)

    # Classification block.
    x = layers.GlobalAvgPool2D()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs, outputs)


def run_experiment(model):
    optimizer = tfa.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    checkpoint_filepath = "C:\\Users\\Tuna\\Desktop\\2021-2022_Fall\\CS401\\Results" # fix here
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
    )

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=num_epochs,
        callbacks=[checkpoint_callback],
    )

    model.load_weights(checkpoint_filepath)
    _, accuracy = model.evaluate(test_dataset)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")

    return history, model

conv_mixer_model = get_conv_mixer_256_8()
history, conv_mixer_model = run_experiment(conv_mixer_model)

