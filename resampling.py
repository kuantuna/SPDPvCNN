import numpy as np
import pandas as pd

from architectures.helpers.constants import hyperparameters
from architectures.helpers.constants import etf_list
from architectures.helpers.constants import threshold
from architectures.helpers.constants import selected_model
from architectures.helpers.wandb_handler import initialize_wandb
from architectures.helpers.custom_callbacks import CustomCallback


def load_dataset():
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    for etf in etf_list:
        x_train.extend(
            np.load(f"ETF/strategy/{threshold}/TrainData/x_{etf}.npy"))
        y_train.extend(
            np.load(f"ETF/strategy/{threshold}/TrainData/y_{etf}.npy"))
        x_test.extend(
            np.load(f"ETF/strategy/{threshold}/TestData/x_{etf}.npy"))
        y_test.extend(
            np.load(f"ETF/strategy/{threshold}/TestData/y_{etf}.npy"))
    return x_train, y_train, x_test, y_test


x_train, y_train, x_test, y_test = load_dataset()
train_unique, train_counts = np.unique(y_train, return_counts=True)
print(np.asarray((train_unique, train_counts)).T)
test_unique, test_counts = np.unique(y_test, return_counts=True)
print(np.asarray((test_unique, test_counts )).T)

x_train_new = []
y_train_new = []

for x_t, y_t in zip(x_train, y_train):
    if y_t != 1:
        x_train_new.append(x_t)
        y_train_new.append(y_t)
        x_train_new.append(x_t)
        y_train_new.append(y_t)

x_test_new = []
y_test_new = []

for x_t, y_t in zip(x_test, y_test):
    if y_t != 1:
        x_test_new.append(x_t)
        y_test_new.append(y_t)
        x_test_new.append(x_t)
        y_test_new.append(y_t)

x_test.extend(x_test_new)
y_test.extend(y_test_new)

train_unique, train_counts = np.unique(y_train, return_counts=True)
print(np.asarray((train_unique, train_counts)).T)
test_unique, test_counts = np.unique(y_test, return_counts=True)
print(np.asarray((test_unique, test_counts )).T)
