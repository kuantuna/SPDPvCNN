import yfinance as yf
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from talib import RSI, WMA, EMA, SMA
from talib import ROC, CMO, CCI, PPO
from talib import TEMA, WILLR, MACD
from PIL import Image

startDate = '2011-10-11'
endDate = '2021-11-11'
coins = ['SPY']

spy_data = yf.download('SPY', start = startDate, end = endDate)

# DataFrame Preparation
rsi = RSI(spy_data["Close"], timeperiod=14).to_frame().reset_index().set_axis(['Date', 'Value'], axis=1)
wma = WMA(spy_data["Close"], timeperiod=30).to_frame().reset_index().set_axis(['Date', 'Value'], axis=1)
ema = EMA(spy_data["Close"], timeperiod=30).to_frame().reset_index().set_axis(['Date', 'Value'], axis=1)
sma = SMA(spy_data["Close"], timeperiod=30).to_frame().reset_index().set_axis(['Date', 'Value'], axis=1)
roc = ROC(spy_data["Close"], timeperiod=10).to_frame().reset_index().set_axis(['Date', 'Value'], axis=1)
cmo = CMO(spy_data["Close"], timeperiod=14).to_frame().reset_index().set_axis(['Date', 'Value'], axis=1)
cci = CCI(spy_data["High"], spy_data["Low"], spy_data["Close"], timeperiod=14).to_frame().reset_index().set_axis(['Date', 'Value'], axis=1)
ppo = PPO(spy_data["Close"], fastperiod=12, slowperiod=26, matype=0).to_frame().reset_index().set_axis(['Date', 'Value'], axis=1)
tema  = TEMA(spy_data["Close"], timeperiod=30).to_frame().reset_index().set_axis(['Date', 'Value'], axis=1)
willr = WILLR(spy_data["High"], spy_data["Low"], spy_data["Close"], timeperiod=14).to_frame().reset_index().set_axis(['Date', 'Value'], axis=1)
macd, macdsignal, macdhist = MACD(spy_data["Close"], fastperiod=12, slowperiod=26, signalperiod=9)
macd = macd.to_frame().reset_index().set_axis(['Date', 'Value'], axis=1)

# macdsignal = macdsignal.to_frame().reset_index().set_axis(['Date', 'Value'], axis=1)
# macdhist = macdhist.to_frame().reset_index().set_axis(['Date', 'Value'], axis=1)

# Dealing with missing values and replacing them by their mean
# rsi['Value'].fillna(rsi['Value'].mean(), inplace=True)
# wma['Value'].fillna(wma['Value'].mean(), inplace=True)
# ema['Value'].fillna(ema['Value'].mean(), inplace=True)
# sma['Value'].fillna(sma['Value'].mean(), inplace=True)
# roc['Value'].fillna(roc['Value'].mean(), inplace=True)
# cmo['Value'].fillna(cmo['Value'].mean(), inplace=True)
# cci['Value'].fillna(cci['Value'].mean(), inplace=True)
# ppo['Value'].fillna(ppo['Value'].mean(), inplace=True)
# tema['Value'].fillna(tema['Value'].mean(), inplace=True)
# willr['Value'].fillna(willr['Value'].mean(), inplace=True)
# macd['Value'].fillna(macd['Value'].mean(), inplace=True)
# macdsignal['Value'].fillna(macdsignal['Value'].mean(), inplace=True)
# macdhist['Value'].fillna(macdhist['Value'].mean(), inplace=True)

# Dealing with missing values
rsi = rsi.iloc[87:]
wma = wma.iloc[87:]
ema = ema.iloc[87:]
sma = sma.iloc[87:]
roc = roc.iloc[87:]
cmo = cmo.iloc[87:]
cci = cci.iloc[87:]
ppo = ppo.iloc[87:]
tema = tema.iloc[87:]
willr = willr.iloc[87:]
macd = macd.iloc[87:]

imageList = []
for index in range(len(rsi)-11):
    concat_ = pd.concat((rsi[index:index+11]['Value'], wma[index:index+11]['Value'], ema[index:index+11]['Value'], sma[index:index+11]['Value'],
                    roc[index:index+11]['Value'], cmo[index:index+11]['Value'], cci[index:index+11]['Value'], ppo[index:index+11]['Value'],
                    tema[index:index+11]['Value'], willr[index:index+11]['Value'], macd[index:index+11]['Value']), axis=1)
    concat_ = concat_.T
    imageList.append(np.array(concat_))

img = Image.fromarray(imageList[0])
img.show()