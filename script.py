import yfinance as yf
from talib import *
from matplotlib import pyplot as plt
import numpy as np

startDate = '2011-11-07'
endDate = '2021-11-07'
coins = ['SPY']

spy_data = yf.download('SPY', start=startDate, end=endDate)

rsi = RSI(spy_data["Close"], timeperiod=14)
willr = WILLR(spy_data["High"], spy_data["Low"], spy_data["Close"], timeperiod=14)
wma = WMA(spy_data["Close"], timeperiod=30)
ema = EMA(spy_data["Close"], timeperiod=30)
sma = SMA(spy_data["Close"], timeperiod=30)
# hma yok !?
tema = TEMA(spy_data["Close"], timeperiod=30)
cci = CCI(spy_data["High"], spy_data["Low"], spy_data["Close"], timeperiod=14)
cmo = CMO(spy_data["Close"], timeperiod=14)
macd, macdsignal, macdhist = MACD(spy_data["Close"], fastperiod=12, slowperiod=26, signalperiod=9)
ppo = PPO(spy_data["Close"], fastperiod=12, slowperiod=26, matype=0)
roc = ROC(spy_data["Close"], timeperiod=10)
# cmfi yok !?
# dmi yok !?
# psi yok !?

rsi = rsi.reset_index()
print(rsi)
