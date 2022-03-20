import yfinance as yf
import pandas as pd
import numpy as np

from talib import RSI, WMA, EMA, SMA
from talib import ROC, CMO, CCI, PPO
from talib import TEMA, WILLR, MACD

from talib import SAR, ADX, STDDEV, OBV

'''
DEFINING SOME VARIABLES
'''
startDate = '2001-10-11'
endDate = '2022-03-19'
axes = ['Date', 'Value']
headers = ['RSI', 'WMA', 'EMA', 'SMA', 'ROC', 'CMO', 'CCI', 'PPO', 'TEMA', 'WILLR', 'MACD', 'SAR', 'ADX', 'STDDEV', 'OBV']
etfList = ['XLF', 'XLU', 'QQQ', 'SPY', 'XLP', 'EWZ', 'EWH', 'XLY', 'XLE']
threshold = 0.0038 # Re-arrange the Threshold Value
imageList = []
labelList = []

'''
DOWNLOADING THE DATA
'''
# DataFrame, size=(n_days, 6), col_names=["Open", "High", "Low", "Close", "Adj Close", "Volume"]
for etf in etfList:
    data = yf.download(etf, start = startDate, end = endDate)

    '''
    CALCULATING THE INDICATOR VALUES
    '''
    # DataFrame, size=(n_days, 2), col_names=["Date", "Value"]
    rsi = RSI(data["Close"], timeperiod=14).to_frame().reset_index().set_axis(axes, axis=1)
    wma = WMA(data["Close"], timeperiod=30).to_frame().reset_index().set_axis(axes, axis=1)
    ema = EMA(data["Close"], timeperiod=30).to_frame().reset_index().set_axis(axes, axis=1)
    sma = SMA(data["Close"], timeperiod=30).to_frame().reset_index().set_axis(axes, axis=1)
    roc = ROC(data["Close"], timeperiod=10).to_frame().reset_index().set_axis(axes, axis=1)
    cmo = CMO(data["Close"], timeperiod=14).to_frame().reset_index().set_axis(axes, axis=1)
    cci = CCI(data["High"], data["Low"], data["Close"], timeperiod=14).to_frame().reset_index().set_axis(axes, axis=1)
    ppo = PPO(data["Close"], fastperiod=12, slowperiod=26, matype=0).to_frame().reset_index().set_axis(axes, axis=1)
    tema  = TEMA(data["Close"], timeperiod=30).to_frame().reset_index().set_axis(axes, axis=1)
    willr = WILLR(data["High"], data["Low"], data["Close"], timeperiod=14).to_frame().reset_index().set_axis(axes, axis=1)
    macd, macdsignal, macdhist = MACD(data["Close"], fastperiod=12, slowperiod=26, signalperiod=9)
    macd = macd.to_frame().reset_index().set_axis(axes, axis=1)

    sar = SAR(data["High"], data["Low"], acceleration=0, maximum=0).to_frame().reset_index().set_axis(axes, axis=1)
    adx = ADX(data["High"], data["Low"], data["Close"], timeperiod=14).to_frame().reset_index().set_axis(axes, axis=1)
    std = STDDEV(data['Close'], timeperiod=5, nbdev=1).to_frame().reset_index().set_axis(axes, axis=1)
    obv = OBV(data['Close'], data['Volume']).to_frame().reset_index().set_axis(axes, axis=1)

    '''
    PREPROCESSING INDICATOR DATA
    '''
    # List of (indicators) DataFrames, size=n_indicators
    indicators = [rsi, cmo, willr, cci, macd, roc, ppo, std, tema, obv, wma, ema, sma, adx, sar]
    # [rsi, wma, ema, sma, roc, cmo, cci, ppo, tema, willr, macd, sar, adx, std, obv]

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
    nDays = len(indicatorValues[0])
    for idx in range(nDays-nIndicators):
        # List, size=n_indicators, contains imageRows of size (n_indicators, 1)
        image = []
        for indicatorValue in indicatorValues:
            # NumPy Array, size=(n_indicators, 1)
            imageRow = indicatorValue[idx:idx+nIndicators][..., np.newaxis]
            image.append(imageRow)
        imageList.append(np.array(image))

    '''
    CREATING THE LABELS
    '''
    # Pandas Series, size=n_days-(maxNullVal+nIndicators-1) -> Check this, size is imageList+1, might be a bug.
    data_close = data[maxNullVal+nIndicators-1:]["Close"]

    # Buy : 0
    # Hold: 1
    # Sell: 2 
    for i in range(len(data_close)-1):
        closePriceDifference = data_close.iloc[i+1] - data_close.iloc[i]
        thresholdPrice = threshold * data_close.iloc[i]
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

imageList = np.array(imageList)
labelList = np.array(labelList)

unique, counts = np.unique(labelList, return_counts=True)
print(np.asarray((unique, counts)).T)

imageList_copy = imageList[:]
imageList_copy = imageList_copy.reshape(len(imageList), -1)
mean = np.mean(imageList_copy, axis=0)
std = np.std(imageList_copy, axis=0)
imageList_copy = (imageList_copy - mean) / std
imageList = imageList_copy.reshape(len(imageList), len(indicators), len(indicators), 1)

np.save("./ETF/Images.npy", imageList)
np.save("./ETF/Labels.npy", labelList)