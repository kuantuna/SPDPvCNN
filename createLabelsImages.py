import yfinance as yf
import pandas as pd
import numpy as np
import talib as tb

'''
DEFINING SOME VARIABLES
'''
startDate = '2001-10-11'
endDate = '2022-03-19'
axes = ['Date', 'Value']
headers = ['RSI', 'WMA', 'EMA', 'SMA', 'ROC', 'CMO', 'CCI', 'PPO', 'TEMA', 'WILLR', 'MACD', 'SAR', 'ADX', 'STDDEV', 'OBV', 'ADXR', 'APO', 'AROONDOWN', 'AROONUP', 'AROONOSC', 
           'BOP', 'DX', 'MACDEXT', 'MACDFİX', 'MFI', 'MINUS_DI', 'MINUS_DM', 'MOM', 'PLUS_DI', 'PLUS_DM', 'ROCP', 'ROCR', 'ROCR100','SLOWK', 'SLOWD', 'FASTK', 'FASTD', 
           'FASTKRSI', 'FASTDRSI', 'TRIX', 'ULTOSC', 'CDLXSIDEGAP3METHODS', 'CDLUPSIDEGAP2CROWS', 'CDLUNIQUE3RIVER', 'CDLTRISTAR', 'CDLTHRUSTING', 'CDLTASUKIGAP', 'CDLTAKURI', 
           'CDLSTICKSANDWICH', 'CDLSTALLEDPATTERN', 'CDLSPINNINGTOP', 'CDLSHORTLINE', 'CDLSHOOTINGSTAR', 'CDLSEPARATINGLINES', 'CDLRISEFALL3METHODS', 'CDLRICKSHAWMAN', 
           'CDLPIERCING', 'CDLONNECK', 'CDLMORNINGSTAR','CDLMORNINGDOJISTAR', 'CDLMATHOLD', 'CDLMATCHINGLOW', 'CDLMARUBOZU', 'CDLLONGLINE', 'CDLLONGLEGGEDDOJI', 'CDLLADDERBOTTOM',
           'CDLKICKINGBYLENGTH', 'CDLKICKING']
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
    rsi = tb.RSI(data["Close"], timeperiod=14).to_frame().reset_index().set_axis(axes, axis=1)
    wma = tb.WMA(data["Close"], timeperiod=30).to_frame().reset_index().set_axis(axes, axis=1)
    ema = tb.EMA(data["Close"], timeperiod=30).to_frame().reset_index().set_axis(axes, axis=1)
    sma = tb.SMA(data["Close"], timeperiod=30).to_frame().reset_index().set_axis(axes, axis=1)
    roc = tb.ROC(data["Close"], timeperiod=10).to_frame().reset_index().set_axis(axes, axis=1)
    cmo = tb.CMO(data["Close"], timeperiod=14).to_frame().reset_index().set_axis(axes, axis=1)
    cci = tb.CCI(data["High"], data["Low"], data["Close"], timeperiod=14).to_frame().reset_index().set_axis(axes, axis=1)
    ppo = tb.PPO(data["Close"], fastperiod=12, slowperiod=26, matype=0).to_frame().reset_index().set_axis(axes, axis=1)
    tema  = tb.TEMA(data["Close"], timeperiod=30).to_frame().reset_index().set_axis(axes, axis=1)
    willr = tb.WILLR(data["High"], data["Low"], data["Close"], timeperiod=14).to_frame().reset_index().set_axis(axes, axis=1)
    macd, macdsignal, macdhist = tb.MACD(data["Close"], fastperiod=12, slowperiod=26, signalperiod=9)
    macd = macd.to_frame().reset_index().set_axis(axes, axis=1)

    sar = tb.SAR(data["High"], data["Low"], acceleration=0, maximum=0).to_frame().reset_index().set_axis(axes, axis=1)
    adx = tb.ADX(data["High"], data["Low"], data["Close"], timeperiod=14).to_frame().reset_index().set_axis(axes, axis=1)
    std = tb.STDDEV(data['Close'], timeperiod=5, nbdev=1).to_frame().reset_index().set_axis(axes, axis=1)
    obv = tb.OBV(data['Close'], data['Volume']).to_frame().reset_index().set_axis(axes, axis=1)


    adxr = tb.ADXR(data["High"], data["Low"], data["Close"], timeperiod=14).to_frame().reset_index().set_axis(axes, axis=1)
    apo = tb.APO(data['Close'], fastperiod=12, slowperiod=26, matype=0).to_frame().reset_index().set_axis(axes, axis=1)
    aroondown, aroonup = tb.AROON(data["High"], data["Low"], timeperiod=14)
    aroondown = aroondown.to_frame().reset_index().set_axis(axes, axis=1)
    aroonup = aroonup.to_frame().reset_index().set_axis(axes, axis=1)
    aroonosc = tb.AROONOSC(data["High"], data["Low"], timeperiod=14).to_frame().reset_index().set_axis(axes, axis=1)
    bop = tb.BOP(data["Open"], data["High"], data["Low"], data["Close"]).to_frame().reset_index().set_axis(axes, axis=1)
    dx = tb.DX(data["High"], data["Low"], data["Close"], timeperiod=14).to_frame().reset_index().set_axis(axes, axis=1)
    macdext, macdextsignal, macdexthist = tb.MACDEXT(data["Close"], fastperiod=12, fastmatype=0, slowperiod=26, slowmatype=0, signalperiod=9, signalmatype=0)
    macdext = macdext.to_frame().reset_index().set_axis(axes, axis=1)
    macdfix, macdfixsignal, macdfixhist = tb.MACDFIX(data["Close"], signalperiod=9)
    macdfix = macdfix.to_frame().reset_index().set_axis(axes, axis=1)
    mfi = tb.MFI(data["High"], data["Low"], data["Close"], data["Volume"], timeperiod=14).to_frame().reset_index().set_axis(axes, axis=1)
    minus_di = tb.MINUS_DI(data["High"], data["Low"], data["Close"], timeperiod=14).to_frame().reset_index().set_axis(axes, axis=1)
    minus_dm = tb.MINUS_DM(data["High"], data["Low"], timeperiod=14).to_frame().reset_index().set_axis(axes, axis=1)
    mom = tb.MOM(data["Close"], timeperiod=10).to_frame().reset_index().set_axis(axes, axis=1)
    plus_di = tb.PLUS_DI(data["High"], data["Low"], data["Close"], timeperiod=14).to_frame().reset_index().set_axis(axes, axis=1)
    plus_dm = tb.PLUS_DM(data["High"], data["Low"], timeperiod=14).to_frame().reset_index().set_axis(axes, axis=1)
    rocp = tb.ROCP(data["Close"], timeperiod=10).to_frame().reset_index().set_axis(axes, axis=1)
    rocr = tb.ROCR(data["Close"], timeperiod=10).to_frame().reset_index().set_axis(axes, axis=1)
    rocr100 = tb.ROCR100(data["Close"], timeperiod=10).to_frame().reset_index().set_axis(axes, axis=1)
    slowk, slowd = tb.STOCH(data["High"], data["Low"], data["Close"], fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
    slowk = slowk.to_frame().reset_index().set_axis(axes, axis=1)
    slowd = slowd.to_frame().reset_index().set_axis(axes, axis=1)
    fastk, fastd = tb.STOCHF(data["High"], data["Low"], data["Close"], fastk_period=5, fastd_period=3, fastd_matype=0)
    fastk = fastk.to_frame().reset_index().set_axis(axes, axis=1)
    fastd = fastd.to_frame().reset_index().set_axis(axes, axis=1)
    fastkrsi, fastdrsi = tb.STOCHRSI(data["Close"], timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)
    fastkrsi = fastkrsi.to_frame().reset_index().set_axis(axes, axis=1)
    fastdrsi = fastdrsi.to_frame().reset_index().set_axis(axes, axis=1)
    trix = tb.TRIX(data["Close"], timeperiod=30).to_frame().reset_index().set_axis(axes, axis=1)
    ultosc = tb.ULTOSC(data["High"], data["Low"], data["Close"], timeperiod1=7, timeperiod2=14, timeperiod3=28).to_frame().reset_index().set_axis(axes, axis=1)

    cdlxsidegap3methods = tb.CDLXSIDEGAP3METHODS(data["Open"], data["High"], data["Low"], data["Close"]).to_frame().reset_index().set_axis(axes, axis=1)
    cdlupsidegap2crows = tb.CDLUPSIDEGAP2CROWS(data["Open"], data["High"], data["Low"], data["Close"]).to_frame().reset_index().set_axis(axes, axis=1)
    cdlunique3river = tb.CDLUNIQUE3RIVER(data["Open"], data["High"], data["Low"], data["Close"]).to_frame().reset_index().set_axis(axes, axis=1)
    cdltristar = tb.CDLTRISTAR(data["Open"], data["High"], data["Low"], data["Close"]).to_frame().reset_index().set_axis(axes, axis=1)
    cdlthrusting = tb.CDLTHRUSTING(data["Open"], data["High"], data["Low"], data["Close"]).to_frame().reset_index().set_axis(axes, axis=1)
    cdltasukigap = tb.CDLTASUKIGAP(data["Open"], data["High"], data["Low"], data["Close"]).to_frame().reset_index().set_axis(axes, axis=1)
    cdltakuri = tb.CDLTAKURI(data["Open"], data["High"], data["Low"], data["Close"]).to_frame().reset_index().set_axis(axes, axis=1)
    cdlstciksandwich = tb.CDLSTICKSANDWICH(data["Open"], data["High"], data["Low"], data["Close"]).to_frame().reset_index().set_axis(axes, axis=1)
    cdlstalledpattern = tb.CDLSTALLEDPATTERN(data["Open"], data["High"], data["Low"], data["Close"]).to_frame().reset_index().set_axis(axes, axis=1)
    cdlspinningtop = tb.CDLSPINNINGTOP(data["Open"], data["High"], data["Low"], data["Close"]).to_frame().reset_index().set_axis(axes, axis=1)
    cdlshortline = tb.CDLSHORTLINE(data["Open"], data["High"], data["Low"], data["Close"]).to_frame().reset_index().set_axis(axes, axis=1)
    cdlshootingstar = tb.CDLSHOOTINGSTAR(data["Open"], data["High"], data["Low"], data["Close"]).to_frame().reset_index().set_axis(axes, axis=1)
    cdlseparatinglines = tb.CDLSEPARATINGLINES(data["Open"], data["High"], data["Low"], data["Close"]).to_frame().reset_index().set_axis(axes, axis=1)
    cdlrisefall3methods = tb.CDLRISEFALL3METHODS(data["Open"], data["High"], data["Low"], data["Close"]).to_frame().reset_index().set_axis(axes, axis=1)
    cdlrickshawman = tb.CDLRICKSHAWMAN(data["Open"], data["High"], data["Low"], data["Close"]).to_frame().reset_index().set_axis(axes, axis=1)
    cdlpiercing = tb.CDLPIERCING(data["Open"], data["High"], data["Low"], data["Close"]).to_frame().reset_index().set_axis(axes, axis=1)
    cdlonneck = tb.CDLONNECK(data["Open"], data["High"], data["Low"], data["Close"]).to_frame().reset_index().set_axis(axes, axis=1)
    cdlmorningstar = tb.CDLMORNINGSTAR(data["Open"], data["High"], data["Low"], data["Close"], penetration=0).to_frame().reset_index().set_axis(axes, axis=1)
    cdlmorningdojistar = tb.CDLMORNINGDOJISTAR(data["Open"], data["High"], data["Low"], data["Close"], penetration=0).to_frame().reset_index().set_axis(axes, axis=1)
    cdlmathold = tb.CDLMATHOLD(data["Open"], data["High"], data["Low"], data["Close"], penetration=0).to_frame().reset_index().set_axis(axes, axis=1)
    cdlmatchinglow = tb.CDLMATCHINGLOW(data["Open"], data["High"], data["Low"], data["Close"]).to_frame().reset_index().set_axis(axes, axis=1)
    cdlmarubozu = tb.CDLMARUBOZU(data["Open"], data["High"], data["Low"], data["Close"]).to_frame().reset_index().set_axis(axes, axis=1)
    cdllongline = tb.CDLLONGLINE(data["Open"], data["High"], data["Low"], data["Close"]).to_frame().reset_index().set_axis(axes, axis=1)
    cdllongleggeddoji = tb.CDLLONGLEGGEDDOJI(data["Open"], data["High"], data["Low"], data["Close"]).to_frame().reset_index().set_axis(axes, axis=1)
    cdlladderbottom = tb.CDLLADDERBOTTOM(data["Open"], data["High"], data["Low"], data["Close"]).to_frame().reset_index().set_axis(axes, axis=1)
    cdlkickingbylength = tb.CDLKICKINGBYLENGTH(data["Open"], data["High"], data["Low"], data["Close"]).to_frame().reset_index().set_axis(axes, axis=1)
    cdlkicking = tb.CDLKICKING(data["Open"], data["High"], data["Low"], data["Close"]).to_frame().reset_index().set_axis(axes, axis=1)

    '''
    PREPROCESSING INDICATOR DATA
    '''
    # List of (indicators) DataFrames, size=n_indicators
    indicators = [rsi, wma, ema, sma, roc, cmo, cci, ppo, tema, willr, macd, sar, adx, std, obv, adxr, apo, aroondown, aroonup, aroonosc, bop, dx, macdext, macdfix, mfi, 
    minus_di, minus_dm, mom, plus_di, plus_dm, rocp, rocr, rocr100, slowk, slowd, fastk, fastd, fastkrsi, fastdrsi, trix, ultosc, cdlxsidegap3methods, cdlupsidegap2crows, 
    cdlunique3river, cdltristar, cdlthrusting, cdltasukigap, cdltakuri, cdlstciksandwich, cdlstalledpattern, cdlspinningtop, cdlshortline, cdlshootingstar, cdlseparatinglines, 
    cdlrisefall3methods, cdlrickshawman, cdlpiercing, cdlonneck, cdlmorningstar, cdlmorningdojistar, cdlmathold, cdlmatchinglow, cdlmarubozu, cdllongline, cdllongleggeddoji, 
    cdlladderbottom, cdlkickingbylength, cdlkicking]
    # [rsi, cmo, willr, cci, macd, roc, ppo, std, tema, obv, wma, ema, sma, adx, sar]

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