import yfinance as yf
import pandas as pd
import numpy as np
import talib as tb

'''
DEFINING SOME VARIABLES
'''
startDate = '2001-10-11'
endDate = '2022-04-15'
axes = ['Date', 'Value']
headers = ['RSI', 'WMA', 'EMA', 'SMA', 'ROC', 'CMO', 'CCI', 'PPO', 'TEMA', 'WILLR', 'MACD', 'SAR', 'ADX', 'STDDEV', 'OBV', 'ADXR', 'APO', 'AROONDOWN', 'AROONUP', 'AROONOSC',
           'BOP', 'DX', 'MACDEXT', 'MACDFİX', 'MFI', 'MINUS_DI', 'MINUS_DM', 'MOM', 'PLUS_DI', 'PLUS_DM', 'ROCP', 'ROCR', 'ROCR100', 'SLOWK', 'SLOWD', 'FASTK', 'FASTD',
           'FASTKRSI', 'FASTDRSI', 'TRIX', 'ULTOSC',  'BBANDSU', 'BBANDSM', 'BBANDSL', 'DEMA',  'HT_TRENDLINE', 'KAMA', 'MA', 'MIDPOINT', 'MIDPRICE', 'SAREXT', 'TRIMA', 'AD',
           'ADOSC', 'TRANGE', 'AVGPRICE', 'MEDPRICE', 'TYPPRICE', 'WCLPRICE',  'BETA', 'CORREL', 'LINEARREG', 'LINEARREG_ANGLE', 'LINEARREG_INTERCEPT',
           'LINEARREG_SLOPE', 'TSF', 'VAR']

'''
'CDLXSIDEGAP3METHODS', 'CDLUPSIDEGAP2CROWS', 'CDLUNIQUE3RIVER', 'CDLTRISTAR', 'CDLTHRUSTING', 'CDLTASUKIGAP', 'CDLTAKURI', 
           'CDLSTICKSANDWICH', 'CDLSTALLEDPATTERN', 'CDLSPINNINGTOP', 'CDLSHORTLINE', 'CDLSHOOTINGSTAR', 'CDLSEPARATINGLINES', 'CDLRISEFALL3METHODS', 'CDLRICKSHAWMAN', 
           'CDLPIERCING', 'CDLONNECK', 'CDLMORNINGSTAR','CDLMORNINGDOJISTAR', 'CDLMATHOLD', 'CDLMATCHINGLOW', 'CDLMARUBOZU', 'CDLLONGLINE', 'CDLLONGLEGGEDDOJI', 'CDLLADDERBOTTOM',
           'CDLKICKINGBYLENGTH', 'CDLKICKING', 'CDL2CROWS', 'CDL3BLACKCROWS', 'CDL3LINESTRIKE', 'CDL3OUTSIDE', 'CDL3STARSINSOUTH', 'CDL3WHITESOLDIERS', 
           'CDLABANDONEDBABY', 'CDLADVANCEBLOCK', 'CDLBELTHOLD', 'CDLBREAKAWAY', 'CDLCLOSINGMARUBOZU', 'CDLCONCEALBABYSWALL', 'CDLCOUNTERATTACK', 'CDLDARKCLOUDCOVER', 'CDLDOJI', 'CDLDOJISTAR', 
           'CDLDRAGONFLYDOJI', 'CDLENGULFING', 'CDLEVENINGDOJISTAR', 'CDLEVENINGSTAR', 'CDLGAPSIDESIDEWHITE', 'CDLGRAVESTONEDOJI', 'CDLHAMMER', 'CDLHANGINGMAN', 'CDLHARAMI', 'CDLHARAMICROSS', 
           'CDLHIGHWAVE', 'CDLHIKKAKE', 'CDLHIKKAKEMOD', 'CDLHOMINGPIGEON', 'CDLIDENTICAL3CROWS', 'CDLINNECK', 'CDLINVERTEDHAMMER',
'''

etfList = ['XLF', 'XLU', 'QQQ', 'SPY', 'XLP', 'EWZ', 'EWH', 'XLY', 'XLE']
#
threshold = 0.01  # Re-arrange the Threshold Value
imageList = []
labelList = []
pd.set_option('display.max_rows', None)

'''
DOWNLOADING THE DATA
'''
# DataFrame, size=(n_days, 6), col_names=["Open", "High", "Low", "Close", "Adj Close", "Volume"]
for etf in etfList:
    data = yf.download(etf, start=startDate, end=endDate)

    '''
    CALCULATING THE INDICATOR VALUES
    '''
    # DataFrame, size=(n_days, 2), col_names=["Date", "Value"]
    rsi = tb.RSI(data["Close"], timeperiod=14).to_frame(
    ).reset_index().set_axis(axes, axis=1)
    wma = tb.WMA(data["Close"], timeperiod=30).to_frame(
    ).reset_index().set_axis(axes, axis=1)
    ema = tb.EMA(data["Close"], timeperiod=30).to_frame(
    ).reset_index().set_axis(axes, axis=1)
    sma = tb.SMA(data["Close"], timeperiod=30).to_frame(
    ).reset_index().set_axis(axes, axis=1)
    roc = tb.ROC(data["Close"], timeperiod=10).to_frame(
    ).reset_index().set_axis(axes, axis=1)
    cmo = tb.CMO(data["Close"], timeperiod=14).to_frame(
    ).reset_index().set_axis(axes, axis=1)
    cci = tb.CCI(data["High"], data["Low"], data["Close"],
                 timeperiod=14).to_frame().reset_index().set_axis(axes, axis=1)
    ppo = tb.PPO(data["Close"], fastperiod=12, slowperiod=26,
                 matype=0).to_frame().reset_index().set_axis(axes, axis=1)
    tema = tb.TEMA(data["Close"], timeperiod=30).to_frame(
    ).reset_index().set_axis(axes, axis=1)
    willr = tb.WILLR(data["High"], data["Low"], data["Close"],
                     timeperiod=14).to_frame().reset_index().set_axis(axes, axis=1)
    macd, macdsignal, macdhist = tb.MACD(
        data["Close"], fastperiod=12, slowperiod=26, signalperiod=9)
    macd = macd.to_frame().reset_index().set_axis(axes, axis=1)

    sar = tb.SAR(data["High"], data["Low"], acceleration=0,
                 maximum=0).to_frame().reset_index().set_axis(axes, axis=1)
    adx = tb.ADX(data["High"], data["Low"], data["Close"],
                 timeperiod=14).to_frame().reset_index().set_axis(axes, axis=1)
    std = tb.STDDEV(data['Close'], timeperiod=5, nbdev=1).to_frame(
    ).reset_index().set_axis(axes, axis=1)
    obv = tb.OBV(data['Close'], data['Volume']).to_frame(
    ).reset_index().set_axis(axes, axis=1)

    adxr = tb.ADXR(data["High"], data["Low"], data["Close"],
                   timeperiod=14).to_frame().reset_index().set_axis(axes, axis=1)
    apo = tb.APO(data['Close'], fastperiod=12, slowperiod=26,
                 matype=0).to_frame().reset_index().set_axis(axes, axis=1)
    aroondown, aroonup = tb.AROON(data["High"], data["Low"], timeperiod=14)
    aroondown = aroondown.to_frame().reset_index().set_axis(axes, axis=1)
    aroonup = aroonup.to_frame().reset_index().set_axis(axes, axis=1)
    aroonosc = tb.AROONOSC(data["High"], data["Low"], timeperiod=14).to_frame(
    ).reset_index().set_axis(axes, axis=1)
    bop = tb.BOP(data["Open"], data["High"], data["Low"], data["Close"]).to_frame(
    ).reset_index().set_axis(axes, axis=1)
    dx = tb.DX(data["High"], data["Low"], data["Close"],
               timeperiod=14).to_frame().reset_index().set_axis(axes, axis=1)
    macdext, macdextsignal, macdexthist = tb.MACDEXT(
        data["Close"], fastperiod=12, fastmatype=0, slowperiod=26, slowmatype=0, signalperiod=9, signalmatype=0)
    macdext = macdext.to_frame().reset_index().set_axis(axes, axis=1)
    macdfix, macdfixsignal, macdfixhist = tb.MACDFIX(
        data["Close"], signalperiod=9)
    macdfix = macdfix.to_frame().reset_index().set_axis(axes, axis=1)
    mfi = tb.MFI(data["High"], data["Low"], data["Close"], data["Volume"],
                 timeperiod=14).to_frame().reset_index().set_axis(axes, axis=1)
    minus_di = tb.MINUS_DI(data["High"], data["Low"], data["Close"],
                           timeperiod=14).to_frame().reset_index().set_axis(axes, axis=1)
    minus_dm = tb.MINUS_DM(data["High"], data["Low"], timeperiod=14).to_frame(
    ).reset_index().set_axis(axes, axis=1)
    mom = tb.MOM(data["Close"], timeperiod=10).to_frame(
    ).reset_index().set_axis(axes, axis=1)
    plus_di = tb.PLUS_DI(data["High"], data["Low"], data["Close"],
                         timeperiod=14).to_frame().reset_index().set_axis(axes, axis=1)
    plus_dm = tb.PLUS_DM(data["High"], data["Low"], timeperiod=14).to_frame(
    ).reset_index().set_axis(axes, axis=1)
    rocp = tb.ROCP(data["Close"], timeperiod=10).to_frame(
    ).reset_index().set_axis(axes, axis=1)
    rocr = tb.ROCR(data["Close"], timeperiod=10).to_frame(
    ).reset_index().set_axis(axes, axis=1)
    rocr100 = tb.ROCR100(data["Close"], timeperiod=10).to_frame(
    ).reset_index().set_axis(axes, axis=1)
    slowk, slowd = tb.STOCH(data["High"], data["Low"], data["Close"], fastk_period=5,
                            slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
    slowk = slowk.to_frame().reset_index().set_axis(axes, axis=1)
    slowd = slowd.to_frame().reset_index().set_axis(axes, axis=1)
    fastk, fastd = tb.STOCHF(
        data["High"], data["Low"], data["Close"], fastk_period=5, fastd_period=3, fastd_matype=0)
    fastk = fastk.to_frame().reset_index().set_axis(axes, axis=1)
    fastd = fastd.to_frame().reset_index().set_axis(axes, axis=1)
    fastkrsi, fastdrsi = tb.STOCHRSI(
        data["Close"], timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)
    fastkrsi = fastkrsi.to_frame().reset_index().set_axis(axes, axis=1)
    fastdrsi = fastdrsi.to_frame().reset_index().set_axis(axes, axis=1)
    trix = tb.TRIX(data["Close"], timeperiod=30).to_frame(
    ).reset_index().set_axis(axes, axis=1)
    ultosc = tb.ULTOSC(data["High"], data["Low"], data["Close"], timeperiod1=7,
                       timeperiod2=14, timeperiod3=28).to_frame().reset_index().set_axis(axes, axis=1)

    '''
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
    bbands_upperband, bbands_middleband, bbands_lowerband = tb.BBANDS(
        data['Close'], timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)
    bbands_upperband = bbands_upperband.to_frame().reset_index().set_axis(axes, axis=1)
    bbands_middleband = bbands_middleband.to_frame().reset_index().set_axis(axes, axis=1)
    bbands_lowerband = bbands_lowerband.to_frame().reset_index().set_axis(axes, axis=1)
    dema = tb.DEMA(data['Close'], timeperiod=30).to_frame(
    ).reset_index().set_axis(axes, axis=1)
    ht_trendline = tb.HT_TRENDLINE(
        data['Close']).to_frame().reset_index().set_axis(axes, axis=1)
    kama = tb.KAMA(data['Close'], timeperiod=30).to_frame(
    ).reset_index().set_axis(axes, axis=1)
    ma = tb.MA(data['Close'], timeperiod=30, matype=0).to_frame(
    ).reset_index().set_axis(axes, axis=1)
    midpoint = tb.MIDPOINT(data['Close'], timeperiod=14).to_frame(
    ).reset_index().set_axis(axes, axis=1)
    midprice = tb.MIDPRICE(data["High"], data["Low"], timeperiod=14).to_frame(
    ).reset_index().set_axis(axes, axis=1)
    sarext = tb.SAREXT(data["High"], data["Low"], startvalue=0, offsetonreverse=0, accelerationinitlong=0, accelerationlong=0, accelerationmaxlong=0,
                       accelerationinitshort=0, accelerationshort=0, accelerationmaxshort=0).to_frame().reset_index().set_axis(axes, axis=1)
    trima = tb.TRIMA(data['Close'], timeperiod=30).to_frame(
    ).reset_index().set_axis(axes, axis=1)

    ad = tb.AD(data["High"], data["Low"], data['Close'],
               data['Volume']).to_frame().reset_index().set_axis(axes, axis=1)
    adosc = tb.ADOSC(data["High"], data["Low"], data['Close'], data['Volume'],
                     fastperiod=3, slowperiod=10).to_frame().reset_index().set_axis(axes, axis=1)

    trange = tb.TRANGE(data["High"], data["Low"], data['Close']).to_frame(
    ).reset_index().set_axis(axes, axis=1)

    avgprice = tb.AVGPRICE(data['Open'], data["High"], data["Low"],
                           data['Close']).to_frame().reset_index().set_axis(axes, axis=1)
    medprice = tb.MEDPRICE(data["High"], data["Low"]).to_frame(
    ).reset_index().set_axis(axes, axis=1)
    typprice = tb.TYPPRICE(data["High"], data["Low"], data['Close']).to_frame(
    ).reset_index().set_axis(axes, axis=1)
    wclprice = tb.WCLPRICE(data["High"], data["Low"], data['Close']).to_frame(
    ).reset_index().set_axis(axes, axis=1)

    '''
    cdl2crows = tb.CDL2CROWS(data['Open'], data["High"], data["Low"], data['Close']).to_frame().reset_index().set_axis(axes, axis=1)
    cdl3blackcrows = tb.CDL3BLACKCROWS(data['Open'], data["High"], data["Low"], data['Close']).to_frame().reset_index().set_axis(axes, axis=1)
    cdl3inside = tb.CDL3INSIDE(data['Open'], data["High"], data["Low"], data['Close']).to_frame().reset_index().set_axis(axes, axis=1)
    cdl3linestrike = tb.CDL3LINESTRIKE(data['Open'], data["High"], data["Low"], data['Close']).to_frame().reset_index().set_axis(axes, axis=1)
    cdl3outside = tb.CDL3OUTSIDE(data['Open'], data["High"], data["Low"], data['Close']).to_frame().reset_index().set_axis(axes, axis=1)
    cdl3starsinsouth = tb.CDL3STARSINSOUTH(data['Open'], data["High"], data["Low"], data['Close']).to_frame().reset_index().set_axis(axes, axis=1)
    cdl3whitesoldiers = tb.CDL3WHITESOLDIERS(data['Open'], data["High"], data["Low"], data['Close']).to_frame().reset_index().set_axis(axes, axis=1)
    cdlabandonedbaby = tb.CDLABANDONEDBABY(data['Open'], data["High"], data["Low"], data['Close'], penetration=0).to_frame().reset_index().set_axis(axes, axis=1)
    cdladvancedblock = tb.CDLADVANCEBLOCK(data['Open'], data["High"], data["Low"], data['Close']).to_frame().reset_index().set_axis(axes, axis=1)
    cdlbelthold = tb.CDLBELTHOLD(data['Open'], data["High"], data["Low"], data['Close']).to_frame().reset_index().set_axis(axes, axis=1)
    cdlbreakaway = tb.CDLBREAKAWAY(data['Open'], data["High"], data["Low"], data['Close']).to_frame().reset_index().set_axis(axes, axis=1)
    cdlclosingmarubozu = tb.CDLCLOSINGMARUBOZU(data['Open'], data["High"], data["Low"], data['Close']).to_frame().reset_index().set_axis(axes, axis=1)
    cdlconcealbabyswall = tb.CDLCONCEALBABYSWALL(data['Open'], data["High"], data["Low"], data['Close']).to_frame().reset_index().set_axis(axes, axis=1)
    cdlcounterattack = tb.CDLCOUNTERATTACK(data['Open'], data["High"], data["Low"], data['Close']).to_frame().reset_index().set_axis(axes, axis=1)
    cdldarkcloudcover = tb.CDLDARKCLOUDCOVER(data['Open'], data["High"], data["Low"], data['Close'], penetration=0).to_frame().reset_index().set_axis(axes, axis=1)
    cdldoji = tb.CDLDOJI(data['Open'], data["High"], data["Low"], data['Close']).to_frame().reset_index().set_axis(axes, axis=1)
    cdldojistar = tb.CDLDOJISTAR(data['Open'], data["High"], data["Low"], data['Close']).to_frame().reset_index().set_axis(axes, axis=1)
    cdldragonflydoji = tb.CDLDRAGONFLYDOJI(data['Open'], data["High"], data["Low"], data['Close']).to_frame().reset_index().set_axis(axes, axis=1)
    cdlengulfing = tb.CDLENGULFING(data['Open'], data["High"], data["Low"], data['Close']).to_frame().reset_index().set_axis(axes, axis=1)
    cdleveningdojistar = tb.CDLEVENINGDOJISTAR(data['Open'], data["High"], data["Low"], data['Close'], penetration=0).to_frame().reset_index().set_axis(axes, axis=1)
    cdleveningstar = tb.CDLEVENINGSTAR(data['Open'], data["High"], data["Low"], data['Close'], penetration=0).to_frame().reset_index().set_axis(axes, axis=1)
    cdlgapsidesidewhite = tb.CDLGAPSIDESIDEWHITE(data['Open'], data["High"], data["Low"], data['Close']).to_frame().reset_index().set_axis(axes, axis=1)
    cdlgravestonedoji = tb.CDLGRAVESTONEDOJI(data['Open'], data["High"], data["Low"], data['Close']).to_frame().reset_index().set_axis(axes, axis=1)
    cdlhammer = tb.CDLHAMMER(data['Open'], data["High"], data["Low"], data['Close']).to_frame().reset_index().set_axis(axes, axis=1)
    cdlhangingman = tb.CDLHANGINGMAN(data['Open'], data["High"], data["Low"], data['Close']).to_frame().reset_index().set_axis(axes, axis=1)
    cdlharami = tb.CDLHARAMI(data['Open'], data["High"], data["Low"], data['Close']).to_frame().reset_index().set_axis(axes, axis=1)
    cdlharamicross = tb.CDLHARAMICROSS(data['Open'], data["High"], data["Low"], data['Close']).to_frame().reset_index().set_axis(axes, axis=1)
    cdlhighwave = tb.CDLHIGHWAVE(data['Open'], data["High"], data["Low"], data['Close']).to_frame().reset_index().set_axis(axes, axis=1)
    cdlhikkake = tb.CDLHIKKAKE(data['Open'], data["High"], data["Low"], data['Close']).to_frame().reset_index().set_axis(axes, axis=1)
    cdlhikkakemod = tb.CDLHIKKAKEMOD(data['Open'], data["High"], data["Low"], data['Close']).to_frame().reset_index().set_axis(axes, axis=1)
    cdlhomingpigeon = tb.CDLHOMINGPIGEON(data['Open'], data["High"], data["Low"], data['Close']).to_frame().reset_index().set_axis(axes, axis=1)
    cdlidentical3crows = tb.CDLIDENTICAL3CROWS(data['Open'], data["High"], data["Low"], data['Close']).to_frame().reset_index().set_axis(axes, axis=1)
    cdlinneck = tb.CDLINNECK(data['Open'], data["High"], data["Low"], data['Close']).to_frame().reset_index().set_axis(axes, axis=1)
    cdlinvertedhammer = tb.CDLINVERTEDHAMMER(data['Open'], data["High"], data["Low"], data['Close']).to_frame().reset_index().set_axis(axes, axis=1)
    '''

    beta = tb.BETA(data["High"], data["Low"], timeperiod=5).to_frame(
    ).reset_index().set_axis(axes, axis=1)
    correl = tb.CORREL(data["High"], data["Low"], timeperiod=30).to_frame(
    ).reset_index().set_axis(axes, axis=1)
    linearreg = tb.LINEARREG(data['Close'], timeperiod=14).to_frame(
    ).reset_index().set_axis(axes, axis=1)
    linearreg_angle = tb.LINEARREG_ANGLE(
        data['Close'], timeperiod=14).to_frame().reset_index().set_axis(axes, axis=1)
    linearreg_intercept = tb.LINEARREG_INTERCEPT(
        data['Close'], timeperiod=14).to_frame().reset_index().set_axis(axes, axis=1)
    linearreg_slope = tb.LINEARREG_SLOPE(
        data['Close'], timeperiod=14).to_frame().reset_index().set_axis(axes, axis=1)
    tsf = tb.TSF(data['Close'], timeperiod=14).to_frame(
    ).reset_index().set_axis(axes, axis=1)
    var = tb.VAR(data['Close'], timeperiod=5, nbdev=1).to_frame(
    ).reset_index().set_axis(axes, axis=1)

    '''
    PREPROCESSING INDICATOR DATA
    '''
    # List of (indicators) DataFrames, size=n_indicators
    indicators = [rsi, cmo, plus_di, minus_di, willr, cci, ultosc, aroonosc, mfi, mom, macd, macdfix, linearreg_angle, linearreg_slope, rocp, roc, rocr, rocr100, slowk, fastd, slowd, aroonup,
                  aroondown, apo, macdext, fastk, ppo, minus_dm, adosc, fastdrsi, fastkrsi, trange, trix, std, bop, var, plus_dm, correl, ad, beta, wclprice, tsf, typprice, avgprice, medprice, bbands_lowerband,
                  linearreg, obv, bbands_middleband, tema, bbands_upperband, dema, midprice, midpoint, sarext, wma, ema, ht_trendline, kama, sma, ma, sar, adxr, adx, trima, linearreg_intercept, dx]
    # [rsi, cmo, willr, cci, macd, roc, ppo, std, tema, obv, wma, ema, sma, adx, sar]
    '''
    cdlxsidegap3methods, cdlupsidegap2crows, 
    cdlunique3river, cdltristar, cdlthrusting, cdltasukigap, cdltakuri, cdlstciksandwich, cdlstalledpattern, cdlspinningtop, cdlshortline, cdlshootingstar, cdlseparatinglines, 
    cdlrisefall3methods, cdlrickshawman, cdlpiercing, cdlonneck, cdlmorningstar, cdlmorningdojistar, cdlmathold, cdlmatchinglow, cdlmarubozu, cdllongline, cdllongleggeddoji, 
    cdlladderbottom, cdlkickingbylength, cdlkicking, cdl2crows, cdl3inside, cdl3linestrike, cdl3outside, cdl3starsinsouth, cdl3whitesoldiers, cdlabandonedbaby, cdladvancedblock,
    cdlbelthold, cdlbreakaway, cdlclosingmarubozu, cdlconcealbabyswall, cdlcounterattack, cdldarkcloudcover, cdldoji, cdldojistar, cdldragonflydoji, cdlengulfing, cdleveningdojistar, cdleveningstar,
    cdlgapsidesidewhite, cdlgravestonedoji, cdlhammer, cdlhangingman, cdlharami, cdlharamicross, cdlhighwave, cdlhikkake, cdlhikkakemod, cdlhomingpigeon, cdlidentical3crows, cdlinneck,
    cdlinvertedhammer,
    '''

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
        # Getting rid of null values
        indicatorValues.append(indicator['Value'].iloc[maxNullVal:])

    # DataFrame, size=(n_days, n_indicators, col_names=headers)
    indicatorValuesMatrix = pd.concat(indicatorValues, axis=1, keys=headers)
    indicatorCorr = indicatorValuesMatrix.corr(method='pearson')

    '''
    dictCor = {}
    for header, value in zip(headers, indicatorCorr.iloc[0]):
        dictCor[header] = value
    sortedDictCor = {k: v for k, v in sorted(dictCor.items(), key=lambda item: abs(item[1]), reverse=True)}
    for k,v in sortedDictCor.items():
        print(k, v)

    '''

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
                labelList.append(np.array([1.0]))  # HOLD
            # enough to pass the threshold
            else:
                labelList.append(np.array([0.0]))  # BUY
        # If the price has decreased
        elif(closePriceDifference < 0):
            # but not so much to pass the thresshold
            if(abs(closePriceDifference) <= thresholdPrice):
                labelList.append(np.array([1.0]))  # HOLD
            # so much to pass the threshold
            else:
                labelList.append(np.array([2.0]))  # SELL
        # If the price hasn't changed
        else:
            labelList.append(np.array([1.0]))  # HOLD

    np.save(f'./ETF/New/{etf}.npy', data_close[:-1])
    print(len(imageList))
    print(len(labelList))
    print(len(data_close[:-1]))

imageList = np.array(imageList)
labelList = np.array(labelList)

unique, counts = np.unique(labelList, return_counts=True)
print(np.asarray((unique, counts)).T)

imageList_copy = imageList[:]
imageList_copy = imageList_copy.reshape(len(imageList), -1)
mean = np.mean(imageList_copy, axis=0)
std = np.std(imageList_copy, axis=0)
imageList_copy = (imageList_copy - mean) / std
imageList = imageList_copy.reshape(
    len(imageList), len(indicators), len(indicators), 1)

np.save("./ETF/New/Images.npy", imageList)
np.save("./ETF/New/Labels001.npy", labelList)
