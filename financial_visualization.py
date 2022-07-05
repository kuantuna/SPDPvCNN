from cProfile import label
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ETF_NAME = "SPY"
etf_list = ['XLF', 'XLU', 'QQQ', 'SPY', 'XLP', 'EWZ', 'EWH', 'XLY', 'XLE']
for ETF_NAME in etf_list:
    vit = pd.read_csv("Results/financial/vit-bsr-0038.csv")
    vit_spy = vit[vit["etf"] == ETF_NAME][["date", "value"]]
    vit_spy = vit_spy.reset_index()
    vit_spy["date"] = pd.to_datetime(
        vit_spy["date"], format="%Y-%m-%d %H:%M:%S")

    convmixer = pd.read_csv("Results/financial/convmixer-bsr-0038.csv")
    cm_spy = convmixer[convmixer["etf"] == ETF_NAME][["date", "value"]]
    cm_spy = cm_spy.reset_index()
    cm_spy["date"] = pd.to_datetime(cm_spy["date"], format="%Y-%m-%d %H:%M:%S")

    cnn = pd.read_csv("Results/financial/cnn-bsr-0038.csv")
    cnn_spy = cnn[cnn["etf"] == ETF_NAME][["date", "value"]]
    cnn_spy = cnn_spy.reset_index()
    cnn_spy["date"] = pd.to_datetime(
        cnn_spy["date"], format="%Y-%m-%d %H:%M:%S")

    bh = pd.read_csv("Results/financial/b&h-0038.csv")
    bh_spy = bh[bh["etf"] == ETF_NAME][["date", "value"]]
    bh_spy = bh_spy.reset_index()
    bh_spy["date"] = pd.to_datetime(bh_spy["date"], format="%Y-%m-%d %H:%M:%S")

    # lb = pd.read_csv("Results/financial/label-0038.csv")
    # lb_spy = lb[lb["etf"] == ETF_NAME][["date", "value"]]
    # lb_spy = lb_spy.reset_index()
    # lb_spy["date"] = pd.to_datetime(lb_spy["date"], format="%Y-%m-%d %H:%M:%S")

    formatter = mdates.DateFormatter("%m/%Y")  # formatter of the date
    # locator = mdates.YearLocator()  # where to put the labels
    locator = mdates.MonthLocator(interval=3)

    fig = plt.figure(figsize=(20, 5))
    ax = plt.gca()
    # calling the formatter for the x-axis
    ax.xaxis.set_major_formatter(formatter)
    ax.xaxis.set_major_locator(locator)  # calling the locator for the x-axis
    plt.plot(vit_spy["date"], vit_spy["value"], label="ViT")
    plt.plot(cm_spy["date"], cm_spy["value"], label="ConvMixer")
    plt.plot(cnn_spy["date"], cnn_spy["value"], label="CNN-TA")
    plt.plot(bh_spy["date"], bh_spy["value"], label="B&H")
    # plt.plot(lb_spy["date"], lb_spy["value"], label="Label")

    fig.autofmt_xdate()  # optional if you want to tilt the date labels - just try it
    plt.xlabel("Date")
    plt.title(f"{ETF_NAME}")
    plt.ylabel("Capital ($)")
    plt.legend(loc="upper left")
    plt.grid(True)
    plt.show()
