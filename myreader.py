import yfinance as yf
import pandas as pd
import numpy as np
import talib as tb

def applyTA():
    return

def readData(etf, startDate, endData):
    data = yf.download(etf, start=startDate, end=endDate)
    print(data)
    print(data.columns)
    import sys
    sys.exit(1)


etf = "XLF"
startDate = '2001-10-11'
endDate = '2022-04-15'
readData(etf, startDate, endDate)
