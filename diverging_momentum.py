import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def calculate_atr(df: pd.DataFrame, period=120):
    """
    Calculate the Average True Range (ATR) of a given DataFrame.

    :param df: pandas DataFrame with 'High', 'Low', and 'Close' price columns
    :param period: The period over which to calculate the ATR
    :return: pandas Series with the ATR values
    """
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['high'] - df['close'].shift())

    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=period, min_periods=1).mean()

    return atr


def calculate_rsi(df: pd.DataFrame, period=120):
    """
    Calculate the Relative Strength Index (RSI) of a given DataFrame.

    :param df: pandas DataFrame with 'Close' price column
    :param period: The period over which to calculate the RSI
    :return: pandas Series with the RSI values
    """
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi / 100