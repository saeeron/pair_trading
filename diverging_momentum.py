import numpy as np
import pandas as pd
from typing import Union, Type
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


def momentum_neutral_backtest(ohlc1: pd.DataFrame, ohlc2: pd.DataFrame, **kwargs) -> (float, pd.DataFrame):
    """
    Backtest for momentum neutral strategy
    """
    rsi_period = kwargs.get("rsi_period", 120)
    z_score_threshold = kwargs.get("z_score_threshold", 1.5)

    rsi1 = calculate_rsi(ohlc1, period=120)
    rsi2 = calculate_rsi(ohlc2, period=120)
    
    z_score = ((rsi1 - rsi2) - (rsi1 - rsi2).expanding().mean() )/ (rsi1 - rsi2).expanding().std()

    rsi = pd.concat([rsi1, rsi2], axis = 1, join="outer")
    rsi.columns = ['rsi1', 'rsi2']
    rsi["diff"] = rsi1 - rsi2
    rsi["z_score"] = z_score

    def apply_fun(x):
        
        if x > z_score_threshold:
            return -1
        elif x < - z_score_threshold:
            return 1
        else:
            return 0

    rsi['position'] = rsi['z_score'].apply(apply_fun)
    
    return rsi