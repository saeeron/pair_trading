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
    rsi_level_u = kwargs.get("rsi_level_u", 0.65)
    rsi_level_l = kwargs.get("rsi_level_l", 0.35)

    rsi1 = calculate_rsi(ohlc1, period=120)
    rsi2 = calculate_rsi(ohlc2, period=120)

    rsi = pd.concat([rsi1, rsi2], axis = 1, join="outer")
    rsi.columns = ['rsi1', 'rsi2']
    rsi["diff"] = rsi1 - rsi2
    rsi["z_score"] = rsi['diff'].subtract(rsi['diff'].expanding().mean()) / rsi['diff'].expanding().std()

    def apply_fun(x):
        
        if (x['rsi1'] > rsi_level_u and x['z_score'] > z_score_threshold) or ( x['rsi2'] < rsi_level_l and x['z_score'] > z_score_threshold ):
            return -1
        elif (x['rsi2'] > rsi_level_u and  x['z_score'] < - z_score_threshold) or ( x['rsi1'] < rsi_level_u and x['z_score'] < - z_score_threshold ):
            return 1
        else:
            return 0

    rsi['momentum_diff'] = rsi.apply(apply_fun, axis=1)
    
    rsi['ret1'] = ohlc1['close'].pct_change()
    rsi['ret2'] = ohlc2['close'].pct_change()


    rsi['ret_diff'] = ohlc1['close'].pct_change() - ohlc2['close'].pct_change()
    rsi['port_ret'] = rsi['position'].shift(1) * rsi['ret_diff']

    
    return rsi

def momentum_strategy(ohlc: pd.DataFrame):

    ret = ohlc['close'].pct_change()
    port = np.sqrt(10)*(ret.rolling(10, min_periods=1).mean() - ret.rolling(365, min_periods=10).mean())
    port = (port / ret.rolling(365, min_periods=10).std()).apply(np.sign)

    return port

def reversal_strategy(ohlc: pd.DataFrame, **kwargs):

    level = kwargs.get("level", 0.05)

    ret = ohlc['close'].pct_change()
    port = ret.rolling(window=5).sum()

    def apply_fun(x):
        
        if x > level:
            return -1
        elif x < - level:
            return 1
        else:
            return 0

    port = ret.rolling(window=5).sum().apply(apply_fun)

    return port