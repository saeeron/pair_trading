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


def momentum_neutral_backtest(ohlc1: pd.DataFrame, ohlc2: pd.DataFrame, **kwargs) -> (pd.Series, pd.Series, pd.Series, pd.Series, pd.Series):
    """
    Backtest for momentum neutral strategy

    :param ohlc1: pandas DataFrame with all open, high, low, close prices and datetime as the index
    :param ohlc2: pandas DataFrame with all open, high, low, close prices and datetime as the index
    :param tc_cost: float transaction cost in basis point 
    :return mom_div_port: pandas Series portfolio return from momentum-neutral strategy
    :return mom_port1: pandas Series portfolio return from momentum-only strategy on ohlc1
    :return mom_port2: pandas Series portfolio return from momentum-only strategy on ohlc2
    :return rev_port1: pandas Series portfolio return from reversal-only strategy on ohlc1
    :return rev_port2: pandas Series portfolio return from reversal-only strategy on ohlc2
    """
    rsi_period = kwargs.get("rsi_period", 120)
    z_score_threshold = kwargs.get("z_score_threshold", 1.5)
    rsi_level_u = kwargs.get("rsi_level_u", 0.65)
    rsi_level_l = kwargs.get("rsi_level_l", 0.35)
    mom_diff_shift = kwargs.get("mom_diff_shift", 1)
    rev_shift = kwargs.get("rev_shift", 1)
    mom_shift = kwargs.get("mom_shift", 1)
    tc_cost = kwargs.get("tc_cost", 0)

    rsi1 = calculate_rsi(ohlc1, period=120)
    rsi2 = calculate_rsi(ohlc2, period=120)

    rsi = pd.concat([rsi1, rsi2], axis = 1, join="outer")
    rsi.columns = ['rsi1', 'rsi2']
    rsi["diff"] = rsi1 - rsi2
    rsi["z_score"] = rsi['diff'].subtract(rsi['diff'].expanding().mean()) / rsi['diff'].expanding().std()

    rsi1_overbought = np.where((rsi['rsi1'] > rsi_level_u) & (rsi['rsi2'] < rsi_level_u), True, False)
    rsi2_overbought = np.where((rsi['rsi2'] > rsi_level_u) & (rsi['rsi1'] < rsi_level_u), True, False)
    rsi1_oversold = np.where((rsi['rsi1'] < rsi_level_l) & (rsi['rsi2'] > rsi_level_l), True, False)
    rsi2_oversold = np.where((rsi['rsi2'] < rsi_level_l) & (rsi['rsi1'] > rsi_level_l), True, False)

    mom_flag = pd.Series(np.where((rsi["z_score"] > z_score_threshold) & (rsi1_overbought | rsi2_oversold), -1, 
                        (np.where((rsi["z_score"] < -z_score_threshold) & (rsi2_overbought | rsi1_oversold),  1, 0))), index = rsi.index) # -1: ohlc1 is good for reversal, ohlc2 is good for momentum; vice versa for 1; 0: they are not diverging in their momentums
    
    mom_flag = mom_flag.shift(mom_diff_shift)

    mom1, mom2 = momentum_strategy(ohlc1, shift = mom_shift), momentum_strategy(ohlc2, shift = mom_shift)
    rev1, rev2 = reversal_strategy(ohlc1, shift = rev_shift), reversal_strategy(ohlc2, shift = rev_shift)

    signal1 = pd.Series(np.where(mom_flag == -1, rev1, np.where(mom_flag == 1, mom1, 0)), index = rsi.index)
    signal2 = pd.Series(np.where(mom_flag == 1, rev2, np.where(mom_flag == -1, mom2, 0)), index = rsi.index)

    mom_div_port = signal1 * ohlc1['close'].pct_change() + signal2 * ohlc2['close'].pct_change()
    mom_div_port -= mom_div_port.diff().abs().fillna(0) * tc_cost  * 1e-4

    mom_port1 = mom1 * ohlc1['close'].pct_change()
    mom_port1 -= mom_port1.diff().abs().fillna(0) * tc_cost  * 1e-4

    mom_port2 = mom2 * ohlc2['close'].pct_change()
    mom_port2 -= mom_port2.diff().abs().fillna(0) * tc_cost  * 1e-4

    mom_pair = mom_port1 + mom_port2

    rev_port1 = rev1 * ohlc1['close'].pct_change()
    rev_port1 -= rev_port1.diff().abs().fillna(0) * tc_cost  * 1e-4

    rev_port2 = rev2 * ohlc2['close'].pct_change()
    rev_port2 -= rev_port2.diff().abs().fillna(0) * tc_cost  * 1e-4

    rev_pair = rev_port1 + rev_port2

    return mom_div_port, mom_pair ,mom_port1, mom_port2, rev_pair, rev_port1, rev_port2

def momentum_strategy(ohlc: pd.DataFrame, **kwargs):

    shift = kwargs.get("shift", 1)

    ret = ohlc['close'].pct_change()
    signal = np.sqrt(10)*(ret.rolling(10, min_periods=1).mean() - ret.rolling(120, min_periods=10).mean())
    signal = (signal / ret.rolling(120, min_periods=10).std()).apply(np.sign)

    return signal.shift(shift)

def reversal_strategy(ohlc: pd.DataFrame, **kwargs):

    level = kwargs.get("level", 0.05)
    shift = kwargs.get("shift", 1)
    accum_window = kwargs.get("accum_window", 5)

    ret = ohlc['close'].pct_change()
    mov_sum = ret.rolling(window=accum_window).sum()
    
    signal = pd.Series(np.where(mov_sum > level, -1, np.where(mov_sum < -1 * level, 1, 0)), index = ohlc.index)

    return signal.shift(shift)