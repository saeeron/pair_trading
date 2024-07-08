import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from filterpy.kalman import KalmanFilter
from binance.client import Client as bnb_client
from typing import Union, Type
from datetime import datetime, timedelta


class KalmanPairSpread(KalmanFilter):
    def __init__(self, price: pd.DataFrame):
        super().__init__(dim_x=2, dim_z=1)
        
        self.price1 = price.iloc[:, 0].values
        self.price2 = price.iloc[:, 1].values
        
        self.x = np.array([0., 0.])
        self.F = np.eye(2)
        self.H = np.array([[1., self.price1[0]]])

        self.P *= 1000  # uncertainty in initial state
        self.R = 0.1 
        self.Q = np.eye(2)

        self._alphas = []
        self._betas = []
        self._spreads = []
        self._fit()

    def estimate_spread(self):
        return np.array(self._spreads)

    def _fit(self):
        for i in range(len(self.price2)):
            self.predict()

            self.update([self.price2[i]])
            
            self._alphas.append(self.x[0])
            self._betas.append(self.x[1])
            self._spreads.append(self.price2[i] - (self.x[0] + self.x[1] * self.price1[i]))
            
            if i < len(self.price2) - 1:
                self.H = np.array([[1., self.price1[i + 1]]])
        
        return None

    def return_alphas(self):
        return np.array(self._alphas)

    def return_betas(self):
        return np.array(self._betas)

class KalmanTripletSpread(KalmanFilter):
    def __init__(self, price: pd.DataFrame):
        super().__init__(dim_x=3, dim_z=1)
        
        self.price1 = price.iloc[:, 0].values
        self.price2 = price.iloc[:, 1].values
        self.price3 = price.iloc[:, 2].values
        
        self.x = np.array([0., 0., 0.])
        self.F = np.eye(3)
        self.H = np.array([[1., self.price1[0], self.price2[0]]])

        self.P *= 1000  # uncertainty in initial state
        self.R = 0.1
        self.Q = np.eye(3)

        self._alphas = []
        self._betas1 = []
        self._betas2 = []
        self._spreads = []
        self._fit()

    def estimate_spread(self):
        return np.array(self._spreads)

    def _fit(self):
        for i in range(len(self.price2)):
            self.predict()

            self.update([self.price3[i]])
            
            self._alphas.append(self.x[0])
            self._betas1.append(self.x[1])
            self._betas2.append(self.x[2])
            
            self._spreads.append(self.price3[i] - (self.x[0] + self.x[1] * self.price1[i] + self.x[2] * self.price2[i]))
            
            if i < len(self.price2) - 1:
                self.H = np.array([[1., self.price1[i + 1], self.price2[i + 1]]])

        return None
    def return_alphas(self):
        return self._alphas

    def return_betas(self):
        return np.array(self._betas1), np.array(self._betas2)  


class SpreadOLS():
    def __init__(self, price: pd.DataFrame, window: int):

        
        self.n = price.shape[0]
        self.m = price.shape[1] - 1 

        if price.shape[1] <= 1:
            raise ValueError("we need at least two columns of pricess in your data!")
        
        self.price_x = price.iloc[:, :-1].values
        self.price_y = price.iloc[:, -1].values
        self.window = window

        self._alphas = np.full((self.n, 1), np.nan)
        self._betas = np.full((self.n, self.m), np.nan)
        self._spreads = np.full((self.n, 1), np.nan)

        try:
            self._fit()
        except np.linalg.LinAlgError:
            pass

    def estimate_spread(self):
        return self._spreads
    
    def _fit(self):
        
        for i in range(self.window - 1, self.n):
            
            x_window = self.price_x[i - self.window + 1: i+1, :]
            y_window = self.price_y[i - self.window + 1: i+1]
            
            X = np.hstack([x_window, np.ones((self.window, 1))])
            beta, _, _, _ = np.linalg.lstsq(X, y_window, rcond=None)
            self._betas[i, :] = beta[:self.m]
            self._alphas[i] = beta[self.m]
            resids = y_window - X @ beta
            self._spreads[i] = resids[-1]
            
        return None
            
    def return_alphas(self):
        return self._alphas
    def return_betas(self):
        return self._betas

class SpreadCopula:
    pass

class SpreadVineCopula:
    pass


def spread_backtest(all_data: pd.DataFrame, symbol_list: list[str], SpreadEstimator: Union[Type[KalmanPairSpread], Type[KalmanTripletSpread], Type[SpreadOLS]], **kwargs) -> (float, pd.DataFrame):

    """
    performs backtest using a Spread Estimator class

    Parameters
    ----------
    freq : str
        frequency for bar data
    start_ts : str
        start date string
    
    end_ts: str
        end date string

    Returns
    -------
    sr : float
        Sharpe Ratio from backtest
    
    data : pd.DataFrame
        DataFrame including positions and all prices data

    Raises
    ------
        ValueError
        
    """

    if len(symbol_list) not in [2,3]:
        raise ValueError("This backtest only supports pairs and triplets")
    
    data = all_data[symbol_list].copy()

    if (data.notnull().all(1).sum() / len(data)) < 0.8:
        return np.nan, np.nan

    data = data[data.notna().any(axis=1).cummax()]
    
    data = data.ffill()
    if "window" in kwargs.keys():
        window = kwargs["window"]
        se = SpreadEstimator(data, window)
    else:
        se = SpreadEstimator(data)
    
    if "transaction_cost"  in kwargs.keys():
        transaction_cost = kwargs["transaction_cost"]
    else:
        transaction_cost = 0

    data['spread'] = se.estimate_spread()
    data = data[data.notna().any(axis=1).cummax()]

    cumulative_mean_spread = data['spread'].expanding().mean()
    cumulative_std_spread = data['spread'].expanding().std()
    data['spread_zscore'] = (data['spread'] - cumulative_mean_spread) / cumulative_std_spread

    entry_threshold = 2
    exit_threshold = 0

    data['long_entry'] = data['spread_zscore'] < -entry_threshold
    data['long_exit'] = data['spread_zscore'] > -exit_threshold
    data['short_entry'] = data['spread_zscore'] > entry_threshold
    data['short_exit'] = data['spread_zscore'] < exit_threshold
    
    # Signal positions   1 long; -1 short; 0; no position 
    data['positions_long'] = np.nan
    data.loc[data['long_entry'], 'positions_long'] = 1
    data.loc[data['long_exit'], 'positions_long'] = 0
    data['positions_long'] = data['positions_long'].ffill().fillna(0)

    data['positions_short'] = np.nan
    data.loc[data['short_entry'], 'positions_short'] = -1
    data.loc[data['short_exit'], 'positions_short'] = 0
    data['positions_short'] = data['positions_short'].ffill().fillna(0)
    data['positions'] = data['positions_long'] + data['positions_short']

    

    data['return_spread'] = data[symbol_list].pct_change().iloc[:, -1] - data[symbol_list].pct_change().iloc[:, :-1].sum(axis=1)

    data["port"] = data['positions'].shift(1)
    data["tc"] = data["port"].diff().abs().fillna(0) * transaction_cost * 1e-4

    data['strategy_returns'] = data["port"] * data['return_spread'] - data["tc"]
    data['cumulative_returns'] = (1 + data['strategy_returns']).cumprod() - 1
    delta_time = timedelta(days=365) / (data.index[-1] - data.index[-2])
    if data['strategy_returns'].std() != 0: 
        sr = data['strategy_returns'].mean() / data['strategy_returns'].std() * np.sqrt(delta_time)
    else:
        sr = np.nan
    return sr, data

def return_all_cryptos(freq: str, start_ts: str, end_ts: str):
    """
    returns all available cryptos prices from binance

    Parameters
    ----------
    freq : str
        frequency for bar data
    start_ts : str
        starting date string
    
    end_ts: str
        endinf date string

    Returns
    -------
    pd.DataFrame
        DataFrame of crypto prices

    Raises
    ------
    None
        
    """
    client = bnb_client(tld='US')
    all_symbols = client.get_all_tickers()
    univ = [sym['symbol'] for sym in all_symbols if sym['symbol'].endswith("USDT") ]
    px = {}
    for x in univ:
        data = client.get_historical_klines(x, freq, start_ts, end_ts)
        columns = ['open_time','open','high','low','close','volume','close_time','quote_volume',
                    'num_trades','taker_base_volume','taker_quote_volume','ignore']
    
        data = pd.DataFrame(data, columns = columns)
        data['open_time'] = data['open_time'].map(lambda x: datetime.utcfromtimestamp(x/1000))
        data['close_time'] = data['close_time'].map(lambda x: datetime.utcfromtimestamp(x/1000))
        px[x] = data.set_index('open_time')['close']
    px = pd.DataFrame(px).astype(float)
    px = px.reindex(pd.date_range(px.index[0],px.index[-1],freq=freq))

    return px