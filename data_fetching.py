import numpy as np
import pandas as pd
from binance.client import Client as bnb_client
from datetime import datetime


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

def return_ohlc(symbol: str, freq: str, start_ts: str, end_ts: str):
    """
    returns ohlc prices from binance for a symbol

    Parameters
    ----------
    symbol: str
        symbol. e.g., BTCUSDT
    freq : str
        frequency for bar data
    start_ts : str
        starting date string
    
    end_ts: str
        endinf date string

    Returns
    -------
    pd.DataFrame
        DataFrame of crypto prices with columns = ['open', 'high','low','close']

    Raises
    ------
    None
    """
    client = bnb_client(tld='US')
    data = client.get_historical_klines(symbol, freq, start_ts, end_ts)
    columns = ['open_time','open','high','low','close','volume','close_time','quote_volume',
                    'num_trades','taker_base_volume','taker_quote_volume','ignore']
    data = pd.DataFrame(data, columns = columns)
    data['open_time'] = data['open_time'].map(lambda x: datetime.utcfromtimestamp(x/1000))
    data['close_time'] = data['close_time'].map(lambda x: datetime.utcfromtimestamp(x/1000))
    data = data.set_index('open_time')
    px = data[['open','high', 'low', 'close']].astype(float)
    px = px.reindex(pd.date_range(px.index[0],px.index[-1],freq=freq))

    return px
