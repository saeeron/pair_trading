## A Python Module to Study Mispricing (market-neutral) and Momentum Divergence (momentum-neutral) Strategies for pairs (triplets, ...)

  This module implements some pair (triplets) trading ideas regarding mispricing and momentum divergence.  

Clone the module via:

```bash

git  clone  git@github.com:saeeron/pair_trading.git

cd  pair_trading

```


### Running mispricing strategy

  

Get all cryptos data from binance:

```python

from mispricing import KalmanPairSpread, KalmanTripletSpread, SpreadOLS, spread_backtest

from data_fetching import return_all_cryptos

df = return_all_cryptos("4h", "2021-01-01", "2022-01-01")

```

Run a backtest on retrieved data:

```python

sr, data = spread_backtest(df, ["NEOUSDT","QTUMUSDT"], KalmanPairSpread, transaction_cost = 20)

  

```

`sr` is the Sharpe Ratio and `data` includes all time-series details of spreads, signals and returns.

  

### Running Momentum Divergence strategy

  

Get two assets' OHLC

  

```python

from data_fetching import return_ohlc

from diverging_momentum import momentum_neutral_backtest

  

ohlc1 = return_ohlc('ETCUSDT', '4h','2021-01-1','2022-01-01')

ohlc2 = return_ohlc('KNCUSDT', '4h','2021-01-1','2022-01-01')

```

Run momentum-neutralized backtest on them using some example parameters.

  

```python

mom_div_port, mom_pair, mom_port1, mom_port2, rev_pair, rev_port1, rev_port2 = momentum_neutral_backtest(ohlc1, ohlc2,z_score_threshold= 1.5, rsi_period = 120, rev_shift = 2, rsi_level_u = 0.7, rsi_level_l = 0.3, tc_cost = 20)

  

print(f'Sharpe Ratio: {mom_div_port.mean() / mom_div_port.std() * 365 ** 0.5}' )

```

---
for further info read articles:
1. https://medium.com/@saeedroshanbox/is-momentum-neutral-trading-possible-1ffa5b2400cd
2. https://medium.com/@saeedroshanbox/pair-and-triplet-spread-trading-on-cryptos-using-kalman-filter-65cdacbee5a2

---

  

Requirements:

- numpy

- pandas

- filterpy

- tensorflow

- scikit-learn