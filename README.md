## A Python Module to Study Mispricing Strategies

This module runs backtest of a Kalman Filter and Ordinary Least Square strategy used to deterimine and exploit mispricing

Clone the module via:

```bash
git clone git@github.com:saeeron/pair_trading.git
cd pair_trading
```
Get all cryptos data from binance:
```python
from mispricing return_all_cryptos, KalmanPairSpread, KalmanTripletSpread, SpreadOLS, spread_backtest 
import pandas as pd
df = return_all_cryptos("4h", "2021-01-01", "2022-01-01")
```
Run a backtest on retrieved data:
```python
sr, data = spread_backtest(df, ["NEOUSDT","QTUMUSDT"], KalmanPairSpread, transaction_cost = 20)

```
`sr` is the Sharpe Ratio and `data` includes all time-series details of spreads, signals and returns.  


Requirements:
- numpy
- pandas
- filterpy

