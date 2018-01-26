"""
Data Generation
"""
import pandas as pd
import numpy as np
from datetime import datetime

__author__ = 'Seung Hyeon Yu'
__email__ = 'rambor12@business.kaist.ac.kr'

NOW = pd.to_datetime(datetime.now().date())


def geometric_brownian_motion(N=100, T=1, mu=0.1, sigma=0.1, S0=100):

    dt = float(T)/N
    t = np.linspace(0, T, N)
    W = np.random.standard_normal(size=N)
    W = np.cumsum(W)*np.sqrt(dt) ### standard brownian motion ###
    X = (mu-0.5*sigma**2)*t + sigma*W
    S = S0*np.exp(X) ### geometric brownian motion ###

    return S




import KSIF as kf

kf.Strategy













def price(ticker, start='2000-01-01', end=NOW, freq='D', *args, **kwargs):

    dates = pd.date_range(start=start, end=end, freq=freq)
    if freq.lower() == 'd':
        T = 1
    elif freq == 'm':
        T = 12
    elif freq == 'y':
        T = 252
    else:
        raise NotImplementedError
    price = pd.DataFrame(data=geometric_brownian_motion(N=len(dates), T=T, *args, **kwargs),
                         index=dates,
                         columns=[ticker])

    return price


def prices(tickers, start='2000-01-01', end='2001-01-01', freq='D', *args, **kwargs):

    if isinstance(tickers, str):

        return price(ticker=tickers, start=start, end=end, freq=freq, *args, **kwargs)

    else:
        for ticker in tickers:
            if tickers.index(ticker) == 0:
                prices = price(ticker=ticker, start=start, end=end, freq=freq,  *args, **kwargs)
            else:
                prices = prices.join(price(ticker=ticker, start=start, end=end, freq=freq, *args, **kwargs))

        return prices

