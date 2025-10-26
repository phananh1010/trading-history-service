# TODO: scann nasdaq symbols, and rank them by metrics

import sys, os
import pandas as pd
print (sys.path)


from yf_history import CacheConfig, TimeSeriesCache, HistoryService
from scanner import Scanner

cache = TimeSeriesCache(CacheConfig(root="~/market_cache"))
hist  = HistoryService(cache, provider="yfinance", naive_tz="America/New_York")

scanner = Scanner(hist, lookback_months=6)

univ = scanner.fetch_universe()
symbols = [t["symbol"] for t in univ[:1000]]

scores = scanner.score(symbols)  # tidy DataFrame (index=symbol)

# Ranking is now simple pandas usage
top_sw  = scores.sort_values("swing_score", ascending=False).head(30)
top_vol = scores.sort_values("volume_mil", ascending=False).head(30)
top_ar  = scores.sort_values("AR_score", ascending=False).head(30)
top_atr = scores.sort_values("ATR_score", ascending=False).head(30)
