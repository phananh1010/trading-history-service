from pathlib import Path
from yf_history.reliable_cache import CacheConfig, TimeSeriesCache, HistoryService
from yf_history.feed import HistoryFeed
import time

cache = TimeSeriesCache(CacheConfig(root=Path("~/.hist_cache")))
svc = HistoryService(cache)

feed = HistoryFeed(
    svc,
    mode="replay",
    start="2025-10-24 10:00",
    speed=12.0,
    interval="1m",
    symbol="AAPL",
)

while True:
    df = feed.get_new()
    if not df.empty:
        print("New bar(s):")
        print(df.tail())
    time.sleep(5)  # 5s real = 1m virtual @12x speed