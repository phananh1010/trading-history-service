# yf-history

A lightweight, self-contained yfinance history service with a local Parquet cache. This package copies only the yfinance provider logic from your repo, so you can import and use it independently without changing the original codebase.

## Install (editable)

- From the repo root:

```
pip install -e ./package/yf_history
```

Dependencies: pandas, pyarrow, yfinance, python-dotenv.

## Environment

An example `.env` is placed alongside this package. yfinance does not require API keys, but you can still use `.env` for any local overrides. If you plan to publish this package, replace secrets with placeholders first.

To load `.env` automatically in your app:

```
from dotenv import load_dotenv; load_dotenv()
```

## Usage

- Minimal example (same API as before):

```
from yf_history import CacheConfig, TimeSeriesCache, HistoryService

cache = TimeSeriesCache(CacheConfig(root="~/market_cache"))
hist = HistoryService(cache, provider="yfinance", naive_tz="America/New_York")

# Single symbol
bars = hist.get_history(
    symbol="AAPL",
    start="2024-09-01 09:30 America/New_York",
    end="2024-09-03 16:00 America/New_York",
    interval="1h",
    policy="auto_extend",
    align=True,
    fill="ffill_zero_volume",
)

# Multiple symbols -> dict[str, DataFrame]
multi = hist.get_history(["AAPL", "MSFT"], start="2024-09-01", end="2024-09-10", interval="1d")
```

- Drop-in for your existing notebooks/code:

Replace the import from `tradingmini.reliable_cache` with `yf_history` and keep the rest intact:

```
from yf_history import CacheConfig, TimeSeriesCache, HistoryService

cache = TimeSeriesCache(CacheConfig(root="~/market_cache"))
hist_yf = HistoryService(cache, provider="yfinance", naive_tz="America/New_York")
hist = hist_yf
```

## Notes
- This package intentionally includes only the yfinance path; Polygon/Alpaca code and keys are not required here.
- Cache files live under `~/market_cache/parquet` by default; configure via `CacheConfig(root=...)`.
- Indexes are stored/returned in tz-aware UTC. The service normalizes naive timestamps using `naive_tz`.
