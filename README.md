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

refer to `examples` folder for use cases

## Notes
- This package intentionally includes only the yfinance path; Polygon/Alpaca code and keys are not required here.
- Cache files live under `~/market_cache/parquet` by default; configure via `CacheConfig(root=...)`.
- Indexes are stored/returned in tz-aware UTC. The service normalizes naive timestamps using `naive_tz`.
