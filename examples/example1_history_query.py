from pathlib import Path
import pandas as pd
from yf_history import CacheConfig, TimeSeriesCache, HistoryService

# --- One-time setup ---
cache = TimeSeriesCache(CacheConfig(root=Path("./.yf_cache")))
hist = HistoryService(cache, provider="yfinance", naive_tz="America/New_York")

# Common request parameters (choose a business-day window)
symbol = "AAPL"
interval = "1h"                        # normalized to "60m" internally
policy = "auto_extend"                 # fills left/right cache gaps
rth_only = True                        # 09:30–16:00 NY, Mon–Fri
align = "60min"                        # explicit grid (comprehensive: not just True/False)
fill = "ffill_zero_volume"             # forward-fill O/H/L/C; volume -> 0

# =========================
# 1) Naive local strings (interpreted in naive_tz="America/New_York")
# =========================
bars_naive_local = hist.get_history(
    symbol=symbol,
    start="2024-09-03 09:30",          # naive local time
    end="2024-09-05 16:00",            # naive local time
    interval=interval,
    policy=policy,
    rth_only=rth_only,
    align=align,
    fill=fill,
)

# =========================
# 2) ISO 8601 strings with numeric timezone offset (tz-aware)
# =========================
bars_iso_offset = hist.get_history(
    symbol=symbol,
    start="2024-09-03T09:30:00-04:00",  # EDT offset
    end="2024-09-05T16:00:00-04:00",
    interval=interval,
    policy=policy,
    rth_only=rth_only,
    align=align,
    fill=fill,
)

# =========================
# 3) pandas Timestamps with IANA timezone (tz-aware)
# =========================
bars_pd_tzaware = hist.get_history(
    symbol=symbol,
    start=pd.Timestamp("2024-09-03 09:30", tz="America/New_York"),
    end=pd.Timestamp("2024-09-05 16:00", tz="America/New_York"),
    interval=interval,
    policy=policy,
    rth_only=rth_only,
    align=align,
    fill=fill,
)

# Optional sanity checks (not required)
print("naive_local:", bars_naive_local.index[:3])
print("iso_offset :", bars_iso_offset.index[:3])
print("pd_tzaware :", bars_pd_tzaware.index[:3])