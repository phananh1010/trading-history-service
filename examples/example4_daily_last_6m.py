from pathlib import Path
import pandas as pd

from yf_history import CacheConfig, TimeSeriesCache, HistoryService
from scanner import daily_last_6m
from scanner.indicators import atr, ar, ar_cross, swing, support, volume_millions

# --- setup ---
cache = TimeSeriesCache(CacheConfig(root=Path("./.yf_cache")))
hist = HistoryService(cache, provider="yfinance", naive_tz="America/New_York")

symbol = "AAPL"

# =========================
# Daily last 6 months
# =========================
daily_bars = daily_last_6m(hist, symbol)
print(f"daily: {symbol} -> {len(daily_bars)} bars")

daily = {
    **swing(daily_bars),
    "volume_mil": volume_millions(daily_bars),
    "AR_score": ar(daily_bars),
    "AR_cross": ar_cross(daily_bars),
    "ATR_score": atr(daily_bars, n=14),
    **support(daily_bars, atr_n=14),
}
print("daily metrics:")
for k, v in daily.items():
    print(f"  {k:18s} = {v}")

# =========================
# Intraday (30m) recent 5 business days
# =========================
tz = "America/New_York"
as_of = pd.Timestamp.now(tz).normalize()
if not pd.tseries.offsets.BDay().is_on_offset(as_of):
    as_of = as_of - pd.offsets.BDay(1)
start = (as_of - pd.offsets.BDay(5)).replace(hour=9, minute=30)
end = as_of.replace(hour=16, minute=0)

intraday_bars = hist.get_history(
    symbol=symbol,
    start=start.strftime("%Y-%m-%d %H:%M"),
    end=end.strftime("%Y-%m-%d %H:%M"),
    interval="30m",
    policy="auto_extend",
    rth_only=True,
    align="30min",
    fill="ffill_zero_volume",
)
print(f"intraday(30m): {symbol} -> {len(intraday_bars)} bars")

intraday = {
    **swing(intraday_bars),
    "volume_mil": volume_millions(intraday_bars),
    "AR_score": ar(intraday_bars),
    "AR_cross": ar_cross(intraday_bars),
    "ATR_score": atr(intraday_bars, n=14),
    **support(intraday_bars, atr_n=14),
}
print("intraday metrics (30m):")
for k, v in intraday.items():
    print(f"  {k:18s} = {v}")
