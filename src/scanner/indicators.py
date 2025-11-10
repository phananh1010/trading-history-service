from __future__ import annotations

"""
Generic indicator library (pure functions over OHLCV).

Contract:
- Stateless functions; depend only on pandas/numpy.
- Input is a pandas DataFrame with at least ['high','low','close'].
  'open' and 'volume' are optional depending on the function.
- Outputs are scalars or small dicts with fixed keys.

Versioned API: increment INDICATORS_VERSION for breaking changes.
"""

import numpy as np
import pandas as pd


INDICATORS_VERSION = "v1"


def swing(df: pd.DataFrame) -> dict[str, float]:
    """Swing characteristics over the provided window.

    Returns a dict with keys: 'swing_score', 'swing_amp', 'swing_freq', 'swing_auto'.

    Notes: Math intentionally mirrors the previous implementation.
    """
    s = df['close']
    r = np.log(s / s.shift(1))
    # Drop inf/NaN returns to avoid empty-length divide warnings
    r = r.replace([np.inf, -np.inf], np.nan).dropna()

    n = len(r)
    amp = float((r.std() * 2) if n else 0.0)  # rough 2σ amplitude

    if n:
        flips = int(np.sign(r).diff().ne(0).sum())
        freq = float(flips / n)  # sign-flip rate
    else:
        freq = 0.0

    ac1 = r.autocorr(1) if n else 0.0
    auto = float(-(ac1 if pd.notna(ac1) else 0.0))  # anti-trend bonus

    # If amp or freq is NaN (e.g., too few samples), score becomes NaN
    score = float(amp * freq * (1 + auto)) if (pd.notna(amp) and pd.notna(freq)) else float('nan')
    return {
        "swing_score": score,
        "swing_amp": amp,
        "swing_freq": freq,
        "swing_auto": auto,
    }


def volume_millions(df: pd.DataFrame) -> float:
    """Total traded volume across the window, in millions of shares."""
    return float(df['volume'].sum() / 1e6)


def ar(df: pd.DataFrame) -> float:
    """Average range proxy: mean((high - low) / low)."""
    return float(((df['high'] - df['low']) / df['low']).mean())

def ar_cross(df: pd.DataFrame) -> float:
    """Inter-day range proxy: mean((high_t - low_{t-1}) / low_{t-1})."""
    low_prev = df['low'].shift(2)
    ratio = (df['high'] - low_prev) / low_prev
    return float(ratio.dropna().mean())


def atr(df: pd.DataFrame, n: int = 14) -> float:
    """Average True Range using EWMA with alpha=1/n."""
    hl = df['high'] - df['low']
    hc = (df['high'] - df['close'].shift(1)).abs()
    lc = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return float(tr.ewm(alpha=1 / n, adjust=False).mean().iloc[-1])


def support(
    df: pd.DataFrame,
    *,
    atr_n: int = 14,
    k: int = 2,
    confirm_n: int = 3,
) -> dict[str, float]:
    """Support levels and ATR-normalized distances.

    Returns a dict with keys:
    - 'support_primary'
    - 'support_secondary'
    - 'dist_primary_atr'
    - 'dist_secondary_atr'

    Notes: Math intentionally mirrors the previous SupportMetrics.compute.
    """
    if df.empty or any(c not in df.columns for c in ["high", "low", "close"]):
        return {
            "support_primary": np.nan,
            "support_secondary": np.nan,
            "dist_primary_atr": np.nan,
            "dist_secondary_atr": np.nan,
        }

    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift(1)).abs()
    lc = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    atr_s = tr.ewm(alpha=1 / atr_n, adjust=False).mean()

    lows = df["low"]
    pivots = []
    for i in range(k, len(lows) - k):
        if lows.iloc[i] == lows.iloc[i - k:i + k + 1].min():
            future = df["close"].iloc[i + 1:i + 1 + confirm_n]
            if (future > lows.iloc[i]).all():
                pivots.append((df.index[i], lows.iloc[i]))
    if not pivots:
        return {
            "support_primary": np.nan,
            "support_secondary": np.nan,
            "dist_primary_atr": np.nan,
            "dist_secondary_atr": np.nan,
        }

    piv_df = pd.DataFrame(pivots, columns=["ts", "low"]).set_index("ts")
    piv_df["age"] = (df.index[-1] - piv_df.index).days
    piv_df["atr"] = atr_s.reindex(piv_df.index, method="ffill")
    piv_df["dist_atr"] = (df["close"].iloc[-1] - piv_df["low"]) / piv_df["atr"]
    piv_df = piv_df[piv_df["low"] <= df["close"].iloc[-1]]
    if piv_df.empty:
        return {
            "support_primary": np.nan,
            "support_secondary": np.nan,
            "dist_primary_atr": np.nan,
            "dist_secondary_atr": np.nan,
        }

    piv_df = piv_df.sort_values(["age", "low"], ascending=[True, False])
    primary = piv_df.iloc[0]
    secondary = piv_df.iloc[1] if len(piv_df) > 1 else primary

    return {
        "support_primary": float(primary["low"]),
        "support_secondary": float(secondary["low"]),
        "dist_primary_atr": float(primary["dist_atr"]),
        "dist_secondary_atr": float(secondary["dist_atr"]),
    }


def spp_primary(df: pd.DataFrame, *, atr_n: int = 14, k: int = 2, confirm_n: int = 3) -> float:
    """Thin alias for the primary support level (float).

    Provided for parity; prefer using support(df) for the full dict.
    """
    res = support(df, atr_n=atr_n, k=k, confirm_n=confirm_n)
    return float(res.get("support_primary", np.nan))


def uptrend_score(df: pd.DataFrame) -> float:
    """Composite year-long uptrend score: slope + persistence − drawdown.
    Returns a dimensionless score; higher = stronger/steadier uptrend.
    """
    if df is None or df.empty or "close" not in df.columns:
        return float("nan")
    s = pd.to_numeric(df["close"], errors="coerce").dropna()
    n = len(s)
    if n < 30:
        return float("nan")

    x = np.arange(n, dtype=float)
    slope = np.polyfit(x, s.values, 1)[0]                  # price per bar
    slope_norm = float(slope / (s.mean() if s.mean() else 1.0))

    r = s.pct_change().dropna()
    win_ratio = float((r > 0).mean())                      # 0..1

    roll_max = s.cummax()
    dd = (s - roll_max) / roll_max
    max_dd = float(dd.min()) if len(dd) else 0.0           # negative

    # weights: drift (0.4) + persistence (0.3) + stability (0.3)
    #return float(0.4 * slope_norm + 0.3 * (win_ratio - 0.5) + 0.3 * (1.0 + max_dd))

    return float(0.1 * slope_norm + 0.8 * (win_ratio - 0.5) + 0.1 * (1.0 + max_dd))

import numpy as np
import pandas as pd

def smooth_score(df: pd.DataFrame, col: str = "close") -> tuple[float, float]:
    """
    Fit a linear model to price normalized to base 1000 and return:
    (MSE, slope_per_bar).

    - Normalization: y_norm = price / price[0] * 1000.
    - Linear fit: least squares on x = [0..n-1].
    - MSE: mean squared error of residuals wrt the fit line, in normalized units.
    - slope_per_bar: fitted slope (delta per bar) on the normalized scale.
    """
    if df is None or df.empty or col not in df.columns:
        return float("nan"), float("nan")

    s = pd.to_numeric(df[col], errors="coerce").dropna()
    n = len(s)
    if n < 2:
        return float("nan"), float("nan")

    y = s.values.astype(float)
    y0 = y[0]
    if not np.isfinite(y0) or y0 == 0.0:
        return float("nan"), float("nan")

    # Normalize to base 1000
    y_norm = y * (1000.0 / y0)
    x = np.arange(n, dtype=float)

    # Linear regression (degree 1 polyfit)
    slope, intercept = np.polyfit(x, y_norm, 1)
    y_hat = slope * x + intercept
    resid = y_norm - y_hat
    mse = float(np.mean(resid ** 2))

    return mse, float(slope)


def d_sma200(df: pd.DataFrame) -> float:
    """Signed distance of last close to SMA(200): (close - SMA200) / SMA200.

    Uses window=min(200, n). Requires at least 20 closes to avoid noisy outputs.
    Returns NaN if insufficient data or SMA is zero/NaN.
    """
    if df is None or df.empty or "close" not in df.columns:
        return float("nan")

    s = pd.to_numeric(df["close"], errors="coerce").dropna()
    n = len(s)
    if n < 20:
        return float("nan")

    w = min(200, n)
    sma = s.rolling(window=w, min_periods=20).mean().iloc[-1]
    if not np.isfinite(sma) or sma == 0:
        return float("nan")

    c = float(s.iloc[-1])
    return float((c - sma) / sma)
