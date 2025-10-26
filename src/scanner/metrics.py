# src/yf_history/metrics.py
from __future__ import annotations
import numpy as np
import pandas as pd

def swing_metrics(d180: pd.DataFrame) -> dict[str, float]:
    s = d180['close']
    r = np.log(s / s.shift(1)).dropna()
    amp  = float(r.std() * 2)                          # rough 2σ amplitude
    freq = float(np.sign(r).diff().ne(0).sum() / len(r))  # sign-flip rate
    auto = float(-(r.autocorr(1) or 0))                # anti-trend bonus
    score = float(amp * freq * (1 + auto))
    return {"swing_score": score, "swing_amp": amp, "swing_freq": freq, "swing_auto": auto}

def volume_score(d180: pd.DataFrame) -> float:
    return float(d180['volume'].sum() / 1e6)

def ar_score(d180: pd.DataFrame) -> float:
    # average high-low range vs low
    return float(((d180['high'] - d180['low']) / d180['low']).mean())

def atr_score(d180: pd.DataFrame, n: int = 14) -> float:
    hl = d180['high'] - d180['low']
    hc = (d180['high'] - d180['close'].shift(1)).abs()
    lc = (d180['low']  - d180['close'].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return float(tr.ewm(alpha=1/n, adjust=False).mean().iloc[-1])

class SupportMetrics:
    def __init__(self, d180: pd.DataFrame, *, atr_n: int = 14, k: int = 2, confirm_n: int = 3):
        self.df = d180.copy()
        self.atr_n = atr_n
        self.k = k
        self.confirm_n = confirm_n

    def compute(self) -> dict[str, float]:
        df = self.df
        if df.empty or any(c not in df.columns for c in ["high", "low", "close"]):
            return {"support_primary": np.nan, "support_secondary": np.nan, "dist_primary_atr": np.nan, "dist_secondary_atr": np.nan}

        hl = df["high"] - df["low"]
        hc = (df["high"] - df["close"].shift(1)).abs()
        lc = (df["low"] - df["close"].shift(1)).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1 / self.atr_n, adjust=False).mean()

        lows = df["low"]
        pivots = []
        for i in range(self.k, len(lows) - self.k):
            if lows[i] == lows[i - self.k:i + self.k + 1].min():
                future = df["close"].iloc[i + 1:i + 1 + self.confirm_n]
                if (future > lows[i]).all():
                    pivots.append((df.index[i], lows[i]))
        if not pivots:
            return {"support_primary": np.nan, "support_secondary": np.nan, "dist_primary_atr": np.nan, "dist_secondary_atr": np.nan}

        piv_df = pd.DataFrame(pivots, columns=["ts", "low"]).set_index("ts")
        piv_df["age"] = (df.index[-1] - piv_df.index).days
        piv_df["atr"] = atr.reindex(piv_df.index, method="ffill")
        piv_df["dist_atr"] = (df["close"].iloc[-1] - piv_df["low"]) / piv_df["atr"]
        piv_df = piv_df[piv_df["low"] <= df["close"].iloc[-1]]
        if piv_df.empty:
            return {"support_primary": np.nan, "support_secondary": np.nan, "dist_primary_atr": np.nan, "dist_secondary_atr": np.nan}

        piv_df = piv_df.sort_values(["age", "low"], ascending=[True, False])
        primary = piv_df.iloc[0]
        secondary = piv_df.iloc[1] if len(piv_df) > 1 else primary

        return {
            "support_primary": float(primary["low"]),
            "support_secondary": float(secondary["low"]),
            "dist_primary_atr": float(primary["dist_atr"]),
            "dist_secondary_atr": float(secondary["dist_atr"]),
        }