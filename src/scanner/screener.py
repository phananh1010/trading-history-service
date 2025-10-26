"""
Scanner implementation providing:
 - fetch_universe(): list[dict]
 - top_mcap(k): list[str]
 - score(symbols): pd.DataFrame

Retains the original functional screen() helper for convenience, but removes rank().
"""

from __future__ import annotations
import time
from typing import Callable, Iterable, Optional

import pandas as pd

from yf_history.reliable_cache import HistoryService
from .metrics import swing_metrics, volume_score, ar_score, atr_score
from .universe import fetch_nasdaq_universe


class Scanner:
    """Minimal scanner with exactly three public methods:

    1) fetch_universe() -> list[dict]
       Returns items with at least {"symbol": str, "marketCap": float}.

    2) top_mcap(k: int) -> list[str]
       Returns top-k symbols by market cap from the cached universe.

    3) score(symbols: list[str]) -> pd.DataFrame
       Computes swing/volume/AR/ATR metrics over a daily lookback window.

    Notes
    -----
    - All price history is fetched via the provided HistoryService.
    - Results are deterministic for a given run (start/end are stamped).
    - Symbols that fail to fetch are skipped; details in self._errors.
    """

    def __init__(
        self,
        history: HistoryService,
        *,
        lookback_months: int = 6,
        tz: str = "America/New_York",
        rth_only: bool = True,
        align: bool | str = True,
        fill: Optional[str] = "ffill_zero_volume",
        universe_source: str | Callable[[], list[dict]] = "nasdaq",
        request_pause_s: float = 0.0,
    ) -> None:
        self.history = history
        self.lookback_months = int(lookback_months)
        self.tz = tz
        self.rth_only = bool(rth_only)
        self.align = align
        self.fill = fill
        self.universe_source = universe_source
        self.request_pause_s = float(request_pause_s)

        self._universe: Optional[list[dict]] = None
        self._errors: list[tuple[str, str]] = []  # (symbol, reason)

    # -------------------------- public API ---------------------------
    def fetch_universe(self) -> list[dict]:
        if self._universe is not None:
            return self._universe
        if callable(self.universe_source):
            data = self.universe_source()
        else:
            src = str(self.universe_source).lower()
            if src == "nasdaq":
                data = fetch_nasdaq_universe()
            else:
                raise ValueError(f"unknown universe source: {self.universe_source}")
        # Normalize: ensure required keys
        norm = []
        for t in data:
            sym = t.get("symbol") or t.get("Symbol")
            mcap = t.get("marketCap") or t.get("MarketCap")
            if not sym:
                continue
            # marketCap already normalized in universe.fetch_nasdaq_universe
            try:
                mcap_f = float(mcap) if mcap is not None else 0.0
            except Exception:
                mcap_f = 0.0
            norm.append({"symbol": sym, "marketCap": mcap_f, **t})
        self._universe = norm
        return self._universe

    def top_mcap(self, k: int) -> list[str]:
        u = self.fetch_universe()
        u_sorted = sorted(u, key=lambda d: float(d.get("marketCap", 0.0)), reverse=True)
        return [d["symbol"] for d in u_sorted[: int(k)]]

    def score(self, symbols: list[str], *, atr_n: int = 14) -> pd.DataFrame:
        rows: list[dict] = []
        now_ts = pd.Timestamp.now(self.tz)
        today = now_ts.normalize()
        yday = (today - pd.offsets.BDay(1)).normalize()
        start = (today - pd.DateOffset(months=self.lookback_months)).replace()

        # --- normalize & filter symbols early ---
        def _normalize_yf_symbol(sym: str) -> str | None:
            if not sym or not isinstance(sym, str):
                return None
            s = sym.strip().upper()
            # Convert slash or dot separators used by Nasdaq → Yahoo style
            s = s.replace("/", "-").replace(".", "-")
            # Filter out clearly invalid symbols
            if len(s) == 0 or any(ch in s for ch in ["?", " "]):
                return None
            return s

        symbols = [_normalize_yf_symbol(s) for s in symbols]
        symbols = [s for s in symbols if s is not None]

        def fmt(d, h=0, m=0):
            return (d + pd.Timedelta(hours=h, minutes=m)).strftime("%Y-%m-%d %H:%M")

        for sym in symbols:
            try:
                if self.request_pause_s:
                    time.sleep(self.request_pause_s)
                bars = self.history.get_history(
                    sym,
                    start=fmt(start, 9, 30),
                    end=fmt(yday, 16, 0),
                    interval="1d",
                    rth_only=self.rth_only,
                    align=self.align,
                    fill=self.fill,
                )
                if bars is None or bars.empty:
                    continue
                sw = swing_metrics(bars)
                rows.append(
                    {
                        "symbol": sym,
                        "swing_score": sw["swing_score"],
                        "swing_amp": sw["swing_amp"],
                        "swing_freq": sw["swing_freq"],
                        "swing_auto": sw["swing_auto"],
                        "volume_mil": volume_score(bars),
                        "AR_score": ar_score(bars),
                        "ATR_score": atr_score(bars, n=atr_n),
                        "n_bars": int(len(bars)),
                    }
                )
            except Exception as e:
                self._errors.append((sym, str(e)))
                continue
        return pd.DataFrame(rows).set_index("symbol").sort_index()


# ---------------------- Back-compat convenience ----------------------
def daily_last_6m(hist: HistoryService, symbol: str, tz: str = "America/New_York") -> pd.DataFrame:
    now_ts = pd.Timestamp.now(tz)
    today = now_ts.normalize()
    yday = (today - pd.offsets.BDay(1)).normalize()
    start = (today - pd.DateOffset(months=6)).replace()

    def fmt(d, h=0, m=0):
        return (d + pd.Timedelta(hours=h, minutes=m)).strftime("%Y-%m-%d %H:%M")

    return hist.get_history(
        symbol,
        start=fmt(start, 9, 30),
        end=fmt(yday, 16, 0),
        interval="1d",
        rth_only=True,
        align=True,
        fill="ffill_zero_volume",
    )


def screen(hist: HistoryService, symbols: Iterable[str], *, atr_n: int = 14) -> pd.DataFrame:
    """Return a tidy DataFrame with one row per symbol and metric columns.

    Thin wrapper that delegates to Scanner for backward compatibility.
    """
    scanner = Scanner(hist)
    return scanner.score(list(symbols), atr_n=atr_n)
