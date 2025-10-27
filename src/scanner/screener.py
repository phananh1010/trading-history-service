"""
Scanner implementation providing:
 - fetch_universe(): list[dict]
 - top_mcap(k): list[str]
 - score(symbols): pd.DataFrame

Adds per-symbol, per-day caching under ~/.cache/stock_universe/YYYY-MM-DD/<SYMBOL>.json
so ~7k symbols are only scored once per (prev business) day.
"""

from __future__ import annotations
import json
import time
from typing import Callable, Iterable, Optional
from typing import Union
from datetime import date as _date
from pathlib import Path

import pandas as pd

from yf_history.reliable_cache import HistoryService
from .metrics import swing_metrics, volume_score, ar_score, atr_score, spp_score
from .universe import fetch_nasdaq_universe


class Scanner:
    """Minimal scanner with exactly three public methods:

    1) fetch_universe() -> list[dict]
       Returns items with at least {"symbol": str, "marketCap": float}.

    2) top_mcap(k: int) -> list[str]
       Returns top-k symbols by market cap from the cached universe.

    3) score(symbols: list[str]) -> pd.DataFrame
       Computes swing/volume/AR/ATR/support metrics and caches per-symbol results
       at ~/.cache/stock_universe/YYYY-MM-DD/SYMBOL.json (NY prev business day).

    Notes
    -----
    - All price history is fetched via the provided HistoryService.
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
        cache_root: Optional[Path] = None,
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

        # ~/.cache/stock_universe/
        self.cache_root = (
            Path(cache_root) if cache_root is not None
            else Path.home() / ".cache" / "stock_universe"
        )

    # --------------------- internal helpers (cache) ---------------------
    def _normalize_asof(self, as_of: Union[str, pd.Timestamp, _date, None], *, tz: str) -> pd.Timestamp:
        """
        Returns a NY-local midnight timestamp for the requested as_of 'trading day'.
        If as_of is None -> previous NY business day (existing behavior).
        If as_of is a weekend/holiday, roll back to the previous business day.
        """
        if as_of is None:
            now_ts = pd.Timestamp.now(tz)
            return (now_ts.normalize() - pd.offsets.BDay(1))

        t = pd.Timestamp(as_of)
        if t.tz is None:
            t = t.tz_localize(tz)
        else:
            t = t.tz_convert(tz)
        # roll to previous business day if not a business day
        if pd.tseries.offsets.BDay().is_on_offset(t.normalize()):
            return t.normalize()
        return (t.normalize() - pd.offsets.BDay(1))

    def _trading_day_str(self, as_of_ts: pd.Timestamp) -> str:
        """Format NY-local trading day as YYYY-MM-DD."""
        return as_of_ts.date().isoformat()

    def _cache_dir_for_day(self, day_str: str) -> Path:
        d = self.cache_root / day_str
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _score_path(self, day_str: str, symbol: str) -> Path:
        return self._cache_dir_for_day(day_str) / f"{symbol}.json"

    def _load_cached_score(self, day_str: str, symbol: str) -> Optional[dict]:
        p = self._score_path(day_str, symbol)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text())
        except Exception:
            # Corrupt cache; ignore this file
            return None

    def _save_cached_score(self, day_str: str, payload: dict) -> None:
        symbol = payload.get("symbol")
        if not symbol:
            return
        p = self._score_path(day_str, symbol)
        tmp = p.with_suffix(".json.tmp")
        try:
            tmp.write_text(json.dumps(payload, separators=(",", ":"), ensure_ascii=False))
            tmp.replace(p)  # atomic-ish on POSIX
        finally:
            if tmp.exists():
                try:
                    tmp.unlink()
                except Exception:
                    pass

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

    def score(
        self,
        symbols: list[str],
        *,
        atr_n: int = 14,
        as_of: Union[str, pd.Timestamp, _date, None] = None,
    ) -> pd.DataFrame:
        # resolve the trading day (NY) and cache key
        as_of_ts_ny = self._normalize_asof(as_of, tz=self.tz)
        day_str = self._trading_day_str(as_of_ts_ny)

        # --- normalize & filter symbols early ---
        def _normalize_yf_symbol(sym: str) -> str | None:
            if not sym or not isinstance(sym, str):
                return None
            s = sym.strip().upper()
            s = s.replace("/", "-").replace(".", "-")
            if len(s) == 0 or any(ch in s for ch in ["?", " "]):
                return None
            return s

        symbols = [_normalize_yf_symbol(s) for s in symbols]
        symbols = [s for s in symbols if s is not None]

        # time window (lookback_months -> as_of 16:00 NY)
        end_ny_close = as_of_ts_ny.replace(hour=16, minute=0, second=0, microsecond=0)
        start_ny = (as_of_ts_ny + pd.DateOffset(months=-self.lookback_months)).replace(
            hour=9, minute=30, second=0, microsecond=0
        )

        def fmt_local(ts: pd.Timestamp) -> str:
            # naive strings interpreted in HistoryService.naive_tz
            return ts.strftime("%Y-%m-%d %H:%M")

        # Pass 1: compute & cache only for missing symbols
        for sym in symbols:
            if self._load_cached_score(day_str, sym) is not None:
                continue  # already cached for this day

            try:
                if self.request_pause_s:
                    time.sleep(self.request_pause_s)

                bars = self.history.get_history(
                    sym,
                    start=fmt_local(start_ny),
                    end=fmt_local(end_ny_close),
                    interval="1d",
                    rth_only=self.rth_only,
                    align=self.align,
                    fill=self.fill,
                )
                if bars is None or bars.empty:
                    continue

                # ensure we only use data up to (and including) as_of day
                try:
                    end_utc = end_ny_close.tz_convert("UTC")
                except Exception:
                    end_utc = end_ny_close.tz_localize(self.tz).tz_convert("UTC")
                bars = bars.loc[:end_utc]

                sw = swing_metrics(bars)
                vol_mil = float(volume_score(bars))
                ar = float(ar_score(bars))
                atr = float(atr_score(bars, n=atr_n))
                spp = spp_score(bars)  # compute once

                payload = {
                    "asof": day_str,
                    "symbol": sym,
                    "swing_score": float(sw.get("swing_score", 0.0)),
                    "swing_amp": float(sw.get("swing_amp", 0.0)),
                    "swing_freq": float(sw.get("swing_freq", 0.0)),
                    "swing_auto": float(sw.get("swing_auto", 0.0)),
                    "volume_mil": vol_mil,
                    "AR_score": ar,
                    "ATR_score": atr,
                    "support_primary": float(spp.get("support_primary", float("nan"))),
                    "dist_supp_atr": float(spp.get("dist_primary_atr", float("nan"))),
                    "n_bars": int(len(bars)),
                }
                self._save_cached_score(day_str, payload)

            except Exception as e:
                self._errors.append((sym, str(e)))
                continue

        # Pass 2: load all requested symbols from cache and assemble DataFrame
        rows: list[dict] = []
        for sym in symbols:
            rec = self._load_cached_score(day_str, sym)
            if rec is not None:
                rows.append(rec)

        if not rows:
            return pd.DataFrame(columns=[
                "swing_score","swing_amp","swing_freq","swing_auto",
                "volume_mil","AR_score","ATR_score","support_primary",
                "dist_supp_atr","n_bars"
            ])

        df = (
            pd.DataFrame(rows)
              .set_index("symbol")
              .sort_index()
              .drop(columns=["asof"], errors="ignore")
        )
        return df


# ---------------------- Back-compat convenience ----------------------
def daily_last_6m(
    hist: HistoryService,
    symbol: str,
    tz: str = "America/New_York",
    as_of: Union[str, pd.Timestamp, _date, None] = None,
) -> pd.DataFrame:
    tz = tz or "America/New_York"
    if as_of is None:
        yday = (pd.Timestamp.now(tz).normalize() - pd.offsets.BDay(1))
    else:
        t = pd.Timestamp(as_of)
        t = t.tz_localize(tz) if t.tz is None else t.tz_convert(tz)
        yday = t.normalize() if pd.tseries.offsets.BDay().is_on_offset(t.normalize()) else (t.normalize() - pd.offsets.BDay(1))
    start = (yday + pd.DateOffset(months=-6)).replace(hour=9, minute=30)

    def fmt(d, h=0, m=0):
        return (d + pd.Timedelta(hours=h, minutes=m)).strftime("%Y-%m-%d %H:%M")

    return hist.get_history(
        symbol,
        start=fmt(start),
        end=fmt(yday.replace(hour=16, minute=0)),
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
