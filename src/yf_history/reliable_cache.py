from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, cast
import pandas as pd
import numpy as np 


from ._index_ops import align_many_on_grid

# Supported intervals
Interval = Literal["1m", "5m", "15m", "30m", "60m", "1h", "1d"]


def canon_interval(iv: Interval) -> Interval:
    return cast(Interval, "60m" if iv == "1h" else iv)


@dataclass
class CacheConfig:
    root: Path
    provider: Literal["yfinance"] = "yfinance"
    mkdir: bool = True


class TimeSeriesCache:
    """
    Very small Parquet cache: one file per (symbol, interval).
    - Parquet index is tz-aware UTC.
    - Public API returns UTC index; callers can localize if desired.
    """

    def __init__(self, cfg: CacheConfig):
        self.root = Path(cfg.root).expanduser()
        if cfg.mkdir:
            (self.root / "parquet").mkdir(parents=True, exist_ok=True)

    def _path(self, symbol: str, interval: Interval) -> Path:
        return self.root / "parquet" / f"{symbol.upper()}__{canon_interval(interval)}.parquet"

    def read(self, symbol: str, interval: Interval) -> pd.DataFrame | None:
        p = self._path(symbol, interval)
        if not p.exists():
            return None
        df = pd.read_parquet(p)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        idx = df.index
        if idx.tz is None:
            df.index = idx.tz_localize("UTC")
        else:
            df.index = idx.tz_convert("UTC")
        df = df[~df.index.duplicated(keep="last")].sort_index()
        return df

    def write_atomic(self, symbol: str, interval: Interval, df: pd.DataFrame) -> None:
        p = self._path(symbol, interval)
        tmp = p.with_suffix(".tmp.parquet")
        out = df.copy()

        if not isinstance(out.index, pd.DatetimeIndex):
            out.index = pd.to_datetime(out.index)
        idx = out.index
        if idx.tz is None:
            out.index = idx.tz_localize("UTC")
        else:
            out.index = idx.tz_convert("UTC")
        out = out[~out.index.duplicated(keep="last")].sort_index()
        out.to_parquet(tmp)
        tmp.replace(p)


def _yf_interval(iv: Interval) -> str:
    return "60m" if iv == "1h" else iv


def _normalize_ohlcv_columns(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if df is None or df.empty:
        out = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        out.index.name = "datetime"
        return out
    if isinstance(df.columns, pd.MultiIndex):
        try:
            df = df.xs(symbol, level=0, axis=1)
        except Exception:
            df.columns = ["_".join(map(str, t)) for t in df.columns]
    df = df.rename(columns={c: str(c).lower() for c in df.columns})
    df.index.name = "datetime"
    want = ("open", "high", "low", "close", "volume")
    cols_present = set(df.columns)

    if not set(want).issubset(cols_present):
        resolved: dict[str, str] = {}

        def pick(name: str) -> str | None:
            if name in cols_present:
                return name
            syns = [name]
            if name == "close":
                syns += ["adj close", "adj_close", "adjclose"]
            for col in df.columns:
                cc = str(col).lower()
                for s in syns:
                    base = s.replace(" ", "_")
                    if cc == base or cc.replace(" ", "_") == base:
                        return col
                    if cc.startswith(base + "_") or cc.endswith("_" + base):
                        return col
            return None

        for name in want:
            sel = pick(name)
            if sel is not None:
                resolved[name] = sel

        if resolved:
            df = df[list(resolved.values())].rename(columns={v: k for k, v in resolved.items()})

    for c in want:
        if c not in df.columns:
            # Don't fabricate trades: keep volume as NaN if missing
            df[c] = 0.0 if c != "volume" else np.nan
    return df[list(want)]



def _fetch_yf(symbol: str, start: pd.Timestamp, end: pd.Timestamp, interval: Interval) -> pd.DataFrame:
    """
    Fetch via yfinance with robust handling for intraday ranges.
    - tz-aware UTC datetimes for inputs
    - fallback to period-based download for minute ranges beyond Yahoo limits
    """
    import yfinance as yf

    iv = _yf_interval(interval)

    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    s = s.tz_localize("UTC") if s.tz is None else s.tz_convert("UTC")
    e = e.tz_localize("UTC") if e.tz is None else e.tz_convert("UTC")

    e_inclusive = (e + (pd.Timedelta(days=1) if iv == "1d" else pd.Timedelta(seconds=1)))

    def _normalize(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or len(df) == 0:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"]).set_index(
                pd.DatetimeIndex([], tz="UTC")
            )
        df2 = _normalize_ohlcv_columns(df, symbol)
        idx = pd.to_datetime(df2.index)
        df2.index = (idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC"))
        df2 = df2[~df2.index.duplicated(keep="last")].sort_index()
        return df2

    # Attempt explicit start/end
    try:
        df = yf.download(
            symbol,
            start=s,
            end=e_inclusive,
            interval=iv,
            auto_adjust=False,
            progress=False,
        )
        df = _normalize(df)
    except Exception as ex:
        df = None
        first_err = ex
    else:
        first_err = None

    # Fallback for intraday minute ranges
    if (df is None or df.empty) and iv != "1d":
        span = (e - s)
        if span <= pd.Timedelta(days=1, minutes=5):
            period = "1d"
        elif span <= pd.Timedelta(days=7, minutes=5):
            period = "7d"
        else:
            period = "7d"
        try:
            df2 = yf.download(
                symbol,
                period=period,
                interval=iv,
                auto_adjust=False,
                progress=False,
            )
            df2 = _normalize(df2)
            df = df2.loc[s:e]
        except Exception:
            if first_err is not None:
                raise first_err
            raise

    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"]).set_index(
            pd.DatetimeIndex([], tz="UTC")
        )
    return df


class HistoryService:
    """yfinance-backed history service using a simple Parquet cache.

    - policy='auto_extend' (default): merge both left and right gaps.
    - policy='respect_cache': never fetch; return cached intersection if fully covered, else raise.
    - policy='force_refresh': refetch full [start, end] and overwrite.
    Naive datetimes are interpreted in `naive_tz` (default America/New_York), then converted to UTC.
    """

    def __init__(self, cache: TimeSeriesCache, provider: str = "yfinance", naive_tz: str = "America/New_York"):
        provider = (provider or "").lower()
        if provider != "yfinance":
            raise NotImplementedError("yf_history only supports provider='yfinance'")
        self.cache = cache
        self.provider = provider
        self.naive_tz = naive_tz

    def _fetch(self, symbol: str, start: pd.Timestamp, end: pd.Timestamp, interval: Interval) -> pd.DataFrame:
        return _fetch_yf(symbol, start, end, interval)

    def _normalize_range(
        self,
        start: str | pd.Timestamp,
        end: str | pd.Timestamp | None,
    ) -> tuple[pd.Timestamp, pd.Timestamp]:
        s = pd.Timestamp(start)
        if s.tz is None:
            s = s.tz_localize(self.naive_tz).tz_convert("UTC")
        else:
            s = s.tz_convert("UTC")
        if end is None:
            now_ts = pd.Timestamp.utcnow()
            e = now_ts.tz_localize("UTC") if now_ts.tz is None else now_ts.tz_convert("UTC")
        else:
            e = pd.Timestamp(end)
            if e.tz is None:
                e = e.tz_localize(self.naive_tz).tz_convert("UTC")
            else:
                e = e.tz_convert("UTC")
        return s, e

    def get_history(
        self,
        symbol: str | Iterable[str],
        start: str | pd.Timestamp,
        end: str | pd.Timestamp | None = None,
        interval: Interval = "1h",
        policy: Literal["auto_extend", "respect_cache", "force_refresh"] = "auto_extend",
        rth_only: bool = False,
        align: bool | str = False,
        fill: str | None = None,
    ):
        syms = [symbol] if isinstance(symbol, str) else list(symbol)
        interval = canon_interval(interval)
        s, e = self._normalize_range(start, end)
        s_slice = s
        e_slice = e
        if interval == "1d":
            s_slice = s.normalize()
            e_slice = e.normalize()

        out: dict[str, pd.DataFrame] = {}
        for sym in syms:
            cached = self.cache.read(sym, interval)

            if cached is None or policy == "force_refresh":
                fresh = self._fetch(sym, s, e, interval)
                self.cache.write_atomic(sym, interval, fresh)
                out[sym] = fresh.loc[s_slice:e_slice]
                continue

            c_start = cached.index.min()
            c_end = cached.index.max()
            need_left = s < c_start
            need_right = e > c_end

            if policy == "respect_cache":
                if need_left or need_right:
                    raise ValueError(
                        f"Cache miss for {sym} {interval}: have {c_start}–{c_end}, need {s}–{e}"
                    )
                out[sym] = cached.loc[s:e]
                continue

            pieces = [cached]

            if need_left:
                left = self._fetch(sym, s, c_start - pd.Timedelta(seconds=1), interval)
                if not left.empty:
                    pieces.insert(0, left)

            if need_right:
                right = self._fetch(sym, c_end + pd.Timedelta(seconds=1), e, interval)
                if not right.empty:
                    pieces.append(right)

            merged = pd.concat(pieces) if len(pieces) > 1 else pieces[0]
            merged = merged[~merged.index.duplicated(keep="last")].sort_index()
            self.cache.write_atomic(sym, interval, merged)

            out[sym] = merged.loc[s_slice:e_slice]

        def _filter_rth_utc(df: pd.DataFrame) -> pd.DataFrame:
            if df is None or df.empty:
                return df
            if interval == "1d":
                return df
            idx = pd.DatetimeIndex(df.index)
            idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
            local = idx.tz_convert("America/New_York")
            mask_time = (
                ((local.hour > 9) | ((local.hour == 9) & (local.minute >= 30))) & (local.hour < 16)
            )
            mask_dow = local.weekday < 5
            mask = mask_time & mask_dow
            return df.loc[mask]

        if rth_only:
            for k in list(out.keys()):
                out[k] = _filter_rth_utc(out[k])

        if align:
            if isinstance(align, str) and align.lower() not in ("true", "false"):
                freq = align
            else:
                if interval == "1d":
                    freq = "1D"
                else:
                    freq = interval.replace("m", "min")

            if interval == "1d":
                start_align = s_slice
                end_align = e_slice
            else:
                start_align = s
                end_align = e

            out = align_many_on_grid(out, start=start_align, end=end_align, freq=freq)

            if rth_only:
                for k in list(out.keys()):
                    out[k] = _filter_rth_utc(out[k])

            if fill and fill.lower() == "ffill_zero_volume":
                for k, df in out.items():
                    if df is None or df.empty:
                        continue
                    for c in ("open", "high", "low", "close"):
                        if c in df.columns:
                            s = pd.to_numeric(df[c], errors="coerce")
                            df[c] = s.ffill()
                    if "volume" in df.columns:
                        s = pd.to_numeric(df["volume"], errors="coerce")
                        df["volume"] = s.fillna(0)
                    out[k] = df
    
            elif fill and fill.lower() in ("ffill_prices_nan_volume", "prices_only"):
                # Forward-fill prices but keep volume untouched (NaN on synthetic bars)
                for k, df in out.items():
                    if df is None or df.empty:
                        continue
                    for c in ("open", "high", "low", "close"):
                        if c in df.columns:
                            s = pd.to_numeric(df[c], errors="coerce")
                            df[c] = s.ffill()
                    # DO NOT touch 'volume'
                    out[k] = df


        if isinstance(symbol, str):
            return out[syms[0]]
        return out

