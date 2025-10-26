# src/yf_history/feed.py
from __future__ import annotations
import pandas as pd
from typing import Literal
from .reliable_cache import HistoryService, Interval, canon_interval


class HistoryFeed:
    """Clock-driven feed that emits new bars as time advances."""

    def __init__(
        self,
        history: HistoryService,
        *,
        mode: Literal["replay", "live"] = "replay",
        start: str | pd.Timestamp | None = None,
        speed: float = 1.0,
        interval: Interval = "1m",
        symbol: str | None = None,
        naive_tz: str = "America/New_York",
        rth_only: bool = False,
        align: bool | str = False,
        fill: str | None = None,
    ):
        if mode == "replay" and start is None:
            raise ValueError("replay mode requires `start`")

        self.history = history
        self.mode = mode
        self.speed = float(speed)
        self.interval = canon_interval(interval)
        self.symbol = symbol
        self.naive_tz = naive_tz
        self.rth_only = rth_only
        self.align = align
        self.fill = fill

        self._real_anchor = self._utc_now()
        self._virtual_anchor = (
            self._to_utc(start, naive_tz) if start else self._real_anchor
        )

        # gate (last fully closed bar)
        self._gate_ts = self._gate(self.interval, self._virtual_anchor)
        self._last_bar = self._gate_ts  # watermark for emitted bars

    # ------------------------------------------------------------------
    def now(self) -> pd.Timestamp:
        """Current virtual/live UTC time."""
        if self.mode == "live":
            return self._utc_now()
        delta = self._utc_now() - self._real_anchor
        return self._virtual_anchor + pd.to_timedelta(delta.total_seconds() * self.speed, unit="s")

    def get_new(self, symbol: str | None = None) -> pd.DataFrame:
        """Emit bars newly closed since the previous call."""
        sym = symbol or self.symbol
        if not sym:
            raise ValueError("symbol must be set")

        # compute current gate (latest fully closed bar)
        new_gate = self._gate(self.interval, self.now())
        old_gate = self._gate_ts
        if new_gate <= old_gate:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"]).set_index(
                pd.DatetimeIndex([], tz="UTC")
            )

        # pull new slice (exclusive old_gate, inclusive new_gate)
        df = self.history.get_history(
            sym,
            start=old_gate,
            end=new_gate,
            interval=self.interval,
            policy="auto_extend",
            rth_only=self.rth_only,
            align=self.align,
            fill=self.fill,
        )

        if df is None or df.empty:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"]).set_index(
                pd.DatetimeIndex([], tz="UTC")
            )

        df = df[(df.index > old_gate) & (df.index <= new_gate)]
        self._gate_ts = new_gate
        self._last_bar = df.index[-1] if not df.empty else old_gate
        return df

    # ------------------------------------------------------------------
    def _utc_now(self) -> pd.Timestamp:
        t = pd.Timestamp.utcnow()
        return t.tz_localize(None).tz_localize("UTC")

    def _to_utc(self, ts: str | pd.Timestamp, tz: str) -> pd.Timestamp:
        t = pd.Timestamp(ts)
        if t.tz is None:
            t = t.tz_localize(tz).tz_convert("UTC")
        else:
            t = t.tz_convert("UTC")
        return t

    def _gate(self, interval: Interval, now_utc: pd.Timestamp) -> pd.Timestamp:
        """Return the most recent fully closed bar close time for given interval."""
        if interval == "1d":
            ny = now_utc.tz_convert(self.naive_tz)
            close = ny.replace(hour=16, minute=0, second=0, microsecond=0)
            if ny < close:
                ny = (ny - pd.Timedelta(days=1)).normalize()
            else:
                ny = ny.normalize()
            return ny.tz_localize(self.naive_tz).tz_convert("UTC").normalize()
        freq = {"1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min", "60m": "60min"}[interval]
        return now_utc.floor(freq)
