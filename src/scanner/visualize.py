# src/scanner/visualize.py
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from .screener import daily_last_6m, intraday_last_n_days
from yf_history.reliable_cache import HistoryService

def _normalize_to_1000(series: pd.Series) -> pd.Series:
    """Scale so the series' mean equals $1000 (dimensionless view)."""
    m = float(series.mean()) if len(series) else 1.0
    return (series / (m if m else 1.0)) * 1000.0

class SymbolChart:
    """
    One-method visualizer for High/Low/Close with business-time compression
    and a secondary axis normalized to a $1000 mean scale.
    """

    def __init__(self, history: HistoryService, tz: str = "America/New_York"):
        self.history = history
        self.tz = tz

    def render(
        self,
        symbol: str,
        *,
        last_n: int = 60,
        interval: str = "1d",
        as_of: str | pd.Timestamp | None = None,
        out: str | None = None
    ) -> None:
        """Plot H/L/C with optional intraday interval and past-day window.

        - interval="1d" -> last_n daily sessions (existing behavior)
        - interval in {"1m","5m","15m","30m","60m","1h"} -> last_n business days
          of intraday bars (RTH), aligned to the interval grid.
        If `out` is provided, save PNG; otherwise show the figure.
        """
        iv = ("60m" if interval == "1h" else interval).lower()

        if iv == "1d":
            df = daily_last_6m(self.history, symbol, tz=self.tz, as_of=as_of)
            if df is None or df.empty:
                raise ValueError(f"No data for symbol: {symbol}")
            df = df.tail(int(last_n)).copy()
            x = np.arange(len(df))  # equal spacing per business day
            tick_pos = None
            tick_lbl = None
        else:
            df = intraday_last_n_days(
                self.history,
                symbol,
                int(last_n),
                interval=iv,
                tz=self.tz,
                as_of=as_of,
            )
            if df is None or df.empty:
                raise ValueError(f"No data for symbol: {symbol}")
            df = df.copy()
            x = np.arange(len(df))  # equal spacing per bar

            # Place ticks at session starts (NY local days)
            local_idx = pd.DatetimeIndex(df.index).tz_convert(self.tz)
            local_days = local_idx.normalize()
            day_change_pos = [0]
            for i in range(1, len(local_days)):
                if local_days[i] != local_days[i - 1]:
                    day_change_pos.append(i)
            # Limit tick count for readability (≈6 labels)
            if len(day_change_pos) > 6:
                step = max(1, len(day_change_pos) // 6)
                tick_pos = day_change_pos[::step]
            else:
                tick_pos = day_change_pos
            tick_lbl = []
            for i, pos in enumerate(tick_pos):
                d = local_idx[pos]
                tick_lbl.append(d.strftime('%Y-%m-%d') if i == 0 else d.strftime('%m-%d'))

        fig, ax = plt.subplots()
        ax.plot(x, df['low'].to_numpy(),  label='Low')
        ax.plot(x, df['high'].to_numpy(), label='High')
        ax.plot(x, df['close'].to_numpy(), label='Close', linestyle='--')

        # x-axis ticks
        if tick_pos is None:
            # sparse ticks for daily view
            step = max(1, len(df)//6)
            tick_pos = np.arange(0, len(df), step)
            tick_lbl = [
                df.index[0].strftime('%Y-%m-%d') if i == 0 else df.index[i].strftime('%m-%d')
                for i in tick_pos
            ]
        ax.set_xticks(tick_pos, tick_lbl, rotation=45, ha='right')

        # light grid: y-grid always; x-grid at day boundaries for intraday
        if iv == "1d":
            for i in range(len(df)):
                ax.axvline(i, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
        else:
            for pos in tick_pos:
                ax.axvline(pos, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
        for y in ax.get_yticks():
            ax.axhline(y, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)

        # right axis: normalized-to-1000 scale
        ax2 = ax.twinx()
        norm_close = _normalize_to_1000(df['close'])
        # scale secondary axis to match current left y-limits
        y0, y1 = ax.get_ylim()
        # Map left (raw price) -> right (normalized) by ratio of means
        base_mean = float(df['close'].mean()) if len(df) else 1.0
        scale = float(norm_close.mean() / (base_mean if base_mean else 1.0))
        ax2.set_ylim(y0 * scale, y1 * scale)
        ax2.set_ylabel('Normalized ($1000 mean)')

        ax.set_xlim(-0.5, len(df)-0.5)
        title_span = f"{last_n} sessions" if iv == "1d" else f"last {last_n} business days @ {iv}"
        ax.set_title(f'{symbol} High/Low/Close ({title_span})')
        ax.legend()
        plt.tight_layout()

        if out:
            plt.savefig(out, dpi=120, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()
