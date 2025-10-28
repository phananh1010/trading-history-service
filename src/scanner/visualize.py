# src/scanner/visualize.py
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from .screener import daily_last_6m
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

    def render(self, symbol: str, *, last_n: int = 60, out: str | None = None) -> None:
        """Fetch last 6m daily bars, plot H/L/C, add normalized right axis.
        If `out` is provided, save PNG; otherwise show the figure.
        """
        df = daily_last_6m(self.history, symbol, tz=self.tz)
        if df is None or df.empty:
            raise ValueError(f"No data for symbol: {symbol}")
        df = df.tail(int(last_n)).copy()

        x = np.arange(len(df))  # equal spacing per business day

        fig, ax = plt.subplots()
        ax.plot(x, df['low'].to_numpy(),  label='Low')
        ax.plot(x, df['high'].to_numpy(), label='High')
        ax.plot(x, df['close'].to_numpy(), label='Close', linestyle='--')

        # sparse date ticks
        step = max(1, len(df)//6)
        tick_pos = np.arange(0, len(df), step)
        tick_lbl = [
            df.index[0].strftime('%Y-%m-%d') if i == 0 else df.index[i].strftime('%m-%d')
            for i in tick_pos
        ]
        ax.set_xticks(tick_pos, tick_lbl, rotation=45, ha='right')

        # light grid
        for i in range(len(df)):
            ax.axvline(i, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
        for y in ax.get_yticks():
            ax.axhline(y, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)

        # right axis: normalized-to-1000 scale
        ax2 = ax.twinx()
        norm_close = _normalize_to_1000(df['close'])
        # scale secondary axis to match current left y-limits
        y0, y1 = ax.get_ylim()
        # Map left (raw price) -> right (normalized) by ratio of means
        scale = float(norm_close.mean() / (df['close'].mean() if df['close'].mean() else 1.0))
        ax2.set_ylim(y0 * scale, y1 * scale)
        ax2.set_ylabel('Normalized ($1000 mean)')

        ax.set_xlim(-0.5, len(df)-0.5)
        ax.set_title(f'{symbol} High/Low/Close ({last_n} sessions)')
        ax.legend()
        plt.tight_layout()

        if out:
            plt.savefig(out, dpi=120, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()
