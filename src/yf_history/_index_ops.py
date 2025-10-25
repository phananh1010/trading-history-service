from __future__ import annotations
import pandas as pd


def ensure_schema(df: pd.DataFrame, cols: tuple[str, ...]) -> pd.DataFrame:
    if df is None or df.empty:
        out = pd.DataFrame(columns=cols)
        out.index.name = "datetime"
        return out
    df = df.copy()
    # standardize columns
    df.columns = [c.lower() for c in df.columns]
    keep = {c for c in df.columns if c in cols}
    for c in cols:
        if c not in keep:
            df[c] = 0.0 if c != "volume" else 0
    df = df[list(cols)]
    # index to DatetimeIndex
    df.index = pd.to_datetime(df.index)
    df.index.name = "datetime"
    return df


def to_naive_tz(df: pd.DataFrame, tz: str, interval: str) -> pd.DataFrame:
    if df.empty:
        return df
    idx = pd.to_datetime(df.index)
    # Daily stays as session dates (naive). Intraday is UTC->tz then naive.
    if interval == "1d":
        df.index = idx.tz_localize(None)
    else:
        # Assume current index either tz‐aware or UTC; localize/convert to tz then drop tz
        idx = idx.tz_localize("UTC") if idx.tz is None else idx
        df.index = idx.tz_convert(tz).tz_localize(None)
    return df


def slice_end(df: pd.DataFrame, start, end, *, end_inclusive: bool) -> pd.DataFrame:
    if df.empty:
        return df
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    if end_inclusive:
        return df.loc[(df.index >= s) & (df.index <= e)]
    else:
        return df.loc[(df.index >= s) & (df.index < e)]


def merge_by_index(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    if a is None or a.empty:
        return b
    if b is None or b.empty:
        return a
    m = pd.concat([a, b]).sort_index().groupby(level=0).last()
    return m


def _to_utc(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tz is None else t.tz_convert("UTC")


def _utc_index(df: pd.DataFrame) -> pd.DatetimeIndex:
    idx = pd.to_datetime(df.index)
    return idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")


def align_on_grid(
    a: pd.DataFrame,
    b: pd.DataFrame,
    start=None,
    end=None,
    *,
    freq: str = "1min",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reindex two frames onto a common UTC time grid.

    - Leaves gaps as NaN (no synthetic bars).
    - `freq` uses pandas offset aliases (e.g., "1min", "5min", "60min", "1D").
    - If `start`/`end` are None, uses min/max across provided indices.
    """
    if (a is None or a.empty) and (b is None or b.empty):
        if start is None or end is None:
            # Nothing to align, return as-is
            return a, b
        s = _to_utc(start)
        e = _to_utc(end)
        full = pd.date_range(s, e, freq=freq, tz="UTC")
        return a.reindex(full), b.reindex(full)

    a_idx = _utc_index(a) if a is not None and not a.empty else None
    b_idx = _utc_index(b) if b is not None and not b.empty else None

    s = _to_utc(start) if start is not None else (
        (a_idx.min() if a_idx is not None else b_idx.min())
        if b_idx is not None else a_idx.min()
    )
    e = _to_utc(end) if end is not None else (
        (a_idx.max() if a_idx is not None else b_idx.max())
        if b_idx is not None else a_idx.max()
    )

    full = pd.date_range(s, e, freq=freq, tz="UTC")
    a2 = a.copy()
    b2 = b.copy()
    a2.index = _utc_index(a2)
    b2.index = _utc_index(b2)
    return a2.reindex(full), b2.reindex(full)


def align_many_on_grid(
    frames: dict[str, pd.DataFrame] | list[pd.DataFrame],
    start=None,
    end=None,
    *,
    freq: str = "1min",
) -> dict[str, pd.DataFrame] | list[pd.DataFrame]:
    """Reindex N frames onto a common UTC time grid.

    Returns the same container type (dict or list) with reindexed frames.
    """
    # Collect bounds from all non-empty frames if start/end not given
    non_empty_idxs: list[pd.DatetimeIndex] = []
    if isinstance(frames, dict):
        items = frames.items()
    else:
        items = enumerate(frames)

    for _, df in items:
        if df is not None and not df.empty:
            non_empty_idxs.append(_utc_index(df))

    if start is None and non_empty_idxs:
        start = min(idx.min() for idx in non_empty_idxs)
    if end is None and non_empty_idxs:
        end = max(idx.max() for idx in non_empty_idxs)

    if start is None or end is None:
        # Nothing to align use passthrough
        return frames

    s = _to_utc(start)
    e = _to_utc(end)
    full = pd.date_range(s, e, freq=freq, tz="UTC")

    def reindex_df(df: pd.DataFrame) -> pd.DataFrame:
        if df is None:
            return pd.DataFrame().reindex(full)
        if df.empty:
            return df.reindex(full)
        out = df.copy()
        out.index = _utc_index(out)
        return out.reindex(full)

    if isinstance(frames, dict):
        return {k: reindex_df(v) for k, v in frames.items()}
    else:
        return [reindex_df(v) for v in frames]

