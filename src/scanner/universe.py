# src/yf_history/universe.py
from __future__ import annotations
import requests
from tqdm import tqdm

def fetch_nasdaq_universe(limit: int = 5000) -> list[dict]:
    headers = {'User-Agent': 'Mozilla/5.0'}
    tickers, offset = [], 0
    while True:
        url = f"https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit={limit}&offset={offset}"
        data = requests.get(url, headers=headers).json()
        rows = data['data']['table']['rows']
        if not rows: break
        tickers += rows
        offset += limit
        if offset >= int(data['data']['totalrecords']): break
        print (f"fetching progress: {offset}/{data['data']['totalrecords']} tickers")
    # normalize a few fields
    def _mcap(x: str | None) -> float:
        # Robustly parse market cap strings like "$1.2B", "750M", "NA", "N/A", "-", etc.
        if x is None:
            return 0.0
        raw = str(x).strip()
        if not raw:
            return 0.0
        upper = raw.upper()
        if upper in {"NA", "N/A", "NONE", "NULL", "-", "—", "NAN"}:
            return 0.0
        s = upper.replace("$", "").replace(",", "").strip()
        mult = {"T": 1e12, "B": 1e9, "M": 1e6, "K": 1e3}
        try:
            if s and s[-1] in mult:
                return float(s[:-1]) * mult[s[-1]]
            return float(s)
        except (ValueError, TypeError):
            return 0.0
    for t in tickers:
        t['marketCap'] = _mcap(t.get('marketCap'))
    tickers.sort(key=lambda d: d['marketCap'], reverse=True)
    return tickers
