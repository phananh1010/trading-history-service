# src/yf_history/universe.py
from __future__ import annotations
import json, os, time
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import requests

def _parse_mcap(x: str | None) -> float:
    if x is None:
        return 0.0
    raw = str(x).strip()
    if not raw:
        return 0.0
    u = raw.upper().replace("$", "").replace(",", "").strip()
    if u in {"NA", "N/A", "NONE", "NULL", "-", "—", "NAN"}:
        return 0.0
    mult = {"T": 1e12, "B": 1e9, "M": 1e6, "K": 1e3}
    try:
        if u and u[-1] in mult:
            return float(u[:-1]) * mult[u[-1]]
        return float(u)
    except (ValueError, TypeError):
        return 0.0

def _today_str_ny() -> str:
    return datetime.now(ZoneInfo("America/New_York")).date().isoformat()

def _normalize_rows(rows: list[dict]) -> list[dict]:
    for t in rows:
        t["marketCap"] = _parse_mcap(t.get("marketCap"))
    rows.sort(key=lambda d: d.get("marketCap", 0.0), reverse=True)
    return rows

def fetch_nasdaq_universe(limit: int = 5000,
                          cache_dir: str | os.PathLike = "~/.cache/stock_universe",
                          force_refresh: bool = False) -> list[dict]:
    """
    Fetch NASDAQ screener and cache **per NY trading day**.
    If today's cache exists (and not forcing), load from cache instead of calling the API.
    """
    # Expand user home ("~") so cache writes to ~/.cache as intended
    cache_dir = Path(cache_dir).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)
    stamp = _today_str_ny()
    cache_path = cache_dir / f"nasdaq_universe_{stamp}.json"

    if cache_path.exists() and not force_refresh:
        with cache_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json, text/plain, */*",
        "Referer": "https://www.nasdaq.com/",
    }

    rows, offset = [], 0
    total = None

    # simple retry loop to be resilient
    def _get(url: str, tries: int = 3, backoff: float = 0.8):
        last = None
        for i in range(tries):
            try:
                r = requests.get(url, headers=headers, timeout=10)
                r.raise_for_status()
                return r.json()
            except Exception as e:
                last = e
                time.sleep(backoff * (2 ** i))
        raise last

    while True:
        url = f"https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit={limit}&offset={offset}"
        data = _get(url)
        # defensive parsing
        d = data.get("data") or {}
        t = (d.get("table") or {}).get("rows") or []
        if total is None:
            total = int(d.get("totalrecords") or 0)
        if not t:
            break
        rows += t
        offset += limit
        if offset >= total:
            break

    normalized = _normalize_rows(rows)

    # atomic write
    tmp = cache_path.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(normalized, f, ensure_ascii=False)
    tmp.replace(cache_path)

    return normalized
