"""Stadio 1 della pipeline: dati grezzi -> eventi e snapshot normalizzati in Parquet."""
from __future__ import annotations

import datetime as dt
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from . import storage
from .parsers import (
    fi2010_stock_segments,
    iter_fi2010_snapshots,
    iter_hyperliquid_book_diffs,
    load_fi2010_matrix,
)
from .parsers.itch import SESSION_CLOSE_NS, extract_itch_symbols

ITCH_SYMBOLS = ["AAPL", "MSFT", "INTC"]


def _paths(root: Path) -> tuple[Path, Path, Path]:
    root = Path(root)
    return root / "data" / "raw", root / "data" / "interim", root / "results" / "ingest"


def _write_stats(out: Path, name: str, stats: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")


def _itch_day(args: tuple[str, str, list[str]]) -> dict:
    raw_file, interim, symbols = args
    start = time.perf_counter()
    code = Path(raw_file).name[:8]
    session = dt.datetime.strptime(code, "%m%d%Y").date().isoformat()
    writers = {s: storage.EventWriter(storage.partition(Path(interim), "itch", session, s) / "events.parquet")
               for s in symbols}
    _, stats = extract_itch_symbols(raw_file, symbols, stop_ns=SESSION_CLOSE_NS,
                                    sink=lambda sym, cols: writers[sym].write(cols))
    rows = {s: w.close() for s, w in writers.items()}
    seconds = time.perf_counter() - start
    return {"session": session, **stats, "rows": rows, "seconds": seconds,
            "messages_per_second": stats["messages"] / seconds,
            "coverage_end": str(dt.timedelta(microseconds=stats["last_timestamp_symbols"] // 1000))}


def ingest_itch(root: Path, symbols: list[str] = ITCH_SYMBOLS, workers: int = 7) -> list[dict]:
    raw, interim, out = _paths(root)
    files = sorted((raw / "itch").glob("*.NASDAQ_ITCH50.gz"))
    with ProcessPoolExecutor(workers) as pool:
        results = list(pool.map(_itch_day, [(str(f), str(interim), symbols) for f in files]))
    results.sort(key=lambda r: r["session"])
    _write_stats(out, "itch", {"sessions": results})
    return results


def ingest_fi2010(root: Path) -> list[dict]:
    raw, interim, out = _paths(root)
    results = []
    for path in sorted((raw / "fi2010").glob("fi2010_day*.txt")):
        start = time.perf_counter()
        matrix = load_fi2010_matrix(path)
        session = path.stem.replace("fi2010_", "")
        segments = fi2010_stock_segments(matrix)
        for number, (begin, stop) in enumerate(segments, start=1):
            symbol = f"stock{number}"
            rows = storage.write_snapshots(iter_fi2010_snapshots(matrix, symbol, start=begin, stop=stop),
                                           storage.partition(interim, "fi2010", session, symbol)
                                           / "snapshots.parquet")
            results.append({"session": session, "symbol": symbol, "rows": rows})
        results.append({"session": session, "symbol": "*", "segments": len(segments),
                        "seconds": time.perf_counter() - start})
    _write_stats(out, "fi2010", {"sessions": results})
    return results


def _hl_hour(args: tuple[str, str, str, str]) -> dict:
    diffs, trades, interim, session = args
    start = time.perf_counter()
    stats: dict = {}
    rows = storage.write_events(storage.events_to_columns(iter_hyperliquid_book_diffs(diffs, trades, "BTC", stats)),
                                storage.partition(Path(interim), "hyperliquid", session, "BTC") / "events.parquet")
    return {"session": session, "rows": rows, **stats, "seconds": time.perf_counter() - start}


def ingest_hyperliquid(root: Path, workers: int = 6) -> list[dict]:
    raw, interim, out = _paths(root)
    jobs = []
    for diffs in sorted((raw / "hyperliquid").glob("book_diffs_*.gz")):
        day, hour = diffs.stem.replace(".gz", "").split("_")[-2:]
        trades = raw / "hyperliquid" / f"trades_{day}_{hour}.gz"
        jobs.append((str(diffs), str(trades), str(interim), f"{day}T{hour}"))
    with ProcessPoolExecutor(workers) as pool:
        results = sorted(pool.map(_hl_hour, jobs), key=lambda r: r["session"])
    _write_stats(out, "hyperliquid", {"sessions": results})
    return results


def ingest_all(root: Path, datasets=("fi2010", "itch", "hyperliquid")) -> None:
    steps = {"fi2010": ingest_fi2010, "itch": ingest_itch, "hyperliquid": ingest_hyperliquid}
    for name in datasets:
        print(f"ingest {name}", flush=True)
        steps[name](Path(root))
