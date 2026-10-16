from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

from market_making_tesi.parsers import (
    iter_fi2010_snapshots,
    iter_hyperliquid_jsonl,
    iter_itch50_gzip,
)
from market_making_tesi.orderbook import MarketByOrderBook


ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "data" / "samples"


def _common(kind: bytes, timestamp: int, tracking: int) -> bytes:
    return b"".join(
        [kind, (1).to_bytes(2, "big"), tracking.to_bytes(2, "big"), timestamp.to_bytes(6, "big")]
    )


def _add(timestamp: int, tracking: int, oid: int, side: str, shares: int, price: float) -> bytes:
    return b"".join(
        [
            _common(b"A", timestamp, tracking),
            oid.to_bytes(8, "big"),
            side.encode("ascii"),
            shares.to_bytes(4, "big"),
            b"AAPL    ",
            round(price * 10_000).to_bytes(4, "big"),
        ]
    )


def _execute(timestamp: int, tracking: int, oid: int, shares: int, match: int) -> bytes:
    return b"".join(
        [
            _common(b"E", timestamp, tracking),
            oid.to_bytes(8, "big"),
            shares.to_bytes(4, "big"),
            match.to_bytes(8, "big"),
        ]
    )


def create_itch_sample() -> None:
    definitions = [
        ("ADD", 1, 1001, "B", 5, 100.00),
        ("ADD", 2, 2001, "S", 5, 100.01),
        ("ADD", 3, 1002, "B", 5, 99.99),
        ("ADD", 4, 2002, "S", 5, 100.02),
        ("EXECUTE", 5, 1001, "", 5, 100.00),
        ("EXECUTE", 6, 2001, "", 5, 100.01),
        ("ADD", 7, 1003, "B", 5, 100.00),
        ("ADD", 8, 2003, "S", 5, 100.01),
        ("EXECUTE", 9, 1003, "", 5, 100.00),
        ("EXECUTE", 10, 2003, "", 5, 100.01),
    ]
    messages: list[bytes] = []
    for kind, index, oid, side, shares, price in definitions:
        timestamp = 34_200_000_000_000 + index * 1_000_000
        if kind == "ADD":
            messages.append(_add(timestamp, index, oid, side, shares, price))
        else:
            messages.append(_execute(timestamp, index, oid, shares, 9000 + index))

    output = SAMPLES / "itch_aapl_synthetic_10.gz"
    with gzip.open(output, "wb") as stream:
        for message in messages:
            stream.write(len(message).to_bytes(2, "big"))
            stream.write(message)

    with (SAMPLES / "itch_aapl_synthetic_10_decoded.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(["event_index", "event_type", "order_id", "side", "size", "price"])
        for kind, index, oid, side, shares, price in definitions:
            writer.writerow([index, kind, oid, side, shares, f"{price:.2f}"])


def create_hyperliquid_queue_sample() -> None:
    rows = [
        {"timestamp": 1, "coin": "BTC", "type": "open", "oid": 1001, "side": "B", "px": "100.0", "sz": "5"},
        {"timestamp": 2, "coin": "BTC", "type": "open", "oid": 2001, "side": "A", "px": "100.1", "sz": "5"},
        {"timestamp": 3, "coin": "BTC", "type": "open", "oid": 1002, "side": "B", "px": "99.9", "sz": "5"},
        {"timestamp": 4, "coin": "BTC", "type": "open", "oid": 2002, "side": "A", "px": "100.2", "sz": "5"},
        {"timestamp": 5, "coin": "BTC", "type": "fill", "oid": 1001, "sz": "5"},
        {"timestamp": 6, "coin": "BTC", "type": "fill", "oid": 2001, "sz": "5"},
        {"timestamp": 7, "coin": "BTC", "type": "open", "oid": 1003, "side": "B", "px": "100.0", "sz": "5"},
        {"timestamp": 8, "coin": "BTC", "type": "open", "oid": 2003, "side": "A", "px": "100.1", "sz": "5"},
        {"timestamp": 9, "coin": "BTC", "type": "fill", "oid": 1003, "sz": "5"},
        {"timestamp": 10, "coin": "BTC", "type": "fill", "oid": 2003, "sz": "5"},
    ]
    with (SAMPLES / "hyperliquid_queue_synthetic_10.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")


def create_snapshot_preview() -> None:
    rows: list[dict[str, object]] = []

    fi_snapshot = next(iter_fi2010_snapshots(SAMPLES / "fi2010_real_60.txt"))
    rows.append(_snapshot_row("FI-2010 reale", fi_snapshot))

    itch_book = MarketByOrderBook()
    for event in iter_itch50_gzip(SAMPLES / "itch_aapl_synthetic_10.gz", symbol="AAPL"):
        itch_book.apply(event)
        snapshot = itch_book.snapshot(event.timestamp, event.event_index)
        if snapshot is not None:
            rows.append(_snapshot_row("ITCH sintetico", snapshot))
            break

    hyper_book = MarketByOrderBook()
    for event in iter_hyperliquid_jsonl(
        SAMPLES / "hyperliquid_book_diffs_real_500.jsonl", symbol="BTC"
    ):
        hyper_book.apply(event)
        snapshot = hyper_book.snapshot(event.timestamp, event.event_index)
        if snapshot is not None:
            rows.append(_snapshot_row("Hyperliquid reale", snapshot))
            break

    with (SAMPLES / "snapshots_preview.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _snapshot_row(dataset: str, snapshot: object) -> dict[str, object]:
    return {
        "dataset": dataset,
        "event_index": snapshot.event_index,
        "best_bid": snapshot.best_bid,
        "bid_size": snapshot.bid_sizes[0],
        "best_ask": snapshot.best_ask,
        "ask_size": snapshot.ask_sizes[0],
        "midpoint": snapshot.midpoint,
        "spread": snapshot.spread,
    }


def main() -> None:
    SAMPLES.mkdir(parents=True, exist_ok=True)
    create_itch_sample()
    create_hyperliquid_queue_sample()
    create_snapshot_preview()
    print(f"Campioni creati in {SAMPLES}")


if __name__ == "__main__":
    main()
