from __future__ import annotations

import gzip
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from market_making_tesi.parsers.fi2010 import iter_fi2010_snapshots
from market_making_tesi.parsers.hyperliquid import iter_hyperliquid_jsonl
from market_making_tesi.parsers.itch import iter_itch50_gzip
from market_making_tesi.models import EventType, Side


def six_byte_integer(value: int) -> bytes:
    return value.to_bytes(6, "big")


def add_message(order_id: int, side: str, shares: int, stock: str, price: int) -> bytes:
    return b"".join(
        [
            b"A",
            (1).to_bytes(2, "big"),
            (1).to_bytes(2, "big"),
            six_byte_integer(100),
            order_id.to_bytes(8, "big"),
            side.encode("ascii"),
            shares.to_bytes(4, "big"),
            stock.ljust(8).encode("ascii"),
            price.to_bytes(4, "big"),
        ]
    )


def execute_message(order_id: int, shares: int) -> bytes:
    return b"".join(
        [
            b"E",
            (1).to_bytes(2, "big"),
            (2).to_bytes(2, "big"),
            six_byte_integer(200),
            order_id.to_bytes(8, "big"),
            shares.to_bytes(4, "big"),
            (99).to_bytes(8, "big"),
        ]
    )


class AdapterTests(unittest.TestCase):
    def test_fi2010_layout(self) -> None:
        matrix = np.zeros((149, 2))
        for level in range(10):
            matrix[level * 4 + 0, :] = 101 + level
            matrix[level * 4 + 1, :] = 10 + level
            matrix[level * 4 + 2, :] = 100 - level
            matrix[level * 4 + 3, :] = 20 + level
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fi.txt"
            np.savetxt(path, matrix)
            snapshots = list(iter_fi2010_snapshots(path))
        self.assertEqual(len(snapshots), 2)
        self.assertEqual(snapshots[0].best_bid, 100)
        self.assertEqual(snapshots[0].best_ask, 101)
        self.assertEqual(len(snapshots[0].bid_prices), 10)

    def test_itch_add_and_execute(self) -> None:
        messages = [add_message(42, "B", 100, "AAPL", 1_234_500), execute_message(42, 40)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.gz"
            with gzip.open(path, "wb") as stream:
                for message in messages:
                    stream.write(len(message).to_bytes(2, "big"))
                    stream.write(message)
            events = list(iter_itch50_gzip(path, symbol="AAPL"))
        self.assertEqual([event.event_type for event in events], [EventType.ADD, EventType.EXECUTE])
        self.assertEqual(events[0].side, Side.BUY)
        self.assertAlmostEqual(events[0].price or 0, 123.45)
        self.assertEqual(events[1].size, 40)

    def test_hyperliquid_aliases(self) -> None:
        rows = [
            {"time": 1, "coin": "BTC", "type": "open", "oid": 7, "side": "B", "px": "100", "sz": "2"},
            {"time": 2, "coin": "BTC", "type": "fill", "oid": 7, "sz": "1"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            events = list(iter_hyperliquid_jsonl(path, symbol="BTC"))
        self.assertEqual(events[0].event_type, EventType.ADD)
        self.assertEqual(events[0].side, Side.BUY)
        self.assertEqual(events[1].event_type, EventType.EXECUTE)

    def test_hyperliquid_public_book_diff_schema(self) -> None:
        rows = [
            {
                "user": "0x1",
                "oid": 11,
                "coin": "BTC",
                "side": "B",
                "px": "86373.0",
                "raw_book_diff": {"new": {"sz": "0.17363"}},
            },
            {
                "user": "0x1",
                "oid": 11,
                "coin": "BTC",
                "side": "B",
                "px": "86373.0",
                "raw_book_diff": {"update": {"origSz": "0.17363", "newSz": "0.1"}},
            },
            {
                "user": "0x1",
                "oid": 11,
                "coin": "BTC",
                "side": "B",
                "px": "86373.0",
                "raw_book_diff": "remove",
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "book_diffs.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            events = list(iter_hyperliquid_jsonl(path, symbol="BTC"))
        self.assertEqual(
            [event.event_type for event in events],
            [EventType.ADD, EventType.UPDATE, EventType.DELETE],
        )
        self.assertAlmostEqual(events[0].size or 0, 0.17363)
        self.assertAlmostEqual(events[1].size or 0, 0.1)


if __name__ == "__main__":
    unittest.main()
