"""Adattatori che convertono i dataset nel formato comune del motore."""

from .fi2010 import fi2010_stock_segments, iter_fi2010_snapshots, load_fi2010_labels, load_fi2010_matrix
from .hyperliquid import hyperliquid_maker_fills, iter_hyperliquid_book_diffs, iter_hyperliquid_jsonl
from .itch import iter_itch50_gzip

__all__ = [
    "fi2010_stock_segments",
    "hyperliquid_maker_fills",
    "iter_fi2010_snapshots",
    "iter_hyperliquid_book_diffs",
    "iter_hyperliquid_jsonl",
    "iter_itch50_gzip",
    "load_fi2010_labels",
    "load_fi2010_matrix",
]
