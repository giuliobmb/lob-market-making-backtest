"""Dati intermedi in Parquet partizionato (stile Hive).

Struttura: ``data/interim/<dataset>/session=<id>/symbol=<simbolo>/<kind>.parquet``
con ``kind`` = ``events`` (flusso order-level normalizzato) oppure ``snapshots``
(book L2 a 10 livelli). Ogni sessione e un'unita indipendente: giornata,
titolo o ora, mai mescolate nello stesso replay.
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .models import BookSnapshot, EventType, OrderEvent, Side

EVENT_SCHEMA = pa.schema([
    ("timestamp", pa.int64()), ("event_index", pa.int64()), ("event_type", pa.string()),
    ("order_id", pa.string()), ("side", pa.string()), ("price", pa.float64()),
    ("size", pa.float64()), ("new_order_id", pa.string()), ("new_price", pa.float64()),
    ("new_size", pa.float64()), ("bootstrap", pa.bool_()),
])
LEVELS = 10


def partition(root: Path, dataset: str, session: str, symbol: str) -> Path:
    return Path(root) / dataset / f"session={session}" / f"symbol={symbol}"


def sessions(root: Path, dataset: str, kind: str = "events") -> list[tuple[str, str, Path]]:
    """Elenco ordinato (sessione, simbolo, file) presenti per un dataset."""
    out = []
    for path in sorted((Path(root) / dataset).glob(f"session=*/symbol=*/{kind}.parquet")):
        out.append((path.parent.parent.name.split("=", 1)[1], path.parent.name.split("=", 1)[1], path))
    return out


def write_events(columns: dict[str, list], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table({name: columns[name] for name in EVENT_SCHEMA.names}, schema=EVENT_SCHEMA)
    pq.write_table(table, path, compression="zstd")
    return table.num_rows


class EventWriter:
    """Scrittura a blocchi di un file di eventi (giornate complete senza tenerle in memoria)."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.writer = pq.ParquetWriter(self.path, EVENT_SCHEMA, compression="zstd")
        self.rows = 0

    def write(self, columns: dict[str, list]) -> None:
        table = pa.table({name: columns[name] for name in EVENT_SCHEMA.names}, schema=EVENT_SCHEMA)
        self.writer.write_table(table)
        self.rows += table.num_rows

    def close(self) -> int:
        self.writer.close()
        return self.rows


def events_to_columns(events: Iterable[OrderEvent]) -> dict[str, list]:
    cols: dict[str, list] = {name: [] for name in EVENT_SCHEMA.names}
    for e in events:
        cols["timestamp"].append(e.timestamp)
        cols["event_index"].append(e.event_index)
        cols["event_type"].append(e.event_type.value)
        cols["order_id"].append(str(e.order_id))
        cols["side"].append(None if e.side is None else e.side.value)
        cols["price"].append(e.price)
        cols["size"].append(e.size)
        cols["new_order_id"].append(e.new_order_id)
        cols["new_price"].append(e.new_price)
        cols["new_size"].append(e.new_size)
        cols["bootstrap"].append(bool(e.metadata.get("bootstrap", False)))
    return cols


def read_events(path: Path, symbol: str = "", stop_timestamp: int | None = None,
                batch_size: int = 200_000) -> Iterator[OrderEvent]:
    """Eventi in ordine di file, letti a blocchi per contenere la memoria.

    ``stop_timestamp`` interrompe la lettura al primo evento con timestamp
    maggiore o uguale (fine della finestra di sessione).
    """
    types = {t.value: t for t in EventType}
    sides = {"BUY": Side.BUY, "SELL": Side.SELL, None: None}
    bootstrap, normal = {"bootstrap": True}, {}
    for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size):
        cols = {name: batch.column(name).to_pylist() for name in EVENT_SCHEMA.names}
        for i in range(batch.num_rows):
            ts = cols["timestamp"][i]
            if stop_timestamp is not None and ts >= stop_timestamp:
                return
            yield OrderEvent(ts, cols["event_index"][i], symbol, types[cols["event_type"][i]],
                             cols["order_id"][i], sides[cols["side"][i]], cols["price"][i],
                             cols["size"][i], cols["new_order_id"][i], cols["new_price"][i],
                             cols["new_size"][i], metadata=bootstrap if cols["bootstrap"][i] else normal)


def write_snapshots(snapshots: Iterable[BookSnapshot], path: Path) -> int:
    rows = {"timestamp": [], "event_index": []}
    for side in ("bid", "ask"):
        for level in range(1, LEVELS + 1):
            rows[f"{side}_px_{level}"] = []
            rows[f"{side}_sz_{level}"] = []
    for s in snapshots:
        rows["timestamp"].append(s.timestamp)
        rows["event_index"].append(s.event_index)
        for side, prices, sizes in (("bid", s.bid_prices, s.bid_sizes), ("ask", s.ask_prices, s.ask_sizes)):
            for level in range(LEVELS):
                rows[f"{side}_px_{level + 1}"].append(float(prices[level]) if level < len(prices) else None)
                rows[f"{side}_sz_{level + 1}"].append(float(sizes[level]) if level < len(sizes) else None)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table({k: pa.array(v, type=pa.int64() if k in ("timestamp", "event_index") else pa.float64())
                      for k, v in rows.items()})
    pq.write_table(table, path, compression="zstd")
    return table.num_rows


def read_snapshots(path: Path, symbol: str = "") -> Iterator[BookSnapshot]:
    table = pq.read_table(path)
    ts = table.column("timestamp").to_numpy()
    idx = table.column("event_index").to_numpy()
    data = {side: (np.column_stack([table.column(f"{side}_px_{l}").to_numpy(zero_copy_only=False)
                                    for l in range(1, LEVELS + 1)]),
                   np.column_stack([table.column(f"{side}_sz_{l}").to_numpy(zero_copy_only=False)
                                    for l in range(1, LEVELS + 1)]))
            for side in ("bid", "ask")}
    for i in range(table.num_rows):
        bp, bs = data["bid"][0][i], data["bid"][1][i]
        ap, as_ = data["ask"][0][i], data["ask"][1][i]
        nb = int(np.sum(~np.isnan(bp)))
        na = int(np.sum(~np.isnan(ap)))
        yield BookSnapshot(int(ts[i]), int(idx[i]), symbol, tuple(bp[:nb].tolist()), tuple(bs[:nb].tolist()),
                           tuple(ap[:na].tolist()), tuple(as_[:na].tolist()))
