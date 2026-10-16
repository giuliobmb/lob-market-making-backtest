"""Controlli di qualita sui flussi normalizzati (timestamp, quantita, Order ID, book incrociato)."""
from __future__ import annotations

from collections.abc import Iterable

from .models import BookSnapshot, EventType, OrderEvent, Side
from .orderbook import MarketByOrderBook


def check_events(events: Iterable[OrderEvent]) -> dict[str, float]:
    """Riproduce il flusso nel book e conta le anomalie.

    * ``ts_decreasing``: eventi con timestamp minore del precedente;
    * ``nonpositive_size``: ADD/EXECUTE/CANCEL con quantita <= 0;
    * ``unknown_order``: riduzioni su Order ID mai osservati (o gia chiusi);
    * ``duplicate_add``: ADD su un Order ID gia vivo;
    * ``overfill``: esecuzioni/cancellazioni maggiori del residuo;
    * ``crossed_share``: quota di record con best bid >= best ask.
    """
    book = MarketByOrderBook()
    c = {"events": 0, "bootstrap_events": 0, "ts_decreasing": 0, "nonpositive_size": 0,
         "unknown_order": 0, "duplicate_add": 0, "overfill": 0, "records": 0, "crossed": 0}
    last_ts = None
    for e in events:
        c["events"] += 1
        c["bootstrap_events"] += bool(e.metadata.get("bootstrap"))
        if last_ts is not None and e.timestamp < last_ts:
            c["ts_decreasing"] += 1
        last_ts = e.timestamp
        current = book.get(e.order_id)
        if e.event_type is EventType.ADD:
            c["nonpositive_size"] += e.size is None or e.size <= 0
            c["duplicate_add"] += current is not None
        elif e.order_id != "NOOP":
            if current is None:
                c["unknown_order"] += 1
            elif e.event_type in {EventType.EXECUTE, EventType.CANCEL} and e.size is not None:
                c["nonpositive_size"] += e.size <= 0
                c["overfill"] += e.size > current.remaining + 1e-9
        book.apply(e)
        bid, ask = book.best(Side.BUY), book.best(Side.SELL)
        if bid is not None and ask is not None and not e.metadata.get("bootstrap"):
            c["records"] += 1
            c["crossed"] += bid >= ask
    c["crossed_share"] = c["crossed"] / c["records"] if c["records"] else 0.0
    return c


def check_snapshots(snapshots: Iterable[BookSnapshot]) -> dict[str, float]:
    c = {"records": 0, "ts_decreasing": 0, "crossed": 0, "nonpositive_size": 0, "nonfinite_mid": 0}
    last_ts = None
    for s in snapshots:
        c["records"] += 1
        if last_ts is not None and s.timestamp < last_ts:
            c["ts_decreasing"] += 1
        last_ts = s.timestamp
        c["crossed"] += s.best_bid >= s.best_ask
        c["nonpositive_size"] += any(v <= 0 for v in (*s.bid_sizes, *s.ask_sizes))
        c["nonfinite_mid"] += not (s.midpoint == s.midpoint)
    c["crossed_share"] = c["crossed"] / c["records"] if c["records"] else 0.0
    return c
