from __future__ import annotations

import gzip
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from ..models import EventType, OrderEvent, Side


def _first(record: dict[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in record and record[name] is not None:
            return record[name]
    return default


def _side(record: dict[str, Any]) -> Side | None:
    value = _first(record, "side", "dir")
    if value is not None:
        text = str(value).upper()
        if text in {"A", "ASK", "SELL", "S"}:
            return Side.SELL
        if text in {"B", "BID", "BUY"}:
            return Side.BUY
    if "isAsk" in record:
        return Side.SELL if bool(record["isAsk"]) else Side.BUY
    return None


def _event_type(record: dict[str, Any]) -> EventType:
    raw_diff = record.get("raw_book_diff")
    if raw_diff == "remove":
        return EventType.DELETE
    if isinstance(raw_diff, dict):
        if "new" in raw_diff:
            return EventType.ADD
        if "update" in raw_diff:
            return EventType.UPDATE

    raw = str(
        _first(record, "event_type", "type", "diff_type", "status", default="update")
    ).upper()
    aliases = {
        "NEW": EventType.ADD,
        "OPEN": EventType.ADD,
        "ADD": EventType.ADD,
        "UPDATE": EventType.UPDATE,
        "MODIFY": EventType.UPDATE,
        "FILL": EventType.EXECUTE,
        "FILLED": EventType.EXECUTE,
        "EXECUTE": EventType.EXECUTE,
        "TRADE": EventType.TRADE,
        "CANCEL": EventType.CANCEL,
        "CANCELED": EventType.CANCEL,
        "CANCELLED": EventType.CANCEL,
        "REMOVE": EventType.DELETE,
        "DELETE": EventType.DELETE,
    }
    if raw not in aliases:
        raise ValueError(f"Tipo evento Hyperliquid non riconosciuto: {raw!r}")
    return aliases[raw]


def _records(value: Any) -> Iterator[dict[str, Any]]:
    """Espande righe JSON che contengono un evento o una lista di eventi."""

    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                yield item
        return
    if not isinstance(value, dict):
        return
    for key in ("events", "diffs", "orders", "data"):
        nested = value.get(key)
        if isinstance(nested, list):
            inherited = {
                name: value[name]
                for name in ("time", "timestamp", "ts", "coin", "symbol")
                if name in value
            }
            for item in nested:
                if isinstance(item, dict):
                    yield {**inherited, **item}
            return
    yield value


def iter_hyperliquid_jsonl(
    path: str | Path,
    symbol: str | None = None,
    max_events: int | None = None,
) -> Iterator[OrderEvent]:
    """Normalizza un JSONL di eventi order-level Hyperliquid.

    Sono accettati nomi alternativi comuni (``oid``/``order_id``,
    ``px``/``price``, ``sz``/``size``). Un record ``remove`` viene trattato
    come cancellazione/eliminazione e non come fill: per distinguere le cause
    occorre unire book diffs, order status e trades prima del backtest.
    """

    wanted = symbol.upper() if symbol else None
    emitted = 0
    sequence = 0

    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSON non valido alla riga {line_number}") from exc

            for record in _records(payload):
                sequence += 1
                coin = str(_first(record, "coin", "symbol", "asset", default="HYPERLIQUID"))
                if wanted is not None and coin.upper() != wanted:
                    continue

                order_id = _first(record, "oid", "order_id", "orderId", "id")
                if order_id is None:
                    raise ValueError(f"order id assente alla riga {line_number}")
                timestamp = int(float(_first(record, "time", "timestamp", "ts", default=sequence)))
                event_type = _event_type(record)
                price = _first(record, "px", "price", "limit_px", "limitPrice")
                size = _first(record, "sz", "size", "remaining_size", "remainingSize")
                raw_diff = record.get("raw_book_diff")
                if isinstance(raw_diff, dict) and "new" in raw_diff:
                    size = _first(raw_diff["new"], "sz", "size")
                elif isinstance(raw_diff, dict) and "update" in raw_diff:
                    size = _first(raw_diff["update"], "newSz", "new_size", "sz")

                yield OrderEvent(
                    timestamp=timestamp,
                    event_index=sequence,
                    symbol=coin,
                    event_type=event_type,
                    order_id=str(order_id),
                    side=_side(record),
                    price=float(price) if price is not None else None,
                    size=float(size) if size is not None else None,
                    user_id=str(_first(record, "user", "user_id", "address", default="")) or None,
                    metadata={
                        "source_line": line_number,
                        "raw_book_diff": raw_diff,
                    },
                )
                emitted += 1
                if max_events is not None and emitted >= max_events:
                    return


def _open_text(path: str | Path):
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def hyperliquid_maker_fills(trades_path: str | Path, coin: str) -> dict[str, float]:
    """Volume eseguito per ordine maker, dai trade pubblici Hyperliquid.

    Nel file trade ``side`` e il lato dell'aggressore e ``side_info`` elenca
    [acquirente, venditore]: il maker e quindi il venditore se ``side == "B"``
    e l'acquirente se ``side == "A"``.
    """

    wanted = coin.upper()
    fills: dict[str, float] = {}
    with _open_text(trades_path) as stream:
        for line in stream:
            if not line.strip():
                continue
            trade = json.loads(line)
            if str(trade.get("coin", "")).upper() != wanted:
                continue
            buyer, seller = trade["side_info"][0], trade["side_info"][1]
            maker = seller if str(trade["side"]).upper() == "B" else buyer
            oid = str(maker["oid"])
            fills[oid] = fills.get(oid, 0.0) + float(trade["sz"])
    return fills


def iter_hyperliquid_book_diffs(
    diffs_path: str | Path,
    trades_path: str | Path,
    coin: str,
    stats: dict[str, float] | None = None,
    max_events: int | None = None,
) -> Iterator[OrderEvent]:
    """Book diff L4 con le riduzioni classificate in fill o cancellazioni.

    Le riduzioni di un ordine (``update`` o ``remove``) vengono attribuite
    prima al volume eseguito osservato nei trade per quell'``oid`` maker
    (EXECUTE) e solo per il residuo a una cancellazione (CANCEL/DELETE). Gli
    ordini precedenti all'inizio del file non sono osservabili: gli eventi su
    di essi sono scartati e contati in ``stats``. I diff non hanno timestamp:
    il tempo e l'indice progressivo dell'evento.
    """

    wanted = coin.upper()
    pending = hyperliquid_maker_fills(trades_path, wanted)
    trade_volume = sum(pending.values())
    remaining: dict[str, float] = {}
    meta: dict[str, tuple[Side, float, str | None]] = {}
    counters = {"diffs": 0, "unknown_order_diffs": 0, "fill_events": 0, "fill_volume": 0.0,
                "cancel_events": 0, "cancel_volume": 0.0, "trade_volume": trade_volume}
    index = 0

    def event(kind: EventType, oid: str, size: float | None) -> OrderEvent:
        side, price, user = meta[oid]
        return OrderEvent(index, index, wanted, kind, oid, side, price, size, user_id=user)

    with _open_text(diffs_path) as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            if str(record.get("coin", "")).upper() != wanted:
                continue
            counters["diffs"] += 1
            oid = str(record["oid"])
            diff = record["raw_book_diff"]
            if isinstance(diff, dict) and "new" in diff:
                index += 1
                size = float(diff["new"]["sz"])
                meta[oid] = (_side(record), float(record["px"]), record.get("user"))
                remaining[oid] = size
                yield event(EventType.ADD, oid, size)
            else:
                if oid not in remaining:
                    counters["unknown_order_diffs"] += 1
                    continue
                if isinstance(diff, dict) and "update" in diff:
                    new_size = float(diff["update"]["newSz"])
                else:
                    new_size = 0.0
                reduction = remaining[oid] - new_size
                filled = min(max(reduction, 0.0), pending.get(oid, 0.0))
                if filled > 1e-12:
                    index += 1
                    pending[oid] -= filled
                    counters["fill_events"] += 1
                    counters["fill_volume"] += filled
                    yield event(EventType.EXECUTE, oid, filled)
                    if max_events is not None and index >= max_events:
                        break
                cancelled = reduction - filled
                if new_size <= 1e-12:
                    if cancelled > 1e-9 or filled <= 1e-12:
                        index += 1
                        counters["cancel_events"] += 1
                        counters["cancel_volume"] += max(cancelled, 0.0)
                        yield event(EventType.DELETE, oid, None)
                    remaining.pop(oid, None)
                    meta.pop(oid, None)
                    pending.pop(oid, None)
                elif cancelled > 1e-9:
                    index += 1
                    counters["cancel_events"] += 1
                    counters["cancel_volume"] += cancelled
                    yield event(EventType.CANCEL, oid, cancelled)
                    remaining[oid] = new_size
                else:
                    if cancelled < -1e-9:
                        # Aumento di size: perde priorita nel modello a coda.
                        index += 1
                        yield event(EventType.UPDATE, oid, new_size)
                    remaining[oid] = new_size
            if max_events is not None and index >= max_events:
                break
    counters["unmatched_trade_volume"] = trade_volume - counters["fill_volume"]
    if stats is not None:
        stats.update(counters)
