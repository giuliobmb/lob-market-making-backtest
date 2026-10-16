from __future__ import annotations

import gzip
from collections.abc import Iterator
from pathlib import Path

from ..models import EventType, OrderEvent, Side


def _u(data: bytes) -> int:
    return int.from_bytes(data, byteorder="big", signed=False)


def _stock(data: bytes) -> str:
    return data.decode("ascii", errors="replace").strip()


def iter_itch50_gzip(
    path: str | Path,
    symbol: str | None = None,
    max_messages: int | None = None,
) -> Iterator[OrderEvent]:
    """Legge i messaggi Nasdaq TotalView-ITCH 5.0 da un file ``.gz``.

    Sono normalizzati i messaggi necessari al book MBO: A/F, E/C, X, D e U.
    Il filtro per simbolo conserva internamente gli order reference number
    pertinenti, perché i messaggi successivi all'aggiunta non riportano lo
    stock. I prezzi ITCH sono espressi in decimillesimi di dollaro.
    """

    wanted = symbol.strip().upper() if symbol else None
    tracked: set[str] = set()
    message_index = 0

    with gzip.open(Path(path), "rb") as stream:
        while True:
            raw_length = stream.read(2)
            if not raw_length:
                break
            if len(raw_length) != 2:
                raise EOFError("Prefisso di lunghezza ITCH incompleto")
            length = _u(raw_length)
            message = stream.read(length)
            if len(message) != length:
                raise EOFError("Messaggio ITCH incompleto")

            message_index += 1
            if max_messages is not None and message_index > max_messages:
                break
            if not message:
                continue

            kind = chr(message[0])
            timestamp = _u(message[5:11]) if len(message) >= 11 else message_index
            event: OrderEvent | None = None

            if kind in {"A", "F"} and len(message) >= 36:
                order_id = str(_u(message[11:19]))
                stock = _stock(message[24:32])
                if wanted is not None and stock.upper() != wanted:
                    continue
                tracked.add(order_id)
                event = OrderEvent(
                    timestamp=timestamp,
                    event_index=message_index,
                    symbol=stock,
                    event_type=EventType.ADD,
                    order_id=order_id,
                    side=Side.BUY if chr(message[19]) == "B" else Side.SELL,
                    size=float(_u(message[20:24])),
                    price=_u(message[32:36]) / 10_000.0,
                    metadata={"itch_type": kind},
                )
            elif kind in {"E", "C"} and len(message) >= 31:
                order_id = str(_u(message[11:19]))
                if wanted is not None and order_id not in tracked:
                    continue
                event = OrderEvent(
                    timestamp=timestamp,
                    event_index=message_index,
                    symbol=wanted or "ITCH",
                    event_type=EventType.EXECUTE,
                    order_id=order_id,
                    size=float(_u(message[19:23])),
                    price=_u(message[32:36]) / 10_000.0 if kind == "C" and len(message) >= 36 else None,
                    metadata={"itch_type": kind, "match_number": _u(message[23:31])},
                )
            elif kind == "X" and len(message) >= 23:
                order_id = str(_u(message[11:19]))
                if wanted is not None and order_id not in tracked:
                    continue
                event = OrderEvent(
                    timestamp=timestamp,
                    event_index=message_index,
                    symbol=wanted or "ITCH",
                    event_type=EventType.CANCEL,
                    order_id=order_id,
                    size=float(_u(message[19:23])),
                    metadata={"itch_type": kind},
                )
            elif kind == "D" and len(message) >= 19:
                order_id = str(_u(message[11:19]))
                if wanted is not None and order_id not in tracked:
                    continue
                tracked.discard(order_id)
                event = OrderEvent(
                    timestamp=timestamp,
                    event_index=message_index,
                    symbol=wanted or "ITCH",
                    event_type=EventType.DELETE,
                    order_id=order_id,
                    metadata={"itch_type": kind},
                )
            elif kind == "U" and len(message) >= 35:
                old_id = str(_u(message[11:19]))
                new_id = str(_u(message[19:27]))
                if wanted is not None and old_id not in tracked:
                    continue
                if old_id in tracked:
                    tracked.remove(old_id)
                    tracked.add(new_id)
                event = OrderEvent(
                    timestamp=timestamp,
                    event_index=message_index,
                    symbol=wanted or "ITCH",
                    event_type=EventType.REPLACE,
                    order_id=old_id,
                    new_order_id=new_id,
                    new_size=float(_u(message[27:31])),
                    new_price=_u(message[31:35]) / 10_000.0,
                    metadata={"itch_type": kind},
                )

            if event is not None:
                yield event


SESSION_OPEN_NS = 34_200 * 10**9   # 09:30:00 (nanosecondi dalla mezzanotte)
SESSION_CLOSE_NS = 57_600 * 10**9  # 16:00:00


def extract_itch_symbols(
    path: str | Path,
    symbols: list[str],
    chunk_bytes: int = 1 << 24,
    stop_ns: int | None = None,
    sink=None,
    flush_rows: int = 500_000,
) -> tuple[dict[str, dict[str, list]], dict[str, float]]:
    """Estrae gli eventi MBO di alcuni titoli da un file ITCH 5.0 gzip, anche troncato.

    La decompressione procede a blocchi; un messaggio incompleto alla fine di un
    file troncato viene scartato. Il filtro usa lo *stock locate* (byte 1-2 di
    ogni messaggio), assegnato a ciascun titolo dai messaggi ``R`` di inizio
    giornata. Gli eventi precedenti alle 09:30 sono marcati ``bootstrap``:
    servono a ricostruire il book di apertura ma non alla strategia.
    Restituisce le colonne per titolo (schema di ``storage.EVENT_SCHEMA``) e
    statistiche di lettura.

    ``stop_ns`` interrompe la decompressione appena il flusso supera quell'ora
    (es. la chiusura delle 16:00: la negoziazione after-hours non serve).
    ``sink(simbolo, colonne)`` riceve gli eventi a blocchi di ``flush_rows``
    righe, cosi una giornata completa non viene tenuta in memoria; in quel caso
    le colonne restituite sono vuote.
    """
    import zlib

    wanted = {s.upper() for s in symbols}
    locate_to_symbol: dict[int, str] = {}
    columns = {s: {k: [] for k in ("timestamp", "event_index", "event_type", "order_id", "side", "price",
                                   "size", "new_order_id", "new_price", "new_size", "bootstrap")}
               for s in wanted}
    decompressor = zlib.decompressobj(16 + zlib.MAX_WBITS)
    buffer = b""
    messages = 0
    last_ts = 0
    sides = {66: "BUY", 83: "SELL"}  # 'B', 'S'
    rows = 0
    stream_ts = 0

    def flush(force: bool = False) -> None:
        nonlocal rows
        if sink is None:
            return
        for sym, cols in columns.items():
            if cols["timestamp"] and (force or len(cols["timestamp"]) >= flush_rows):
                sink(sym, cols)
                for values in cols.values():
                    values.clear()

    def push(sym, ts, kind, oid, side=None, price=None, size=None, new_oid=None, new_price=None, new_size=None):
        c = columns[sym]
        c["timestamp"].append(ts)
        c["event_index"].append(messages)
        c["event_type"].append(kind)
        c["order_id"].append(oid)
        c["side"].append(side)
        c["price"].append(price)
        c["size"].append(size)
        c["new_order_id"].append(new_oid)
        c["new_price"].append(new_price)
        c["new_size"].append(new_size)
        c["bootstrap"].append(ts < SESSION_OPEN_NS)

    with Path(path).open("rb") as stream:
        while True:
            chunk = stream.read(chunk_bytes)
            if not chunk:
                break
            buffer += decompressor.decompress(chunk)
            while decompressor.eof and decompressor.unused_data:
                # File gzip composto da piu membri concatenati.
                rest = decompressor.unused_data
                decompressor = zlib.decompressobj(16 + zlib.MAX_WBITS)
                buffer += decompressor.decompress(rest)
            pos, end = 0, len(buffer)
            start = None
            while pos + 2 <= end:
                length = (buffer[pos] << 8) | buffer[pos + 1]
                stop = pos + 2 + length
                if stop > end:
                    break
                start = pos + 2
                pos = stop
                messages += 1
                kind = buffer[start]
                locate = (buffer[start + 1] << 8) | buffer[start + 2]
                if kind == 82:  # 'R' stock directory
                    name = buffer[start + 11:start + 19].decode("ascii").strip()
                    if name in wanted:
                        locate_to_symbol[locate] = name
                    continue
                sym = locate_to_symbol.get(locate)
                if sym is None:
                    continue
                msg = buffer[start:stop]
                ts = int.from_bytes(msg[5:11], "big")
                last_ts = ts
                if kind in (65, 70):  # 'A', 'F'
                    push(sym, ts, "ADD", str(int.from_bytes(msg[11:19], "big")), sides[msg[19]],
                         int.from_bytes(msg[32:36], "big") / 10_000.0, float(int.from_bytes(msg[20:24], "big")))
                elif kind in (69, 67):  # 'E', 'C'
                    push(sym, ts, "EXECUTE", str(int.from_bytes(msg[11:19], "big")),
                         size=float(int.from_bytes(msg[19:23], "big")))
                elif kind == 88:  # 'X'
                    push(sym, ts, "CANCEL", str(int.from_bytes(msg[11:19], "big")),
                         size=float(int.from_bytes(msg[19:23], "big")))
                elif kind == 68:  # 'D'
                    push(sym, ts, "DELETE", str(int.from_bytes(msg[11:19], "big")))
                elif kind == 85:  # 'U'
                    push(sym, ts, "REPLACE", str(int.from_bytes(msg[11:19], "big")),
                         new_oid=str(int.from_bytes(msg[19:27], "big")),
                         new_size=float(int.from_bytes(msg[27:31], "big")),
                         new_price=int.from_bytes(msg[31:35], "big") / 10_000.0)
            if start is not None:
                stream_ts = int.from_bytes(buffer[start + 5:start + 11], "big")
            buffer = buffer[pos:]
            flush()
            if stop_ns is not None and stream_ts >= stop_ns:
                break
    flush(force=True)
    stats = {"messages": messages, "last_timestamp_symbols": last_ts, "last_timestamp_stream": stream_ts,
             "truncated_tail_bytes": len(buffer), "symbols_found": len(locate_to_symbol)}
    return columns, stats
