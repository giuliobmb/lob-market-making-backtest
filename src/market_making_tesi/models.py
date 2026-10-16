from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class EventType(str, Enum):
    ADD = "ADD"
    EXECUTE = "EXECUTE"
    CANCEL = "CANCEL"
    DELETE = "DELETE"
    UPDATE = "UPDATE"
    REPLACE = "REPLACE"
    TRADE = "TRADE"


@dataclass(slots=True)
class BookSnapshot:
    timestamp: int
    event_index: int
    symbol: str
    bid_prices: tuple[float, ...]
    bid_sizes: tuple[float, ...]
    ask_prices: tuple[float, ...]
    ask_sizes: tuple[float, ...]

    @property
    def best_bid(self) -> float:
        return self.bid_prices[0]

    @property
    def best_ask(self) -> float:
        return self.ask_prices[0]

    @property
    def midpoint(self) -> float:
        return (self.best_bid + self.best_ask) / 2

    @property
    def spread(self) -> float:
        return self.best_ask - self.best_bid


@dataclass(slots=True)
class OrderEvent:
    timestamp: int
    event_index: int
    symbol: str
    event_type: EventType
    order_id: str
    side: Side | None = None
    price: float | None = None
    size: float | None = None
    new_order_id: str | None = None
    new_price: float | None = None
    new_size: float | None = None
    user_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class HistoricalOrder:
    order_id: str
    symbol: str
    side: Side
    price: float
    remaining: float
    sequence: int
    user_id: str | None = None


@dataclass(slots=True)
class VirtualOrder:
    order_id: str
    pair_id: str
    symbol: str
    side: Side
    price: float
    size: float
    remaining: float
    placed_event: int
    placed_timestamp: int
    expires_event: int
    expires_timestamp: int | None
    active: bool = True
    queue_ahead: float = 0.0
    ahead_orders: dict[str, float] = field(default_factory=dict)
    # Latenza: l'ordine entra nel mercato al record ``live_event``/``live_timestamp``
    # e una cancellazione richiesta ha effetto da ``cancel_event``/``cancel_timestamp``.
    live: bool = True
    live_event: int = 0
    live_timestamp: int | None = None
    cancel_event: int | None = None
    cancel_timestamp: int | None = None


@dataclass(slots=True)
class Fill:
    timestamp: int
    event_index: int
    order_id: str
    pair_id: str
    symbol: str
    side: Side
    price: float
    size: float
    liquidity: str
    reason: str
    fee: float = 0.0
    # Midpoint al momento del fill e dopo l'orizzonte di markout (adverse selection).
    mid: float = float("nan")
    mid_after: float = float("nan")
