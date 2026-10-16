from __future__ import annotations

from bisect import bisect_left, insort

from .models import BookSnapshot, EventType, HistoricalOrder, OrderEvent, Side


class MarketByOrderBook:
    """Ricostruzione minima MBO sufficiente per il replay queue-aware.

    I prezzi di ciascun lato sono mantenuti ordinati in modo incrementale
    (crescenti; il best bid e l'ultimo elemento, il best ask il primo) insieme
    al volume aggregato per livello: lo snapshot costa O(depth) anziche
    richiedere l'ordinamento di tutti i livelli a ogni evento.
    """

    def __init__(self, symbol: str = "") -> None:
        self.symbol = symbol
        self.orders: dict[str, HistoricalOrder] = {}
        self.levels: dict[Side, dict[float, set[str]]] = {Side.BUY: {}, Side.SELL: {}}
        self._volume: dict[Side, dict[float, float]] = {Side.BUY: {}, Side.SELL: {}}
        self._prices: dict[Side, list[float]] = {Side.BUY: [], Side.SELL: []}

    def get(self, order_id: str) -> HistoricalOrder | None:
        return self.orders.get(str(order_id))

    def _change_volume(self, side: Side, price: float, delta: float) -> None:
        volumes = self._volume[side]
        volumes[price] = volumes.get(price, 0.0) + delta

    def _insert(self, order: HistoricalOrder) -> None:
        if order.order_id in self.orders:
            self._remove(order.order_id)
        self.orders[order.order_id] = order
        ids = self.levels[order.side].get(order.price)
        if ids is None:
            self.levels[order.side][order.price] = {order.order_id}
            self._volume[order.side][order.price] = 0.0
            insort(self._prices[order.side], order.price)
        else:
            ids.add(order.order_id)
        self._change_volume(order.side, order.price, order.remaining)
        if order.symbol:
            self.symbol = order.symbol

    def _remove(self, order_id: str) -> HistoricalOrder | None:
        order = self.orders.pop(str(order_id), None)
        if order is None:
            return None
        ids = self.levels[order.side].get(order.price)
        if ids is not None:
            ids.discard(order.order_id)
            if not ids:
                del self.levels[order.side][order.price]
                del self._volume[order.side][order.price]
                prices = self._prices[order.side]
                del prices[bisect_left(prices, order.price)]
            else:
                self._change_volume(order.side, order.price, -order.remaining)
        return order

    def _resize(self, order: HistoricalOrder, remaining: float) -> None:
        if remaining <= 1e-12:
            self._remove(order.order_id)
            return
        self._change_volume(order.side, order.price, remaining - order.remaining)
        order.remaining = remaining

    def apply(self, event: OrderEvent) -> None:
        event_type = event.event_type
        order_id = str(event.order_id)

        if event_type is EventType.ADD:
            if event.side is None or event.price is None or event.size is None:
                raise ValueError("ADD richiede side, price e size")
            if event.size <= 0:
                return
            self._insert(
                HistoricalOrder(
                    order_id=order_id,
                    symbol=event.symbol or self.symbol,
                    side=event.side,
                    price=float(event.price),
                    remaining=float(event.size),
                    sequence=event.event_index,
                    user_id=event.user_id,
                )
            )
            return

        current = self.get(order_id)
        if current is None:
            return

        if event_type in {EventType.EXECUTE, EventType.CANCEL, EventType.TRADE}:
            reduction = current.remaining if event.size is None else float(event.size)
            self._resize(current, max(0.0, current.remaining - reduction))
        elif event_type is EventType.DELETE:
            self._remove(order_id)
        elif event_type is EventType.UPDATE:
            if event.size is None:
                raise ValueError("UPDATE richiede la quantità residua in size")
            self._resize(current, float(event.size))
        elif event_type is EventType.REPLACE:
            old = self._remove(order_id)
            if old is None:
                return
            new_id = str(event.new_order_id or order_id)
            new_price = float(event.new_price if event.new_price is not None else old.price)
            new_size = float(event.new_size if event.new_size is not None else old.remaining)
            if new_size > 0:
                self._insert(
                    HistoricalOrder(
                        order_id=new_id,
                        symbol=old.symbol,
                        side=old.side,
                        price=new_price,
                        remaining=new_size,
                        sequence=event.event_index,
                        user_id=old.user_id,
                    )
                )

    def volume_at(self, side: Side, price: float) -> float:
        return self._volume[side].get(price, 0.0)

    def prices(self, side: Side) -> list[float]:
        """Prezzi del lato in ordine crescente (vista interna, non modificare)."""
        return self._prices[side]

    def best(self, side: Side) -> float | None:
        prices = self._prices[side]
        if not prices:
            return None
        return prices[-1] if side is Side.BUY else prices[0]

    def snapshot(self, timestamp: int, event_index: int, depth: int = 10) -> BookSnapshot | None:
        bids, asks = self._prices[Side.BUY], self._prices[Side.SELL]
        if not bids or not asks:
            return None
        bid_prices = tuple(bids[: -depth - 1 : -1])
        ask_prices = tuple(asks[:depth])
        bid_volume, ask_volume = self._volume[Side.BUY], self._volume[Side.SELL]
        return BookSnapshot(
            timestamp=timestamp,
            event_index=event_index,
            symbol=self.symbol,
            bid_prices=bid_prices,
            bid_sizes=tuple(bid_volume[p] for p in bid_prices),
            ask_prices=ask_prices,
            ask_sizes=tuple(ask_volume[p] for p in ask_prices),
        )
