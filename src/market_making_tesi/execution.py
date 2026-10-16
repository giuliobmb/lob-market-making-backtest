from __future__ import annotations

from typing import Protocol

from .models import BookSnapshot, EventType, Fill, HistoricalOrder, OrderEvent, Side, VirtualOrder
from .orderbook import MarketByOrderBook
from .strategy import CancelAll, CancelOrder, Intent, Liquidate, PlaceOrder


def _fill(order: VirtualOrder, timestamp: int, index: int, size: float, reason: str) -> Fill:
    return Fill(timestamp, index, order.order_id, order.pair_id, order.symbol,
                order.side, order.price, size, "maker", reason)


class L2FillModel(Protocol):
    def fills(self, snapshot: BookSnapshot, orders: list[VirtualOrder]) -> list[Fill]: ...


class TouchFillModel:
    """Ipotesi ottimistica: il best sullo stesso lato raggiunge la quota."""
    reason = "l2_touch"

    def reached(self, snapshot: BookSnapshot, order: VirtualOrder) -> bool:
        return (snapshot.best_bid <= order.price if order.side is Side.BUY
                else snapshot.best_ask >= order.price)

    def fills(self, snapshot: BookSnapshot, orders: list[VirtualOrder]) -> list[Fill]:
        return [_fill(o, snapshot.timestamp, snapshot.event_index, o.remaining, self.reason)
                for o in orders if o.active and o.live and o.remaining > 0
                and snapshot.event_index > o.live_event and self.reached(snapshot, o)]


class PriceThroughFillModel(TouchFillModel):
    """Ipotesi piu restrittiva: il best dello stesso lato supera la quota.

    Anche questa e una proxy L2, non una prova di esecuzione osservata.
    """
    reason = "l2_price_through"

    def reached(self, snapshot: BookSnapshot, order: VirtualOrder) -> bool:
        return (snapshot.best_bid < order.price if order.side is Side.BUY
                else snapshot.best_ask > order.price)


class CrossFillModel(TouchFillModel):
    """Proxy sul lato opposto: il best contrario raggiunge la quota.

    Un acquisto e eseguito se il best ask scende fino al prezzo dell'ordine
    (``best_ask <= price``), una vendita se il best bid sale fino ad esso.
    E la convenzione piu diffusa in letteratura per i backtest su snapshot:
    a differenza di ``touch`` non scatta quando il mercato resta fermo.
    """
    reason = "l2_cross"

    def reached(self, snapshot: BookSnapshot, order: VirtualOrder) -> bool:
        return (snapshot.best_ask <= order.price if order.side is Side.BUY
                else snapshot.best_bid >= order.price)


L2_FILL_MODELS: dict[str, type[TouchFillModel]] = {
    "touch": TouchFillModel,
    "price-through": PriceThroughFillModel,
    "cross": CrossFillModel,
}


class QueueAwareFillModel:
    """Coda FIFO al prezzo: cancel riduce la coda ma non genera fill.

    Il volume eseguito su ordini posteriori puo raggiungere la quota virtuale.
    Una modifica che aumenta la size perde priorita (ipotesi del prototipo).
    """

    def initialize_queue(self, order: VirtualOrder, book: MarketByOrderBook) -> None:
        # Tutti gli ordini gia presenti al livello hanno priorita temporale.
        order.ahead_orders = {oid: book.orders[oid].remaining
                              for oid in book.levels[order.side].get(order.price, ())}
        order.queue_ahead = sum(order.ahead_orders.values())

    def fills_before_apply(self, event: OrderEvent, before: HistoricalOrder | None,
                           orders: list[VirtualOrder], snapshot: BookSnapshot | None) -> list[Fill]:
        if before is None:
            return []
        fills = []
        available = max(0.0, min(before.remaining, event.size if event.size is not None else before.remaining))
        for order in orders:
            if not order.active or not order.live or event.event_index <= order.live_event:
                continue
            if before.side is not order.side or before.price != order.price:
                continue
            oid = str(event.order_id)
            was_ahead = oid in order.ahead_orders
            if was_ahead:
                previous = order.ahead_orders[oid]
                if event.event_type in {EventType.DELETE, EventType.REPLACE}:
                    order.ahead_orders.pop(oid)
                elif event.event_type is EventType.UPDATE and event.size is not None:
                    if event.size > before.remaining or event.size <= 0:
                        order.ahead_orders.pop(oid)
                    else:
                        order.ahead_orders[oid] = min(previous, event.size)
                elif event.event_type in {EventType.EXECUTE, EventType.TRADE, EventType.CANCEL}:
                    left = max(0.0, previous - available)
                    if left > 1e-12:
                        order.ahead_orders[oid] = left
                    else:
                        order.ahead_orders.pop(oid)
                order.queue_ahead = sum(order.ahead_orders.values())
            if (not was_ahead and order.queue_ahead <= 1e-12
                    and event.event_type in {EventType.EXECUTE, EventType.TRADE}
                    and available > 0):
                size = min(order.remaining, available)
                fills.append(_fill(order, event.timestamp, event.event_index, size, "queue_aware"))
                available -= size
        return fills


class OrderManager:
    """Ciclo di vita degli ordini virtuali, con latenza di inserimento e cancellazione.

    Con latenza nulla gli intenti hanno effetto nello stesso record. Altrimenti un
    ordine entra nel mercato dopo ``latency_events`` record (e ``latency_ns``
    nanosecondi, se indicati) con il prezzo deciso in precedenza: se a quel punto
    attraverserebbe il book viene rifiutato (ordine post-only). Una cancellazione
    ha effetto con lo stesso ritardo e nel frattempo l'ordine resta eseguibile;
    la liquidazione parte dopo la cancellazione.
    """

    def __init__(self, latency_events: int = 0, latency_ns: int | None = None,
                 queue_model: QueueAwareFillModel | None = None) -> None:
        self.latency_events = latency_events
        self.latency_ns = latency_ns
        self.queue_model = queue_model
        self.orders: list[VirtualOrder] = []
        self.pending_liquidation: tuple[int, int | None, str] | None = None
        self.placed = 0
        self.cancelled = 0
        self.post_only_rejects = 0

    @property
    def immediate(self) -> bool:
        return self.latency_events == 0 and not self.latency_ns

    def _due_at(self, snapshot: BookSnapshot) -> tuple[int, int | None]:
        return (snapshot.event_index + self.latency_events,
                snapshot.timestamp + self.latency_ns if self.latency_ns else None)

    @staticmethod
    def _is_due(snapshot: BookSnapshot, event: int, timestamp: int | None) -> bool:
        return snapshot.event_index >= event and (timestamp is None or snapshot.timestamp >= timestamp)

    def active(self) -> list[VirtualOrder]:
        self.orders = [o for o in self.orders if o.active and o.remaining > 1e-12]
        return self.orders

    def live(self) -> list[VirtualOrder]:
        return [o for o in self.active() if o.live]

    def working(self) -> list[VirtualOrder]:
        return [o for o in self.active() if o.cancel_event is None]

    @property
    def busy(self) -> bool:
        return bool(self.active()) or self.pending_liquidation is not None

    def _activate(self, order: VirtualOrder, snapshot: BookSnapshot,
                  book: MarketByOrderBook | None) -> None:
        crosses = (order.price >= snapshot.best_ask if order.side is Side.BUY
                   else order.price <= snapshot.best_bid)
        if crosses:
            order.active = False
            self.post_only_rejects += 1
            return
        order.live = True
        order.live_event = snapshot.event_index
        order.live_timestamp = snapshot.timestamp
        if self.queue_model is not None:
            if book is None:
                raise ValueError("Il modello queue-aware richiede il book MBO")
            self.queue_model.initialize_queue(order, book)

    def submit(self, intents: list[Intent], snapshot: BookSnapshot,
               book: MarketByOrderBook | None = None) -> list[str]:
        """Registra gli intenti; restituisce le liquidazioni da eseguire subito."""
        liquidate_now: list[str] = []
        for intent in intents:
            if isinstance(intent, PlaceOrder):
                order = intent.order
                self.placed += 1
                self.orders.append(order)
                if self.immediate:
                    self._activate(order, snapshot, book)
                else:
                    order.live = False
                    order.live_event, order.live_timestamp = self._due_at(snapshot)
            elif isinstance(intent, (CancelAll, CancelOrder)):
                for order in self.active():
                    if order.cancel_event is not None:
                        continue
                    if isinstance(intent, CancelOrder) and order.order_id != intent.order_id:
                        continue
                    if self.immediate:
                        order.active = False
                        self.cancelled += 1
                    else:
                        order.cancel_event, order.cancel_timestamp = self._due_at(snapshot)
            elif isinstance(intent, Liquidate):
                if self.immediate:
                    liquidate_now.append(intent.reason)
                elif self.pending_liquidation is None:
                    self.pending_liquidation = (*self._due_at(snapshot), intent.reason)
        return liquidate_now

    def process(self, snapshot: BookSnapshot, book: MarketByOrderBook | None = None) -> list[str]:
        """Applica attivazioni e cancellazioni dovute; restituisce le liquidazioni dovute."""
        if self.immediate:
            return []
        for order in self.active():
            if not order.live and self._is_due(snapshot, order.live_event, order.live_timestamp):
                self._activate(order, snapshot, book)
            if (order.active and order.cancel_event is not None
                    and self._is_due(snapshot, order.cancel_event, order.cancel_timestamp)):
                order.active = False
                self.cancelled += 1
        if self.pending_liquidation is not None:
            event, timestamp, reason = self.pending_liquidation
            if self._is_due(snapshot, event, timestamp):
                self.pending_liquidation = None
                return [reason]
        return []

    def cancel_everything(self) -> None:
        for order in self.active():
            order.active = False
            self.cancelled += 1
