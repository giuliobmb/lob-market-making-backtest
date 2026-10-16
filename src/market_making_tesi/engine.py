"""Simulatore: collega mercato storico, esecuzione, contabilita e strategia.

Ordine delle operazioni a ogni record di mercato:
1. fill delle quote vive (prima dell'evento per la coda FIFO, sullo snapshot per L2);
2. attivazioni, cancellazioni e liquidazioni giunte a scadenza (latenza);
3. aggiornamento della volatilita e controlli di rischio della strategia;
4. eventuale nuova coppia di quote;
5. registrazione di equity e markout.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Iterable

import pandas as pd

from .accounting import Portfolio
from .config import StrategyConfig
from .execution import L2FillModel, OrderManager, QueueAwareFillModel
from .models import BookSnapshot, Fill, OrderEvent, Side
from .orderbook import MarketByOrderBook
from .strategy import MarketMakingStrategy, PlaceOrder, StrategyState

EPS = 1e-12


@dataclass(slots=True)
class BacktestReport:
    fills: list[Fill]
    equity_curve: list[dict[str, float | int]]
    placed_orders: int
    cancelled_orders: int
    dataset_mode: str
    diagnostics: dict[str, float | int] = field(default_factory=dict)
    pairs: dict[str, dict[str, float]] = field(default_factory=dict)

    def fills_frame(self) -> pd.DataFrame:
        if not self.fills:
            return pd.DataFrame(columns=["timestamp", "event_index", "order_id", "pair_id", "symbol",
                                         "side", "price", "size", "liquidity", "reason", "fee",
                                         "mid", "mid_after"])
        rows = []
        for fill in self.fills:
            row = asdict(fill)
            row["side"] = fill.side.value
            rows.append(row)
        return pd.DataFrame(rows)

    def equity_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.equity_curve)

    def pairs_frame(self) -> pd.DataFrame:
        return pd.DataFrame([{"pair_id": k, **v} for k, v in self.pairs.items()])

    def summary(self) -> dict[str, float | int | str]:
        from .evaluation import summarize
        return summarize(self)


class Simulator:
    def __init__(self, config: StrategyConfig, record_every: int = 1, record_fills: bool = True,
                 markout_records: int = 100, queue_model: QueueAwareFillModel | None = None) -> None:
        if record_every < 1 or markout_records < 1:
            raise ValueError("record_every e markout_records devono essere almeno 1")
        self.config = config
        self.record_every = record_every
        self.record_fills = record_fills
        self.markout_records = markout_records
        self.strategy = MarketMakingStrategy(config)
        self.portfolio = Portfolio()
        self.manager = OrderManager(config.latency_events, config.latency_ns, queue_model)
        self.equity_curve: list[dict[str, float | int]] = []
        self.pairs: dict[str, dict[str, float]] = {}
        self._markouts: list[tuple[int, Fill]] = []
        self.records = 0
        self.halted_records = 0
        self.liquidations = {"high_volatility": 0, "pair_expiry": 0, "orphan_inventory": 0, "max_hold": 0}
        self.expired_pairs = 0
        self.max_abs_inventory_seen = 0.0
        self.last_snapshot: BookSnapshot | None = None
        self._recorded_last = False

    # --- esecuzione e contabilita -------------------------------------------------
    def apply_fills(self, fills: list[Fill], mid: float) -> bool:
        """Applica i fill virtuali rispettando ``max_abs_inventory``."""
        by_id = {order.order_id: order for order in self.manager.orders}
        limit = self.config.max_abs_inventory
        applied = False
        for fill in fills:
            order = by_id.get(fill.order_id)
            if order is None or not order.active:
                continue
            inventory = self.portfolio.inventory
            room = limit - inventory if fill.side is Side.BUY else limit + inventory
            size = min(fill.size, order.remaining, max(0.0, room))
            if size <= EPS:
                continue
            fill = Fill(fill.timestamp, fill.event_index, fill.order_id, fill.pair_id, fill.symbol,
                        fill.side, fill.price, size, fill.liquidity, fill.reason, mid=mid)
            self.portfolio.apply(fill, self.config.maker_fee_bps)
            pair = self.pairs.get(order.pair_id)
            if pair is not None:
                pair[f"filled_{fill.side.value}"] += size
            # Il fill appartiene al record in corso (records + 1).
            self._markouts.append((self.records + 1 + self.markout_records, fill))
            applied = True
            order.remaining -= size
            if order.remaining <= EPS:
                order.remaining = 0.0
                order.active = False
            self.max_abs_inventory_seen = max(self.max_abs_inventory_seen, abs(self.portfolio.inventory))
        return applied

    def _liquidate(self, snapshot: BookSnapshot, reasons: list[str]) -> bool:
        traded = False
        for reason in reasons:
            if abs(self.portfolio.inventory) > EPS:
                self.portfolio.liquidate(snapshot, self.config.taker_fee_bps,
                                         self.config.liquidation_mode, reason=reason)
                self.liquidations[reason] = self.liquidations.get(reason, 0) + 1
                traded = True
        return traded

    def _state(self) -> StrategyState:
        return StrategyState(self.portfolio.inventory, self.manager.working(), self.manager.busy,
                             self.records)

    # --- passo della strategia -----------------------------------------------------
    def step(self, snapshot: BookSnapshot, book: MarketByOrderBook | None = None,
             fills_applied: bool = False) -> None:
        self.last_snapshot = snapshot
        self.records += 1
        self._resolve_markouts(snapshot.midpoint)
        traded = self._liquidate(snapshot, self.manager.process(snapshot, book)) or fills_applied

        sigma = self.strategy.observe(snapshot)
        if self.strategy.high_volatility and self.records > self.config.warmup_events:
            self.halted_records += 1  # solo nel tratto di valutazione
        risk = self.strategy.risk(snapshot, self._state())
        if any(getattr(intent, "reason", "") == "pair_expiry" for intent in risk):
            self.expired_pairs += 1
        traded = self._liquidate(snapshot, self.manager.submit(risk, snapshot, book)) or traded

        quotes = self.strategy.quote(snapshot, self._state())
        for intent in quotes:
            if not isinstance(intent, PlaceOrder):
                continue
            pair = self.pairs.setdefault(intent.order.pair_id, {
                "placed_event": snapshot.event_index, "size": intent.order.size, "sides": 0,
                "filled_BUY": 0.0, "filled_SELL": 0.0})
            pair["sides"] += 1
        traded = self._liquidate(snapshot, self.manager.submit(quotes, snapshot, book)) or traded
        self.record(snapshot, sigma, force=traded)

    def _resolve_markouts(self, mid: float) -> None:
        if self._markouts and self._markouts[0][0] <= self.records:
            due = [fill for record, fill in self._markouts if record <= self.records]
            self._markouts = [(r, f) for r, f in self._markouts if r > self.records]
            for fill in due:
                fill.mid_after = mid

    def record(self, snapshot: BookSnapshot, sigma: float | None, force: bool = False) -> None:
        self._recorded_last = (force and self.record_fills) or self.records % self.record_every == 0
        if not self._recorded_last:
            return
        self.equity_curve.append({
            "timestamp": snapshot.timestamp,
            "event_index": snapshot.event_index,
            "midpoint": snapshot.midpoint,
            "volatility": float("nan") if sigma is None else sigma,
            "low_volatility": int(sigma is not None and not self.strategy.high_volatility),
            "inventory": self.portfolio.inventory,
            "cash": self.portfolio.cash,
            "equity": self.portfolio.equity(snapshot.midpoint),
        })

    def finish(self, mode: str) -> BacktestReport:
        snapshot = self.last_snapshot
        truncated = len(self._markouts)
        if snapshot is not None:
            for _, fill in self._markouts:
                fill.mid_after = snapshot.midpoint
            self._markouts = []
            self.manager.cancel_everything()
            self.manager.pending_liquidation = None
            self._liquidate(snapshot, ["end_of_sample"])
            if not self._recorded_last:
                self.equity_curve.append({"timestamp": snapshot.timestamp,
                                          "event_index": snapshot.event_index,
                                          "midpoint": snapshot.midpoint,
                                          "volatility": float("nan")})
            last = self.equity_curve[-1]
            last["inventory"] = self.portfolio.inventory
            last["cash"] = self.portfolio.cash
            last["equity"] = self.portfolio.equity(snapshot.midpoint)
        diagnostics = {
            "records": self.records,
            "halted_records": self.halted_records,
            "volatility_liquidations": self.liquidations["high_volatility"],
            "expired_pairs": self.expired_pairs,
            "expiry_liquidations": self.liquidations["pair_expiry"],
            "orphan_liquidations": self.liquidations["orphan_inventory"],
            "max_hold_liquidations": self.liquidations["max_hold"],
            **self.strategy.counters,
            "post_only_rejects": self.manager.post_only_rejects,
            "max_abs_inventory_seen": self.max_abs_inventory_seen,
            "markouts_truncated": truncated,
        }
        return BacktestReport(self.portfolio.fills, self.equity_curve, self.manager.placed,
                              self.manager.cancelled, mode, diagnostics, self.pairs)


def run_l2_backtest(
    snapshots: Iterable[BookSnapshot],
    config: StrategyConfig,
    fill_model: L2FillModel,
    record_every: int = 1,
    record_fills: bool = True,
    markout_records: int = 100,
) -> BacktestReport:
    """Replay su snapshot L2 con una regola euristica di fill."""
    sim = Simulator(config, record_every, record_fills, markout_records)
    for snapshot in snapshots:
        live = sim.manager.live()
        applied = bool(live) and sim.apply_fills(fill_model.fills(snapshot, live), snapshot.midpoint)
        sim.step(snapshot, fills_applied=applied)
    return sim.finish("L2")


def run_order_level_backtest(
    events: Iterable[OrderEvent],
    config: StrategyConfig,
    depth: int = 10,
    record_every: int = 1,
    record_fills: bool = True,
    markout_records: int = 100,
) -> BacktestReport:
    """Replay order-level con coda FIFO.

    Gli eventi con ``metadata["bootstrap"]`` ricostruiscono soltanto il book
    (stato iniziale, pre-apertura) e non attivano la strategia.
    """
    queue_model = QueueAwareFillModel()
    sim = Simulator(config, record_every, record_fills, markout_records, queue_model)
    book = MarketByOrderBook()
    for event in events:
        if event.metadata.get("bootstrap"):
            book.apply(event)
            continue
        live = sim.manager.live()
        applied = False
        if live:
            fills = queue_model.fills_before_apply(event, book.get(event.order_id), live,
                                                   sim.last_snapshot)
            if fills:
                mid = sim.last_snapshot.midpoint if sim.last_snapshot else float("nan")
                applied = sim.apply_fills(fills, mid)
        book.apply(event)
        snapshot = book.snapshot(event.timestamp, event.event_index, depth=depth)
        if snapshot is None:
            continue
        sim.step(snapshot, book, fills_applied=applied)
    return sim.finish("ORDER_LEVEL")
