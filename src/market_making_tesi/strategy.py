"""Strategia di market making passivo: genera intenti di ordine, non esegue.

Regola operativa base (identica in tutti gli esperimenti principali):

* **filtro di volatilita:** a ogni record si stima la volatilita come deviazione
  standard dei rendimenti del midpoint su una finestra mobile di
  ``volatility_window`` record, oppure di ``volatility_window`` intervalli di
  ``volatility_sample_ns`` nanosecondi quando i dati hanno timestamp (stima
  causale). La strategia opera solo nel regime di bassa volatilita, cioe con
  stima sotto ``volatility_threshold``; sopra soglia cancella le quote,
  liquida l'inventario e non quota finche la volatilita non torna sotto soglia;
* con inventario nullo e nessun ordine aperto quota una coppia bid/ask al best
  per ``quote_size`` (una coppia alla volta);
* alla scadenza della coppia cancella e liquida l'inventario residuo.

Varianti (estensioni, disattivate per default):

* **inventory skew** (``inventory_mode="skew"``): dopo un fill su un solo lato,
  alla scadenza non si liquida a mercato ma si mantiene una quota passiva solo sul
  lato che riduce l'inventario, al best corrente migliorato di ``skew_ticks`` tick
  se lo spread lo consente (mai attraverso il book). Si liquida a mercato solo
  dopo ``max_hold_factor`` durate della quota o con volatilita sopra soglia;
* **filtro di imbalance** (``imbalance_threshold``): con I = (q_bid - q_ask) /
  (q_bid + q_ask) al best, si apre una coppia solo se |I| <= soglia; una quota
  viva viene cancellata se l'imbalance la minaccia (I > soglia per una vendita,
  I < -soglia per un acquisto), cioe se il prezzo tende a muoversi attraverso di essa.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import count

from .config import StrategyConfig
from .models import BookSnapshot, Side, VirtualOrder
from .volatility import VolatilityEstimator

EPS = 1e-12


@dataclass(slots=True)
class PlaceOrder:
    order: VirtualOrder


@dataclass(slots=True)
class CancelAll:
    reason: str


@dataclass(slots=True)
class CancelOrder:
    order_id: str
    reason: str


@dataclass(slots=True)
class Liquidate:
    reason: str


Intent = PlaceOrder | CancelAll | CancelOrder | Liquidate


@dataclass(slots=True)
class StrategyState:
    """Vista dello stato fornita dal simulatore alla strategia."""
    inventory: float
    working: list[VirtualOrder]   # ordini vivi o in attivazione, non in cancellazione
    busy: bool                    # ordini, cancellazioni o liquidazioni ancora pendenti
    records: int                  # record di mercato osservati finora


def imbalance(snapshot: BookSnapshot) -> float:
    """Imbalance dei volumi al best, in [-1, 1]; positivo = pressione in acquisto."""
    bid, ask = snapshot.bid_sizes[0], snapshot.ask_sizes[0]
    total = bid + ask
    return (bid - ask) / total if total > 0 else 0.0


def threatened(side: Side, value: float, threshold: float) -> bool:
    """Una vendita e minacciata da pressione in acquisto, un acquisto da pressione in vendita."""
    return value > threshold if side is Side.SELL else value < -threshold


class MarketMakingStrategy:
    def __init__(self, config: StrategyConfig) -> None:
        self.config = config
        self.volatility = VolatilityEstimator(
            window=config.volatility_window,
            min_periods=config.volatility_min_periods,
            method=config.volatility_method,
            return_type=config.return_type,
            ewma_lambda=config.ewma_lambda,
            sample_ns=config.volatility_sample_ns,
        )
        self.ids = count(1)
        self.sigma: float | None = None
        self.open_pair: str | None = None
        self.hold_since: tuple[int, int] | None = None
        self.counters = {"unwind_orders": 0, "imbalance_cancels": 0, "imbalance_skips": 0}

    def observe(self, snapshot: BookSnapshot) -> float | None:
        """Aggiorna la volatilita causale con il midpoint del record."""
        self.sigma = self.volatility.update(snapshot.midpoint, snapshot.timestamp)
        return self.sigma

    @property
    def high_volatility(self) -> bool:
        return self.sigma is not None and self.sigma > self.config.volatility_threshold

    @property
    def skew(self) -> bool:
        return self.config.inventory_mode == "skew"

    def _hold_exceeded(self, snapshot: BookSnapshot) -> bool:
        if self.hold_since is None:
            return False
        config = self.config
        event, timestamp = self.hold_since
        if snapshot.event_index - event >= config.max_hold_factor * config.max_order_age_events:
            return True
        return (config.max_order_age_ns is not None
                and snapshot.timestamp - timestamp >= config.max_hold_factor * config.max_order_age_ns)

    def risk(self, snapshot: BookSnapshot, state: StrategyState) -> list[Intent]:
        """Controlli di rischio: volatilita, imbalance avverso e scadenza delle quote."""
        flat = abs(state.inventory) <= EPS
        if flat:
            self.hold_since = None
        elif self.hold_since is None:
            self.hold_since = (snapshot.event_index, snapshot.timestamp)

        if self.high_volatility:
            if state.working or not flat:
                return [CancelAll("high_volatility"), Liquidate("high_volatility")]
            return []

        intents: list[Intent] = []
        working = state.working
        threshold = self.config.imbalance_threshold
        if threshold is not None and working:
            value = imbalance(snapshot)
            for order in working:
                if threatened(order.side, value, threshold):
                    intents.append(CancelOrder(order.order_id, "imbalance"))
                    self.counters["imbalance_cancels"] += 1
            cancelled = {i.order_id for i in intents}
            working = [o for o in working if o.order_id not in cancelled]

        if any(self._expired(order, snapshot) for order in working):
            intents.append(CancelAll("pair_expiry"))
            if not self.skew:
                intents.append(Liquidate("pair_expiry"))
            elif not flat and self._hold_exceeded(snapshot):
                intents.append(Liquidate("max_hold"))
            return intents
        if not flat and not state.busy and not intents:
            if not self.skew:
                # Inventario senza ordini aperti (fill durante una cancellazione in latenza).
                intents.append(Liquidate("orphan_inventory"))
            elif self._hold_exceeded(snapshot):
                intents.append(Liquidate("max_hold"))
        return intents

    def _common(self, snapshot: BookSnapshot, pair_id: str, size: float) -> dict:
        config = self.config
        return {
            "pair_id": pair_id,
            "symbol": snapshot.symbol,
            "size": size,
            "remaining": size,
            "placed_event": snapshot.event_index,
            "placed_timestamp": snapshot.timestamp,
            "expires_event": snapshot.event_index + config.max_order_age_events,
            "expires_timestamp": (snapshot.timestamp + config.max_order_age_ns
                                  if config.max_order_age_ns is not None else None),
        }

    def quote(self, snapshot: BookSnapshot, state: StrategyState) -> list[Intent]:
        config = self.config
        if (self.sigma is None or self.high_volatility or state.records <= config.warmup_events
                or state.busy or snapshot.spread <= 0):
            return []
        if abs(state.inventory) > EPS:
            return self._unwind(snapshot, state) if self.skew else []
        threshold = config.imbalance_threshold
        if threshold is not None and abs(imbalance(snapshot)) > threshold:
            self.counters["imbalance_skips"] += 1
            return []
        pair_id = f"P{next(self.ids)}"
        self.open_pair = pair_id
        offset = config.quote_offset_ticks * (config.tick_size or 0.0)
        common = self._common(snapshot, pair_id, config.quote_size)
        intents: list[Intent] = []
        # Un lato e quotato solo se un suo fill completo rispetta max_abs_inventory.
        if state.inventory + config.quote_size <= config.max_abs_inventory + EPS:
            intents.append(PlaceOrder(VirtualOrder(order_id=f"{pair_id}-B", side=Side.BUY,
                                                   price=snapshot.best_bid - offset, **common)))
        if state.inventory - config.quote_size >= -config.max_abs_inventory - EPS:
            intents.append(PlaceOrder(VirtualOrder(order_id=f"{pair_id}-S", side=Side.SELL,
                                                   price=snapshot.best_ask + offset, **common)))
        return intents

    def _unwind(self, snapshot: BookSnapshot, state: StrategyState) -> list[Intent]:
        """Quota passiva solo sul lato che riduce l'inventario (inventory skew)."""
        config = self.config
        if self._hold_exceeded(snapshot):
            return [Liquidate("max_hold")]
        side = Side.SELL if state.inventory > 0 else Side.BUY
        threshold = config.imbalance_threshold
        if threshold is not None and threatened(side, imbalance(snapshot), threshold):
            self.counters["imbalance_skips"] += 1
            return []
        improve = config.skew_ticks * (config.tick_size or 0.0)
        if side is Side.SELL:
            price = max(snapshot.best_ask - improve, snapshot.best_bid + (config.tick_size or 0.0))
            price = min(price, snapshot.best_ask)
        else:
            price = min(snapshot.best_bid + improve, snapshot.best_ask - (config.tick_size or 0.0))
            price = max(price, snapshot.best_bid)
        pair_id = self.open_pair or f"P{next(self.ids)}"
        self.counters["unwind_orders"] += 1
        order_id = f"{pair_id}-U{self.counters['unwind_orders']}"
        return [PlaceOrder(VirtualOrder(order_id=order_id, side=side, price=price,
                                        **self._common(snapshot, pair_id, abs(state.inventory))))]

    @staticmethod
    def _expired(order: VirtualOrder, snapshot: BookSnapshot) -> bool:
        if snapshot.event_index >= order.expires_event:
            return True
        return order.expires_timestamp is not None and snapshot.timestamp >= order.expires_timestamp
