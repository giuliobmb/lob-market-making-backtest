from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class StrategyConfig:
    quote_size: float = 1.0
    volatility_window: int = 50
    volatility_min_periods: int = 20
    volatility_threshold: float = 0.001
    volatility_method: str = "rolling"
    # None: finestra in record; altrimenti midpoint campionato ogni volatility_sample_ns
    # nanosecondi e finestra di volatility_window intervalli (tempo di calendario).
    volatility_sample_ns: int | None = None
    return_type: str = "log"
    ewma_lambda: float = 0.94
    max_order_age_events: int = 100
    max_order_age_ns: int | None = None
    quote_offset_ticks: int = 0
    tick_size: float | None = None
    maker_fee_bps: float = 0.0
    taker_fee_bps: float = 1.0
    liquidation_mode: str = "market"
    max_abs_inventory: float = 1.0
    warmup_events: int = 0
    latency_events: int = 0
    latency_ns: int | None = None
    # Varianti della strategia (estensioni; i valori di default danno la strategia base).
    # inventory_mode "liquidate": alla scadenza si liquida a mercato; "skew": si mantiene
    # una quota passiva solo sul lato che riduce l'inventario, migliorata di skew_ticks,
    # e si liquida a mercato solo dopo max_hold_factor durate della quota.
    inventory_mode: str = "liquidate"
    skew_ticks: int = 0
    max_hold_factor: int = 10
    # Filtro di imbalance al best: si apre una coppia solo se |I| <= soglia e si cancella
    # la quota minacciata quando I la supera in direzione avversa (None = disattivato).
    imbalance_threshold: float | None = None

    def __post_init__(self) -> None:
        for name in ("quote_size", "max_abs_inventory", "volatility_threshold",
                     "maker_fee_bps", "taker_fee_bps", "ewma_lambda"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} deve essere finito")
        for name in ("volatility_window", "volatility_min_periods", "max_order_age_events",
                     "quote_offset_ticks", "warmup_events", "latency_events"):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} deve essere intero")
        if self.quote_size <= 0 or self.max_abs_inventory < self.quote_size:
            raise ValueError("Richiesto 0 < quote_size <= max_abs_inventory")
        if not 2 <= self.volatility_min_periods <= self.volatility_window:
            raise ValueError("Richiesto 2 <= volatility_min_periods <= volatility_window")
        if self.volatility_threshold < 0 or not 0 < self.ewma_lambda < 1:
            raise ValueError("Soglia o lambda non valida")
        if self.volatility_method not in {"rolling", "ewma"} or self.return_type not in {"log", "diff"}:
            raise ValueError("Metodo di volatilita o rendimento non valido")
        if self.max_order_age_events <= 0:
            raise ValueError("max_order_age_events deve essere positivo")
        if self.max_order_age_ns is not None and (type(self.max_order_age_ns) is not int or self.max_order_age_ns <= 0):
            raise ValueError("max_order_age_ns deve essere un intero positivo")
        if self.warmup_events < 0 or self.latency_events < 0:
            raise ValueError("warmup_events e latency_events devono essere non negativi")
        if self.volatility_sample_ns is not None and (type(self.volatility_sample_ns) is not int
                                                      or self.volatility_sample_ns <= 0):
            raise ValueError("volatility_sample_ns deve essere un intero positivo")
        if self.latency_ns is not None and (type(self.latency_ns) is not int or self.latency_ns < 0):
            raise ValueError("latency_ns deve essere un intero non negativo")
        if self.quote_offset_ticks < 0:
            raise ValueError("quote_offset_ticks deve essere non negativo")
        if self.tick_size is not None and (not math.isfinite(self.tick_size) or self.tick_size <= 0):
            raise ValueError("tick_size deve essere positivo e finito")
        if self.quote_offset_ticks and self.tick_size is None:
            raise ValueError("quote_offset_ticks richiede tick_size")
        if self.liquidation_mode not in {"market", "midpoint"}:
            raise ValueError("liquidation_mode deve essere market o midpoint")
        if self.inventory_mode not in {"liquidate", "skew"}:
            raise ValueError("inventory_mode deve essere liquidate o skew")
        if type(self.skew_ticks) is not int or self.skew_ticks < 0:
            raise ValueError("skew_ticks deve essere un intero non negativo")
        if self.skew_ticks and self.tick_size is None:
            raise ValueError("skew_ticks richiede tick_size")
        if type(self.max_hold_factor) is not int or self.max_hold_factor < 1:
            raise ValueError("max_hold_factor deve essere un intero positivo")
        if self.imbalance_threshold is not None and not 0 < self.imbalance_threshold <= 1:
            raise ValueError("imbalance_threshold deve essere in (0, 1]")

    @classmethod
    def from_json(cls, path: str | Path) -> StrategyConfig:
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))
