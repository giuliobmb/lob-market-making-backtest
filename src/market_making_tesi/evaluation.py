"""Metriche di valutazione e scomposizione del P&L.

Scomposizione (posizione nulla a fine sessione, prezzi in valuta del dataset)::

    P&L lordo = edge maker + costo di liquidazione + P&L di inventario

* edge maker: ``sum(s * (mid - prezzo) * q)`` sui fill passivi, cioe la frazione
  di spread incassata rispetto al midpoint al momento del fill (>= 0);
* costo di liquidazione: lo stesso termine sui fill taker (<= 0 con
  liquidazione al best, nullo con liquidazione al midpoint);
* P&L di inventario: variazione del midpoint mentre si detiene la posizione.

L'adverse selection e misurata con il markout: ``sum(s * (mid_dopo - mid) * q)``
sui fill passivi, con ``mid_dopo`` rilevato ``markout_records`` record dopo il
fill. ``s`` vale +1 per gli acquisti e -1 per le vendite. Le commissioni
cambiano solo la cassa e non le decisioni della strategia, quindi il P&L netto
di ogni scenario di costo si ricava dai fill del P&L lordo.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .engine import BacktestReport
from .models import Side


def _finite(value: float) -> float | None:
    return float(value) if value is not None and math.isfinite(value) else None


def _sign(side: Side) -> float:
    return 1.0 if side is Side.BUY else -1.0


def pair_outcomes(report: BacktestReport) -> dict[str, int]:
    bilateral = unilateral = none = partial = 0
    for pair in report.pairs.values():
        legs = [pair["filled_BUY"], pair["filled_SELL"]]
        filled = [q > 1e-12 for q in legs]
        if all(filled) and pair["sides"] >= 2:
            bilateral += 1
        elif any(filled):
            unilateral += 1
        else:
            none += 1
        partial += any(1e-12 < q < pair["size"] - 1e-12 for q in legs)
    return {"pairs_placed": len(report.pairs), "pairs_bilateral": bilateral,
            "pairs_unilateral": unilateral, "pairs_unfilled": none, "pairs_partial": partial}


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    values = equity.to_numpy(dtype=float)
    return float(np.max(np.maximum.accumulate(values) - values))


def summarize(report: BacktestReport) -> dict[str, float | int | str | None]:
    fills = report.fills
    maker = [f for f in fills if f.liquidity == "maker"]
    taker = [f for f in fills if f.liquidity != "maker"]
    final_equity = float(report.equity_curve[-1]["equity"]) if report.equity_curve else 0.0

    # Per coppia e lato: controvalore e quantita eseguiti (prezzo medio se piu ordini, es. skew).
    legs: dict[str, dict[Side, list[float]]] = {}
    for fill in maker:
        leg = legs.setdefault(fill.pair_id, {}).setdefault(fill.side, [0.0, 0.0])
        leg[0] += fill.price * fill.size
        leg[1] += fill.size
    completed = [pair for pair in legs.values() if len(pair) == 2]
    spread = sum((p[Side.SELL][0] / p[Side.SELL][1] - p[Side.BUY][0] / p[Side.BUY][1])
                 * min(p[Side.SELL][1], p[Side.BUY][1]) for p in completed)

    def edge(group):
        return sum(_sign(f.side) * (f.mid - f.price) * f.size for f in group if math.isfinite(f.mid))

    maker_edge, liquidation = edge(maker), edge(taker)
    markouts = {Side.BUY: [], Side.SELL: []}
    for f in maker:
        if math.isfinite(f.mid) and math.isfinite(f.mid_after):
            markouts[f.side].append(_sign(f.side) * (f.mid_after - f.mid))
    adverse = sum(_sign(f.side) * (f.mid_after - f.mid) * f.size for f in maker
                  if math.isfinite(f.mid) and math.isfinite(f.mid_after))
    curve = report.equity_frame()
    outcomes = pair_outcomes(report)
    return {
        "dataset_mode": report.dataset_mode,
        "placed_orders": report.placed_orders,
        "cancelled_orders": report.cancelled_orders,
        "maker_fills": len(maker),
        "taker_fills": len(taker),
        "total_fills": len(fills),
        "completed_pairs": len(completed),
        "spread_captured": float(spread),
        "final_equity": final_equity,
        "total_fees": float(sum(f.fee for f in fills)),
        **outcomes,
        "fill_rate_pairs": outcomes["pairs_bilateral"] / outcomes["pairs_placed"] if outcomes["pairs_placed"] else None,
        "maker_edge": float(maker_edge),
        "liquidation_cost": float(liquidation),
        "inventory_pnl": float(final_equity - maker_edge - liquidation),
        "pnl_midpoint_liquidation": float(final_equity - liquidation),
        "adverse_selection": float(adverse),
        "markout_bid_mean": _finite(np.mean(markouts[Side.BUY])) if markouts[Side.BUY] else None,
        "markout_ask_mean": _finite(np.mean(markouts[Side.SELL])) if markouts[Side.SELL] else None,
        "turnover": float(sum(abs(f.price * f.size) for f in fills)),
        "max_drawdown": max_drawdown(curve["equity"]) if "equity" in curve else 0.0,
        "pnl_per_pair": final_equity / outcomes["pairs_placed"] if outcomes["pairs_placed"] else None,
        **report.diagnostics,
    }


def fees_from_fills(fills: pd.DataFrame, maker_bps: float, taker_bps: float) -> float:
    """Commissioni in bps sul controvalore (negative = rebate)."""
    if fills.empty:
        return 0.0
    notional = (fills["price"] * fills["size"]).abs()
    bps = np.where(fills["liquidity"] == "maker", maker_bps, taker_bps)
    return float((notional * bps / 10_000).sum())


def bootstrap_mean_ci(values, n: int = 10_000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """Media e intervallo bootstrap percentile (seme fisso: risultati riproducibili)."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n, values.size), replace=True).mean(axis=1)
    low, high = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(values.mean()), float(low), float(high)


def session_statistics(frame: pd.DataFrame, value: str, by: list[str]) -> pd.DataFrame:
    """Per gruppo: sessioni, media, IC bootstrap 95%, quota di sessioni positive."""
    rows = []
    for key, group in frame.groupby(by, sort=False):
        mean, low, high = bootstrap_mean_ci(group[value])
        key = key if isinstance(key, tuple) else (key,)
        rows.append({**dict(zip(by, key)), "sessioni": len(group), "media": mean,
                     "ic95_basso": low, "ic95_alto": high,
                     "quota_positive": float((group[value] > 0).mean())})
    return pd.DataFrame(rows)
