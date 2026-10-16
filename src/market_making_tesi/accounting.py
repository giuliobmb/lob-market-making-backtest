from __future__ import annotations

from .models import BookSnapshot, Fill, Side


class Portfolio:
    def __init__(self) -> None:
        self.inventory = 0.0
        self.cash = 0.0
        self.fills: list[Fill] = []

    def apply(self, fill: Fill, fee_bps: float) -> None:
        fill.fee = abs(fill.price * fill.size) * fee_bps / 10_000
        signed_size = fill.size if fill.side is Side.BUY else -fill.size
        self.inventory += signed_size
        self.cash -= signed_size * fill.price + fill.fee
        self.fills.append(fill)

    def equity(self, midpoint: float) -> float:
        return self.cash + self.inventory * midpoint

    def liquidate(self, snapshot: BookSnapshot, fee_bps: float,
                  mode: str = "market", reason: str = "end_of_sample") -> None:
        if mode not in {"market", "midpoint"}:
            raise ValueError(f"Modalita di liquidazione sconosciuta: {mode}")
        if abs(self.inventory) <= 1e-12:
            return
        side = Side.SELL if self.inventory > 0 else Side.BUY
        price = snapshot.best_bid if side is Side.SELL else snapshot.best_ask
        if mode == "midpoint":
            price = snapshot.midpoint
        self.apply(Fill(snapshot.timestamp, snapshot.event_index,
                        f"LIQ-{snapshot.event_index}", "LIQUIDATION", snapshot.symbol,
                        side, price, abs(self.inventory), "taker", reason,
                        mid=snapshot.midpoint), fee_bps)
