from __future__ import annotations

import unittest

from market_making_tesi.orderbook import MarketByOrderBook
from market_making_tesi.config import StrategyConfig
from market_making_tesi.engine import run_l2_backtest, run_order_level_backtest
from market_making_tesi.execution import (
    PriceThroughFillModel,
    QueueAwareFillModel,
    TouchFillModel,
)
from market_making_tesi.models import (
    BookSnapshot,
    EventType,
    OrderEvent,
    Side,
    VirtualOrder,
)
from market_making_tesi.volatility import VolatilityEstimator


def snapshot(index: int, bid: float = 100.0, ask: float = 101.0) -> BookSnapshot:
    return BookSnapshot(index, index, "TEST", (bid,), (10.0,), (ask,), (10.0,))


class CoreTests(unittest.TestCase):
    def test_volatility_uses_only_observed_returns(self) -> None:
        estimator = VolatilityEstimator(5, 2, return_type="diff")
        self.assertIsNone(estimator.update(100.0))
        self.assertIsNone(estimator.update(101.0))
        self.assertAlmostEqual(estimator.update(100.0), 2**0.5)

    def test_l2_fill_rules_have_expected_bias(self) -> None:
        order = VirtualOrder(
            "v1", "p1", "TEST", Side.BUY, 100.0, 1.0, 1.0, 0, 0, 10, None
        )
        current = snapshot(1, bid=100.0, ask=101.0)
        self.assertEqual(len(TouchFillModel().fills(current, [order])), 1)
        self.assertEqual(len(PriceThroughFillModel().fills(current, [order])), 0)

    def test_queue_model_waits_for_volume_ahead(self) -> None:
        book = MarketByOrderBook("TEST")
        add_ahead = OrderEvent(1, 1, "TEST", EventType.ADD, "a", Side.BUY, 100, 5)
        add_ask = OrderEvent(2, 2, "TEST", EventType.ADD, "s", Side.SELL, 101, 5)
        book.apply(add_ahead)
        book.apply(add_ask)
        virtual = VirtualOrder(
            "v", "p", "TEST", Side.BUY, 100, 2, 2, 2, 2, 20, None
        )
        model = QueueAwareFillModel()
        model.initialize_queue(virtual, book)
        snap = book.snapshot(2, 2)
        assert snap is not None

        execute_ahead = OrderEvent(3, 3, "TEST", EventType.EXECUTE, "a", size=5)
        fills = model.fills_before_apply(execute_ahead, book.get("a"), [virtual], snap)
        self.assertEqual(fills, [])
        self.assertEqual(virtual.queue_ahead, 0)
        book.apply(execute_ahead)

        add_behind = OrderEvent(4, 4, "TEST", EventType.ADD, "b", Side.BUY, 100, 3)
        book.apply(add_behind)
        execute_behind = OrderEvent(5, 5, "TEST", EventType.EXECUTE, "b", size=3)
        fills = model.fills_before_apply(execute_behind, book.get("b"), [virtual], snap)
        self.assertEqual(len(fills), 1)
        self.assertEqual(fills[0].size, 2)

    def test_l2_engine_completes_round_trip(self) -> None:
        config = StrategyConfig(
            volatility_window=3,
            volatility_min_periods=2,
            volatility_threshold=10.0,
            return_type="diff",
            max_order_age_events=10,
            taker_fee_bps=0,
        )
        data = [snapshot(i) for i in range(5)]
        report = run_l2_backtest(data, config, TouchFillModel())
        summary = report.summary()
        self.assertEqual(summary["maker_fills"], 4)
        self.assertAlmostEqual(summary["final_equity"], 2.0)

    def test_order_level_engine_completes_queue_aware_round_trip(self) -> None:
        def add(i: int, oid: str, side: Side, price: float, size: float) -> OrderEvent:
            return OrderEvent(i, i, "TEST", EventType.ADD, oid, side, price, size)

        def execute(i: int, oid: str, size: float) -> OrderEvent:
            return OrderEvent(i, i, "TEST", EventType.EXECUTE, oid, size=size)

        events = [
            add(1, "bid-ahead", Side.BUY, 100, 5),
            add(2, "ask-ahead", Side.SELL, 101, 5),
            add(3, "bid-depth", Side.BUY, 99, 5),
            add(4, "ask-depth", Side.SELL, 102, 5),
            execute(5, "bid-ahead", 5),
            execute(6, "ask-ahead", 5),
            add(7, "bid-behind", Side.BUY, 100, 5),
            add(8, "ask-behind", Side.SELL, 101, 5),
            execute(9, "bid-behind", 5),
            execute(10, "ask-behind", 5),
        ]
        config = StrategyConfig(
            volatility_window=3,
            volatility_min_periods=2,
            volatility_threshold=10,
            max_order_age_events=100,
            taker_fee_bps=0,
        )
        report = run_order_level_backtest(events, config)
        self.assertEqual(report.summary()["maker_fills"], 2)
        self.assertAlmostEqual(report.summary()["final_equity"], 1.0)


if __name__ == "__main__":
    unittest.main()
