import unittest

from market_making_tesi.orderbook import MarketByOrderBook
from market_making_tesi.config import StrategyConfig
from market_making_tesi.execution import QueueAwareFillModel
from market_making_tesi.models import BookSnapshot, EventType, Fill, OrderEvent, Side, VirtualOrder
from market_making_tesi.accounting import Portfolio


class ExecutionTests(unittest.TestCase):
    def setUp(self):
        self.book = MarketByOrderBook("TEST")
        self.book.apply(OrderEvent(1, 1, "TEST", EventType.ADD, "ahead", Side.BUY, 100, 5))
        self.order = VirtualOrder("v", "p", "TEST", Side.BUY, 100, 2, 2, 1, 1, 20, None)
        self.model = QueueAwareFillModel()
        self.model.initialize_queue(self.order, self.book)

    def process(self, event):
        fills = self.model.fills_before_apply(event, self.book.get(event.order_id), [self.order], None)
        self.book.apply(event)
        return fills

    def test_cancel_and_delete_reduce_queue_without_fills(self):
        self.assertEqual(self.process(OrderEvent(2, 2, "TEST", EventType.CANCEL, "ahead", size=2)), [])
        self.assertEqual(self.order.queue_ahead, 3)
        self.assertEqual(self.process(OrderEvent(3, 3, "TEST", EventType.DELETE, "ahead")), [])
        self.assertEqual(self.order.queue_ahead, 0)

    def test_partial_execution_behind_is_limited_to_observed_size(self):
        self.process(OrderEvent(2, 2, "TEST", EventType.EXECUTE, "ahead", size=5))
        self.book.apply(OrderEvent(3, 3, "TEST", EventType.ADD, "behind", Side.BUY, 100, 3))
        fills = self.process(OrderEvent(4, 4, "TEST", EventType.EXECUTE, "behind", size=0.5))
        self.assertEqual(len(fills), 1)
        self.assertEqual(fills[0].size, 0.5)

    def test_cancel_behind_does_not_reduce_volume_ahead(self):
        self.book.apply(OrderEvent(2, 2, "TEST", EventType.ADD, "behind", Side.BUY, 100, 7))
        self.assertEqual(self.process(OrderEvent(3, 3, "TEST", EventType.DELETE, "behind")), [])
        self.assertEqual(self.order.queue_ahead, 5)

    def test_replace_loses_queue_priority(self):
        event = OrderEvent(2, 2, "TEST", EventType.REPLACE, "ahead",
                           new_order_id="replacement", new_price=100, new_size=5)
        self.assertEqual(self.process(event), [])
        self.assertEqual(self.order.queue_ahead, 0)
        fills = self.process(OrderEvent(3, 3, "TEST", EventType.EXECUTE, "replacement", size=1))
        self.assertEqual(fills[0].size, 1)

    def test_market_liquidation_and_fees(self):
        p = Portfolio()
        p.apply(Fill(1, 1, "v", "p", "TEST", Side.BUY, 100, 2, "maker", "test"), 10)
        self.assertAlmostEqual(p.cash, -200.2)
        p.liquidate(BookSnapshot(2, 2, "TEST", (99,), (10,), (101,), (10,)), 20)
        self.assertEqual(p.inventory, 0)
        self.assertAlmostEqual(p.cash, -2.596)
        self.assertAlmostEqual(sum(f.fee for f in p.fills), 0.596)

    def test_inventory_limit_and_invalid_configuration(self):
        for kwargs in ({"quote_size": 2, "max_abs_inventory": 1},
                       {"volatility_min_periods": 1}, {"tick_size": -1},
                       {"quote_offset_ticks": 1}, {"max_order_age_events": 0},
                       {"volatility_threshold": float("nan")}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                StrategyConfig(**kwargs)
