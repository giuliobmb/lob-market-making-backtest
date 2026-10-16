import gzip
import json
import math
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np

from market_making_tesi.parsers import (
    fi2010_stock_segments,
    iter_hyperliquid_book_diffs,
)
from market_making_tesi.orderbook import MarketByOrderBook
from market_making_tesi.config import StrategyConfig
from market_making_tesi.engine import Simulator, run_l2_backtest, run_order_level_backtest
from market_making_tesi.evaluation import bootstrap_mean_ci, summarize
from market_making_tesi.execution import CrossFillModel, PriceThroughFillModel, TouchFillModel
from market_making_tesi.models import BookSnapshot, EventType, Fill, OrderEvent, Side, VirtualOrder
from market_making_tesi.volatility import VolatilityEstimator


def snap(i, bid=100.0, ask=101.0):
    return BookSnapshot(i, i, "T", (bid,), (10.0,), (ask,), (10.0,))


class StrategyGuards(unittest.TestCase):
    def test_inventory_limit_clips_fills(self):
        sim = Simulator(StrategyConfig(quote_size=1, max_abs_inventory=1))
        sim.portfolio.inventory = 0.5
        order = VirtualOrder("o", "p", "T", Side.BUY, 100, 1, 1, 0, 0, 10, None)
        sim.manager.orders = [order]
        sim.pairs["p"] = {"placed_event": 0, "size": 1, "sides": 2, "filled_BUY": 0.0, "filled_SELL": 0.0}
        sim.apply_fills([Fill(1, 1, "o", "p", "T", Side.BUY, 100, 1, "maker", "t")], mid=100.5)
        self.assertAlmostEqual(sim.portfolio.inventory, 1.0)
        self.assertAlmostEqual(order.remaining, 0.5)
        self.assertAlmostEqual(sim.portfolio.fills[0].mid, 100.5)

    def test_volatility_spike_cancels_and_liquidates(self):
        config = StrategyConfig(volatility_window=3, volatility_min_periods=2, volatility_threshold=0.5,
                                return_type="diff", max_order_age_events=100, taker_fee_bps=0)
        # Il bid scende sulla quota (fill touch del lato acquisto), poi il mercato salta.
        data = [snap(0), snap(1), snap(2), snap(3, bid=99.9, ask=101.0), snap(4, 110, 111)]
        report = run_l2_backtest(data, config, TouchFillModel())
        reasons = [f.reason for f in report.fills]
        self.assertIn("high_volatility", reasons)
        self.assertEqual(report.summary()["volatility_liquidations"], 1)

    def test_warmup_blocks_quoting(self):
        config = StrategyConfig(volatility_window=3, volatility_min_periods=2, volatility_threshold=10,
                                return_type="diff", warmup_events=4)
        report = run_l2_backtest([snap(i) for i in range(4)], config, TouchFillModel())
        self.assertEqual(report.placed_orders, 0)

    def test_cross_is_strictest_l2_rule(self):
        order = VirtualOrder("v", "p", "T", Side.BUY, 100, 1, 1, 0, 0, 10, None)
        cases = {"fermo": snap(1, 100, 101), "bid sotto": snap(1, 99, 101), "ask alla quota": snap(1, 99, 100)}
        counts = {name: [len(m.fills(s, [order])) for m in (TouchFillModel(), PriceThroughFillModel(), CrossFillModel())]
                  for name, s in cases.items()}
        self.assertEqual(counts, {"fermo": [1, 0, 0], "bid sotto": [1, 1, 0], "ask alla quota": [1, 1, 1]})

    def test_rolling_volatility_matches_numpy(self):
        rng = random.Random(7)
        estimator = VolatilityEstimator(5, 2, return_type="diff")
        mids, last = [100.0], None
        for _ in range(40):
            mids.append(mids[-1] + rng.uniform(-1, 1))
        for i, mid in enumerate(mids):
            last = estimator.update(mid)
        expected = np.std(np.diff(mids)[-5:], ddof=1)
        self.assertTrue(math.isclose(last, expected, rel_tol=1e-9))


class LatencyAndMetrics(unittest.TestCase):
    CONFIG = dict(volatility_window=3, volatility_min_periods=2, volatility_threshold=10,
                  return_type="diff", max_order_age_events=100, taker_fee_bps=0)

    def test_latency_delays_activation_and_rejects_crossing_quotes(self):
        # Decisione al record 1 (bid 100); dopo 2 record il bid e salito a 101 e l'ask a 101,5:
        # la vendita a 101 attraverserebbe il book e viene rifiutata (post-only).
        data = [snap(0), snap(1), snap(2), snap(3, 101, 101.5), snap(4, 101, 101.5)]
        report = run_l2_backtest(data, StrategyConfig(latency_events=2, **self.CONFIG), TouchFillModel())
        self.assertEqual(report.diagnostics["post_only_rejects"], 1)
        # Senza latenza la stessa sequenza esegue subito la quota in acquisto (touch).
        immediate = run_l2_backtest(data, StrategyConfig(**self.CONFIG), TouchFillModel())
        self.assertEqual(immediate.diagnostics["post_only_rejects"], 0)
        self.assertGreater(immediate.summary()["maker_fills"], 0)

    def test_cancel_latency_keeps_order_executable(self):
        # Decisione all'indice 2, attiva al 5; scade al 4 ma la cancellazione ha effetto al 7:
        # il fill price-through dell'indice 6 avviene durante la latenza di cancellazione.
        config = StrategyConfig(latency_events=3, **{**self.CONFIG, "max_order_age_events": 2})
        data = [snap(i) for i in range(6)] + [snap(6, 99, 101)] + [snap(i) for i in range(7, 9)]
        report = run_l2_backtest(data, config, PriceThroughFillModel())
        maker = [f for f in report.fills if f.liquidity == "maker"]
        taker = [f for f in report.fills if f.liquidity == "taker"]
        self.assertEqual([(f.event_index, f.side) for f in maker], [(6, Side.BUY)])
        self.assertEqual([(f.event_index, f.reason, f.price) for f in taker], [(7, "pair_expiry", 100.0)])

    def test_pnl_decomposition_and_markout(self):
        # Quota all'indice 2 (100/101). All'indice 3 l'ask scende a 100: fill cross in acquisto
        # con mid 99,5 (edge -0,5). Il mid scende a 98,5; a fine campione si vende al bid 98.
        data = [snap(0), snap(1), snap(2), snap(3, 99, 100), snap(4, 98, 99), snap(5, 98, 99)]
        report = run_l2_backtest(data, StrategyConfig(**self.CONFIG), CrossFillModel(), markout_records=1)
        s = summarize(report)
        self.assertAlmostEqual(s["final_equity"], -2.0)
        self.assertAlmostEqual(s["maker_edge"], -0.5)
        self.assertAlmostEqual(s["liquidation_cost"], -0.5)
        self.assertAlmostEqual(s["inventory_pnl"], -1.0)
        self.assertAlmostEqual(s["pnl_midpoint_liquidation"], -1.5)
        self.assertAlmostEqual(s["adverse_selection"], -1.0)
        self.assertAlmostEqual(s["markout_bid_mean"], -1.0)
        self.assertEqual((s["pairs_placed"], s["pairs_unilateral"]), (1, 1))

    def test_queue_replay_counts_partial_pairs(self):
        def add(i, oid, side, price, size):
            return OrderEvent(i, i, "T", EventType.ADD, oid, side, price, size)
        events = [add(1, "b0", Side.BUY, 100, 5), add(2, "a0", Side.SELL, 101, 5),
                  add(3, "b1", Side.BUY, 100, 5), add(4, "a1", Side.SELL, 101, 5),
                  OrderEvent(5, 5, "T", EventType.EXECUTE, "b0", size=5),
                  OrderEvent(6, 6, "T", EventType.EXECUTE, "b1", size=5),
                  add(7, "b2", Side.BUY, 100, 5),
                  OrderEvent(8, 8, "T", EventType.EXECUTE, "b2", size=0.4)]
        config = StrategyConfig(**{**self.CONFIG, "warmup_events": 2})
        s = run_order_level_backtest(events, config).summary()
        self.assertEqual(s["pairs_partial"], 1)
        self.assertEqual(s["pairs_unilateral"], 1)

    def test_bootstrap_is_deterministic(self):
        self.assertEqual(bootstrap_mean_ci([1, 2, 3, 4]), bootstrap_mean_ci([1, 2, 3, 4]))
        mean, low, high = bootstrap_mean_ci([1, 2, 3, 4])
        self.assertTrue(low <= mean <= high)


class BookTests(unittest.TestCase):
    def test_incremental_levels_and_snapshot_order(self):
        book = MarketByOrderBook("T")
        for i, (side, price, size) in enumerate([(Side.BUY, 99, 1), (Side.BUY, 100, 2), (Side.BUY, 98, 3),
                                                 (Side.SELL, 102, 1), (Side.SELL, 101, 4)]):
            book.apply(OrderEvent(i, i, "T", EventType.ADD, f"o{i}", side, price, size))
        book.apply(OrderEvent(9, 9, "T", EventType.EXECUTE, "o1", size=0.5))
        s = book.snapshot(9, 9, depth=2)
        self.assertEqual((s.bid_prices, s.bid_sizes, s.ask_prices), ((100, 99), (1.5, 1), (101, 102)))
        book.apply(OrderEvent(10, 10, "T", EventType.DELETE, "o1"))
        self.assertEqual(book.best(Side.BUY), 99)


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_fi2010_segments_split_on_price_jumps(self):
        matrix = np.zeros((40, 6))
        mids = [0.30, 0.301, 0.302, 0.12, 0.121, 0.122]
        matrix[0], matrix[2] = np.array(mids) + 0.0001, np.array(mids) - 0.0001
        self.assertEqual(fi2010_stock_segments(matrix), [(0, 3), (3, 6)])

    def test_hyperliquid_trades_classify_removals(self):
        diffs = [
            {"oid": 1, "coin": "BTC", "side": "B", "px": "100", "raw_book_diff": {"new": {"sz": "1.0"}}},
            {"oid": 2, "coin": "BTC", "side": "A", "px": "101", "raw_book_diff": {"new": {"sz": "2.0"}}},
            {"oid": 1, "coin": "BTC", "side": "B", "px": "100", "raw_book_diff": {"update": {"origSz": "1.0", "newSz": "0.4"}}},
            {"oid": 1, "coin": "BTC", "side": "B", "px": "100", "raw_book_diff": "remove"},
            {"oid": 2, "coin": "BTC", "side": "A", "px": "101", "raw_book_diff": "remove"},
            {"oid": 7, "coin": "BTC", "side": "A", "px": "101", "raw_book_diff": "remove"},
        ]
        trades = [  # aggressore in vendita: il maker e l'acquirente (oid 1)
            {"coin": "BTC", "side": "A", "px": "100", "sz": "0.6", "side_info": [{"oid": 1}, {"oid": 50}]},
            {"coin": "BTC", "side": "A", "px": "100", "sz": "0.4", "side_info": [{"oid": 1}, {"oid": 51}]},
        ]
        d, t = self.dir / "d.jsonl.gz", self.dir / "t.jsonl.gz"
        with gzip.open(d, "wt") as f:
            f.write("\n".join(json.dumps(x) for x in diffs))
        with gzip.open(t, "wt") as f:
            f.write("\n".join(json.dumps(x) for x in trades))
        stats = {}
        events = [(e.event_type, e.order_id, e.size) for e in iter_hyperliquid_book_diffs(d, t, "BTC", stats)]
        self.assertEqual(events, [
            (EventType.ADD, "1", 1.0), (EventType.ADD, "2", 2.0),
            (EventType.EXECUTE, "1", 0.6), (EventType.EXECUTE, "1", 0.4),
            (EventType.DELETE, "2", None),
        ])
        self.assertEqual(stats["unknown_order_diffs"], 1)
        self.assertAlmostEqual(stats["unmatched_trade_volume"], 0.0)


if __name__ == "__main__":
    unittest.main()


def sized(i, bid, ask, bid_size=5.0, ask_size=5.0):
    return BookSnapshot(i, i, "T", (bid,), (bid_size,), (ask,), (ask_size,))


class StrategyVariants(unittest.TestCase):
    BASE = dict(volatility_window=3, volatility_min_periods=2, volatility_threshold=10, return_type="diff",
                max_order_age_events=2, taker_fee_bps=0, tick_size=1.0)

    def test_skew_unwinds_passively_at_improved_price(self):
        # Coppia 100/101 all'indice 2; acquisto eseguito all'indice 3; alla scadenza (4) niente
        # liquidazione: vendita passiva a 102 (ask 103 migliorato di un tick), eseguita all'indice 5.
        data = [sized(i, 100, 101) for i in range(3)] + [sized(3, 99, 100), sized(4, 99, 103), sized(5, 102, 104)]
        config = StrategyConfig(inventory_mode="skew", skew_ticks=1, max_hold_factor=5, **self.BASE)
        report = run_l2_backtest(data, config, CrossFillModel())
        s = summarize(report)
        self.assertEqual([(f.side, f.price) for f in report.fills], [(Side.BUY, 100), (Side.SELL, 102)])
        self.assertEqual((s["taker_fills"], s["pairs_bilateral"], s["unwind_orders"]), (0, 1, 1))
        self.assertAlmostEqual(s["final_equity"], 2.0)
        base = run_l2_backtest(data, StrategyConfig(**self.BASE), CrossFillModel())
        self.assertEqual([f.reason for f in base.fills if f.liquidity == "taker"], ["pair_expiry"])

    def test_skew_liquidates_after_max_hold(self):
        data = [sized(i, 100, 101) for i in range(3)] + [sized(3, 99, 100)] + [sized(i, 99, 103) for i in range(4, 20)]
        config = StrategyConfig(inventory_mode="skew", skew_ticks=1, max_hold_factor=5, **self.BASE)
        report = run_l2_backtest(data, config, CrossFillModel())
        taker = [f for f in report.fills if f.liquidity == "taker"]
        self.assertEqual([f.reason for f in taker], ["max_hold"])
        self.assertGreaterEqual(taker[0].event_index - 3, 10)  # 5 durate da 2 eventi
        self.assertEqual(report.diagnostics["max_hold_liquidations"], 1)

    def test_imbalance_blocks_entry_and_cancels_threatened_quote(self):
        # Book sbilanciato (I = 0,8) agli indici 2-3: nessuna coppia. Bilanciato all'indice 4: coppia.
        # All'indice 5 la pressione in acquisto minaccia la vendita: cancellata, l'acquisto resta.
        data = [sized(i, 100, 101) for i in range(2)] + [sized(i, 100, 101, 9, 1) for i in (2, 3)] \
            + [sized(4, 100, 101), sized(5, 100, 101, 9, 1), sized(6, 100, 101, 9, 1)]
        config = StrategyConfig(imbalance_threshold=0.5, **{**self.BASE, "max_order_age_events": 100})
        report = run_l2_backtest(data, config, PriceThroughFillModel())
        self.assertEqual([p["placed_event"] for p in report.pairs.values()], [4])
        self.assertEqual(report.diagnostics["imbalance_cancels"], 1)
        self.assertEqual(report.diagnostics["imbalance_skips"] >= 2, True)
        self.assertEqual(report.cancelled_orders, 2)  # vendita per imbalance, acquisto a fine campione

    def test_variant_configuration_is_validated(self):
        for kwargs in ({"inventory_mode": "hold"}, {"skew_ticks": 1}, {"imbalance_threshold": 0.0},
                       {"max_hold_factor": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                StrategyConfig(**kwargs)
