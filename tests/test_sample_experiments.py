from __future__ import annotations

import unittest
from pathlib import Path

from market_making_tesi.parsers import (
    iter_fi2010_snapshots,
    iter_hyperliquid_jsonl,
    iter_itch50_gzip,
)
from market_making_tesi.config import StrategyConfig
from market_making_tesi.engine import run_l2_backtest, run_order_level_backtest
from market_making_tesi.execution import PriceThroughFillModel, TouchFillModel
from market_making_tesi.models import EventType


ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "data" / "samples"
CONFIG = StrategyConfig.from_json(ROOT / "config" / "small_experiment.json")


class SmallSampleExperiments(unittest.TestCase):
    def test_fi2010_real_sample_and_execution_bias(self) -> None:
        snapshots = list(iter_fi2010_snapshots(SAMPLES / "fi2010_real_60.txt"))

        self.assertEqual(len(snapshots), 60)
        self.assertTrue(all(len(snapshot.bid_prices) == 10 for snapshot in snapshots))
        self.assertTrue(all(len(snapshot.ask_prices) == 10 for snapshot in snapshots))
        self.assertTrue(all(snapshot.best_bid < snapshot.best_ask for snapshot in snapshots))

        touch = run_l2_backtest(snapshots, CONFIG, TouchFillModel()).summary()
        conservative = run_l2_backtest(
            snapshots, CONFIG, PriceThroughFillModel()
        ).summary()

        self.assertGreater(touch["maker_fills"], conservative["maker_fills"])
        self.assertEqual(touch["maker_fills"], 22)
        self.assertEqual(conservative["maker_fills"], 5)

    def test_itch_controlled_queue_replay(self) -> None:
        events = list(
            iter_itch50_gzip(
                SAMPLES / "itch_aapl_synthetic_10.gz", symbol="AAPL"
            )
        )
        self.assertEqual(len(events), 10)
        self.assertEqual(sum(event.event_type is EventType.EXECUTE for event in events), 4)

        summary = run_order_level_backtest(events, CONFIG).summary()
        self.assertEqual(summary["maker_fills"], 2)
        self.assertAlmostEqual(summary["final_equity"], 0.01)

    def test_hyperliquid_controlled_queue_replay(self) -> None:
        events = list(
            iter_hyperliquid_jsonl(
                SAMPLES / "hyperliquid_queue_synthetic_10.jsonl", symbol="BTC"
            )
        )
        self.assertEqual(len(events), 10)
        self.assertEqual(sum(event.event_type is EventType.EXECUTE for event in events), 4)

        summary = run_order_level_backtest(events, CONFIG).summary()
        self.assertEqual(summary["maker_fills"], 2)
        self.assertAlmostEqual(summary["final_equity"], 0.1)

    def test_hyperliquid_real_book_diffs_do_not_invent_fills(self) -> None:
        events = list(
            iter_hyperliquid_jsonl(
                SAMPLES / "hyperliquid_book_diffs_real_500.jsonl", symbol="BTC"
            )
        )
        self.assertEqual(len(events), 500)
        self.assertFalse(any(event.event_type is EventType.EXECUTE for event in events))

        summary = run_order_level_backtest(events, CONFIG).summary()
        self.assertEqual(summary["maker_fills"], 0)
        self.assertAlmostEqual(summary["final_equity"], 0.0)


if __name__ == "__main__":
    unittest.main()
