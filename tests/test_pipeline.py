import gzip
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from market_making_tesi import storage
from market_making_tesi.models import BookSnapshot, EventType, OrderEvent, Side
from market_making_tesi.parsers.itch import SESSION_OPEN_NS, extract_itch_symbols
from market_making_tesi.pipeline import DATASETS, l2_view, select
from market_making_tesi.quality import check_events


def _msg(kind: bytes, locate: int, ts: int, body: bytes) -> bytes:
    payload = kind + locate.to_bytes(2, "big") + (0).to_bytes(2, "big") + ts.to_bytes(6, "big") + body
    return len(payload).to_bytes(2, "big") + payload


def _directory(locate: int, symbol: str) -> bytes:
    return _msg(b"R", locate, 0, symbol.ljust(8).encode() + b"\0" * 20)


def _add(locate, ts, oid, side, shares, price, symbol):
    body = oid.to_bytes(8, "big") + side + shares.to_bytes(4, "big") + symbol.ljust(8).encode() \
        + round(price * 10_000).to_bytes(4, "big")
    return _msg(b"A", locate, ts, body)


def _execute(locate, ts, oid, shares):
    return _msg(b"E", locate, ts, oid.to_bytes(8, "big") + shares.to_bytes(4, "big") + (7).to_bytes(8, "big"))


class StorageAndItch(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_event_roundtrip_and_window(self):
        events = [OrderEvent(1, 1, "T", EventType.ADD, "a", Side.BUY, 10.0, 5.0, metadata={"bootstrap": True}),
                  OrderEvent(5, 2, "T", EventType.REPLACE, "a", new_order_id="b", new_price=10.1, new_size=3.0),
                  OrderEvent(9, 3, "T", EventType.DELETE, "b")]
        path = self.dir / "events.parquet"
        storage.write_events(storage.events_to_columns(events), path)
        back = list(storage.read_events(path, "T"))
        self.assertEqual([(e.timestamp, e.event_type, e.order_id, e.new_order_id) for e in back],
                         [(1, EventType.ADD, "a", None), (5, EventType.REPLACE, "a", "b"), (9, EventType.DELETE, "b", None)])
        self.assertTrue(back[0].metadata.get("bootstrap"))
        self.assertEqual(len(list(storage.read_events(path, "T", stop_timestamp=9))), 2)

    def test_snapshot_roundtrip_keeps_depth(self):
        snaps = [BookSnapshot(1, 1, "T", (10.0, 9.9), (1.0, 2.0), (10.1,), (3.0,))]
        path = self.dir / "snapshots.parquet"
        storage.write_snapshots(snaps, path)
        back = list(storage.read_snapshots(path, "T"))
        self.assertEqual((back[0].bid_prices, back[0].ask_sizes), ((10.0, 9.9), (3.0,)))

    def test_itch_extractor_filters_by_locate_and_marks_preopen(self):
        stream = b"".join([
            _directory(7, "AAPL"), _directory(8, "MSFT"),
            _add(7, SESSION_OPEN_NS - 1, 1, b"B", 100, 200.0, "AAPL"),
            _add(8, SESSION_OPEN_NS + 1, 2, b"S", 50, 100.0, "MSFT"),
            _add(7, SESSION_OPEN_NS + 2, 3, b"S", 100, 200.01, "AAPL"),
            _execute(7, SESSION_OPEN_NS + 3, 1, 40),
        ])
        path = self.dir / "day.gz"
        data = gzip.compress(stream)
        path.write_bytes(data[: len(data) - 3])  # file troncato come i prefissi da 1 GB
        cols, stats = extract_itch_symbols(path, ["AAPL"])
        aapl = cols["AAPL"]
        self.assertEqual(aapl["event_type"], ["ADD", "ADD", "EXECUTE"])
        self.assertEqual(aapl["bootstrap"], [True, False, False])
        self.assertEqual(aapl["price"][:2], [200.0, 200.01])
        self.assertEqual(stats["symbols_found"], 1)

    def test_itch_extractor_streams_batches_and_stops_at_close(self):
        from market_making_tesi.parsers.itch import SESSION_CLOSE_NS
        stream = b"".join([_directory(7, "AAPL")] + [
            _add(7, SESSION_OPEN_NS + i, i + 1, b"B", 100, 200.0, "AAPL") for i in range(5)])
        late = _add(7, SESSION_CLOSE_NS + 1, 99, b"S", 100, 201.0, "AAPL")
        path = self.dir / "day.gz"
        path.write_bytes(gzip.compress(stream) + gzip.compress(late * 2000))
        batches = []
        _, stats = extract_itch_symbols(path, ["AAPL"], chunk_bytes=64, stop_ns=SESSION_CLOSE_NS,
                                        sink=lambda sym, cols: batches.append(list(cols["order_id"])),
                                        flush_rows=2)
        ids = [oid for batch in batches for oid in batch]
        self.assertEqual(ids[:5], ["1", "2", "3", "4", "5"])
        self.assertGreater(len(batches), 1)
        self.assertLess(stats["messages"], 2006)  # decompressione interrotta dopo la chiusura
        _, full = extract_itch_symbols(path, ["AAPL"], chunk_bytes=64)
        self.assertEqual(full["messages"], 2006)  # senza stop: entrambi i membri gzip

    def test_quality_and_l2_view(self):
        events = [OrderEvent(1, 1, "T", EventType.ADD, "a", Side.BUY, 10.0, 5.0, metadata={"bootstrap": True}),
                  OrderEvent(2, 2, "T", EventType.ADD, "b", Side.SELL, 10.1, 5.0),
                  OrderEvent(3, 3, "T", EventType.EXECUTE, "a", size=9.0),
                  OrderEvent(4, 4, "T", EventType.CANCEL, "zz", size=1.0)]
        q = check_events(events)
        self.assertEqual((q["overfill"], q["unknown_order"], q["bootstrap_events"], q["crossed"]), (1, 1, 1, 0))
        view = list(l2_view(iter(events)))
        self.assertEqual([s.event_index for s in view], [2])  # bootstrap esclusi, book vuoto dopo l'esecuzione


class Selection(unittest.TestCase):
    def test_parameters_chosen_on_validation_mean(self):
        validation = pd.DataFrame({
            "dataset": ["itch"] * 4, "param_vol_quantile": [0.9, 0.9, 0.99, 0.99],
            "param_max_order_age_ns": [5, 5, 5, 5], "net_nasdaq_base": [1.0, -3.0, 0.5, 0.0]})
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "results").mkdir()
            chosen = select(Path(tmp), validation)
        self.assertEqual(chosen["itch"]["params"], {"vol_quantile": 0.99, "max_order_age_ns": 5})
        self.assertEqual(chosen["itch"]["metric"], "net_nasdaq_base")

    def test_calibration_precedes_validation(self):
        for spec in DATASETS.values():
            ordered = [*spec.calibration, *spec.validation]
            self.assertEqual(ordered, sorted(ordered))
            self.assertFalse(set(spec.calibration) & set(spec.validation))

    def test_thresholds_come_from_calibration_session_of_same_symbol(self):
        from market_making_tesi.pipeline import _calibration
        rows = pd.DataFrame([
            {"dataset": "itch", "session": "2019-01-30", "symbol": "AAPL", "records": 10, "warmup_events": 0, "threshold_w30_q50": 1.0},
            {"dataset": "itch", "session": "2019-01-30", "symbol": "MSFT", "records": 10, "warmup_events": 0, "threshold_w30_q50": 2.0},
            {"dataset": "itch", "session": "2019-07-30", "symbol": "AAPL", "records": 99, "warmup_events": 0, "threshold_w30_q50": 9.0}])
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "results").mkdir()
            rows.to_csv(Path(tmp) / "results" / "calibration.csv", index=False)
            cal = _calibration(Path(tmp))
        test_row = cal[("itch", "2019-07-30", "AAPL")]
        self.assertEqual((test_row["threshold_w30_q50"], test_row["records"]), (1.0, 99))



class VolatilityFilter(unittest.TestCase):
    def test_quotes_only_in_low_volatility_regime(self):
        from market_making_tesi.config import StrategyConfig
        from market_making_tesi.engine import run_l2_backtest
        from market_making_tesi.execution import CrossFillModel

        def snap(i, bid, ask):
            return BookSnapshot(i, i, "T", (bid,), (1.0,), (ask,), (1.0,))
        # Finestra di 3 rendimenti: il salto all'indice 3 resta nella finestra fino all'indice 5.
        data = [snap(i, 100, 101) for i in range(3)] + [snap(i, 110, 111) for i in range(3, 10)]
        config = StrategyConfig(volatility_window=3, volatility_min_periods=2, volatility_threshold=0.5,
                                return_type="diff", max_order_age_events=100, taker_fee_bps=0)
        report = run_l2_backtest(data, config, CrossFillModel())
        placed = sorted({p["placed_event"] for p in report.pairs.values()})
        self.assertEqual(placed[:2], [2, 6])
        regime = [r["low_volatility"] for r in report.equity_curve]
        self.assertEqual(regime[3:6], [0, 0, 0])
        self.assertEqual(report.diagnostics["halted_records"], 3)

    def test_threshold_follows_window_and_ablation_disables_filter(self):
        from market_making_tesi.pipeline import NO_FILTER, build_config
        root = Path(__file__).resolve().parents[1]
        cal = {"warmup_events": 10, "threshold_w100_q50": 0.1, "threshold_w1000_q50": 0.2}
        params = {"volatility_window": 1000, "vol_quantile": 0.5, "max_order_age_ns": 5}
        config = build_config(root, DATASETS["itch"], cal, params)
        self.assertEqual((config.volatility_window, config.volatility_threshold, config.warmup_events),
                         (1000, 0.2, 10))
        off = build_config(root, DATASETS["itch"], cal, {**params, "volatility_window": 100, "vol_filter": False})
        self.assertEqual((off.volatility_window, off.volatility_threshold), (100, NO_FILTER))


class ClockTimeVolatility(unittest.TestCase):
    def test_matches_resampled_rolling_std(self):
        import random

        import numpy as np

        from market_making_tesi.volatility import VolatilityEstimator
        rng = random.Random(3)
        t, mid, times, mids = 0, 100.0, [], []
        for _ in range(400):  # intensita irregolare: raffiche e intervalli vuoti
            t += rng.choice([1, 1, 1, 2, 7, 25])
            mid += rng.choice([-0.01, 0, 0, 0, 0.01])
            times.append(t)
            mids.append(mid)
        est = VolatilityEstimator(window=8, min_periods=3, return_type="diff", sample_ns=10)
        values = [est.update(m, ts) for m, ts in zip(mids, times)]
        # Riferimento: ultimo midpoint di ogni intervallo chiuso, intervalli vuoti riempiti.
        frame = pd.Series(mids, index=pd.Index(times) // 10).groupby(level=0).last()
        closed = frame.loc[: (times[-1] // 10) - 1]
        grid = closed.reindex(range(closed.index[0], times[-1] // 10)).ffill()
        reference = grid.diff().rolling(8, min_periods=3).std().iloc[-1]
        self.assertAlmostEqual(values[-1], reference, places=12)
        # Un rendimento per intervallo: la stima non cambia dentro lo stesso intervallo.
        same = [v for ts, v in zip(times, values) if ts // 10 == times[-1] // 10]
        self.assertEqual(len(set(same)), 1)


class GroupedBootstrap(unittest.TestCase):
    def setUp(self):
        # Due giornate con tre titoli ciascuna: valori identici dentro la giornata.
        self.values = [1.0, 1.0, 1.0, -1.0, -1.0, -1.0]
        self.groups = ["d1", "d1", "d1", "d2", "d2", "d2"]

    def test_symbols_of_a_session_are_resampled_together(self):
        from market_making_tesi.evaluation import grouped_bootstrap_distribution
        draws = grouped_bootstrap_distribution(self.values, self.groups, n=2000, seed=1)
        # Con le giornate intere le medie possibili sono solo 1, 0 e -1; con righe singole
        # comparirebbero valori come 1/3.
        self.assertTrue(set(np.round(draws, 9)) <= {1.0, 0.0, -1.0})
        self.assertEqual(set(np.round(draws, 9)), {1.0, 0.0, -1.0})

    def test_deterministic_with_fixed_seed(self):
        from market_making_tesi.evaluation import grouped_bootstrap_ci
        first = grouped_bootstrap_ci(self.values, self.groups, seed=7)
        self.assertEqual(first, grouped_bootstrap_ci(self.values, self.groups, seed=7))
        mean, low, high = first
        self.assertEqual(mean, 0.0)
        self.assertTrue(low <= mean <= high)

    def test_unequal_group_sizes_use_row_mean(self):
        from market_making_tesi.evaluation import grouped_bootstrap_distribution
        draws = grouped_bootstrap_distribution([3.0, 3.0, 0.0], ["a", "a", "b"], n=500, seed=0)
        # Campione {a, b}: media delle tre righe = 2; {a, a} = 3; {b, b} = 0.
        self.assertTrue(set(np.round(draws, 9)) <= {3.0, 2.0, 0.0})

    def test_sample_counts_and_fill_rate(self):
        from market_making_tesi.evaluation import order_fill_rate, sample_counts
        frame = pd.DataFrame({"session": ["d1", "d1", "d2"], "pairs_bilateral": [1, 0, 2],
                              "pairs_unilateral": [1, 2, 0], "placed_orders": [6, 4, 4]})
        self.assertEqual(sample_counts(frame), {"n_sessioni_indipendenti": 2, "n_titolo_sessioni": 3})
        self.assertEqual(order_fill_rate(frame).round(3).tolist(), [0.5, 0.5, 1.0])


if __name__ == "__main__":
    unittest.main()
