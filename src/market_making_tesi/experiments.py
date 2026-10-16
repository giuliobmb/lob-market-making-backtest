"""Protocollo riproducibile sui cinque campioni, condiviso da CLI e notebook."""
from pathlib import Path
import json

import pandas as pd

from .parsers import iter_fi2010_snapshots, iter_hyperliquid_jsonl, iter_itch50_gzip
from .config import StrategyConfig
from .engine import BacktestReport, run_l2_backtest, run_order_level_backtest
from .execution import TouchFillModel, PriceThroughFillModel


def save_report(report: BacktestReport, directory: str | Path, compress: bool = False) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    report.fills_frame().to_csv(directory / ("fills.csv.gz" if compress else "fills.csv"), index=False)
    report.equity_frame().to_csv(directory / "equity.csv", index=False)
    (directory / "summary.json").write_text(
        json.dumps(report.summary(), indent=2, allow_nan=False), encoding="utf-8")


def run_samples(root: str | Path, output: str | Path | None = None):
    root = Path(root)
    samples = root / "data" / "samples"
    config = StrategyConfig.from_json(root / "config" / "small_experiment.json")
    snapshots = list(iter_fi2010_snapshots(samples / "fi2010_real_60.txt"))
    itch = list(iter_itch50_gzip(samples / "itch_aapl_synthetic_10.gz", symbol="AAPL"))
    synthetic = list(iter_hyperliquid_jsonl(samples / "hyperliquid_queue_synthetic_10.jsonl", symbol="BTC"))
    real = list(iter_hyperliquid_jsonl(samples / "hyperliquid_book_diffs_real_500.jsonl", symbol="BTC"))
    cases = {
        "fi2010_touch": ("FI-2010 touch", "L2", len(snapshots), run_l2_backtest(snapshots, config, TouchFillModel())),
        "fi2010_price_through": ("FI-2010 price-through", "L2", len(snapshots), run_l2_backtest(snapshots, config, PriceThroughFillModel())),
        "itch": ("Nasdaq ITCH controllato", "L3", len(itch), run_order_level_backtest(itch, config)),
        "hyperliquid_synthetic": ("Hyperliquid controllato", "L4", len(synthetic), run_order_level_backtest(synthetic, config)),
        # Senza trade le rimozioni non sono classificabili: zero fill per costruzione.
        "hyperliquid_real": ("Hyperliquid reale book diff (non valutabile)", "L4 senza trade", len(real), run_order_level_backtest(real, config)),
    }
    rows, reports = [], {}
    for key, (name, granularity, count, report) in cases.items():
        summary = report.summary()
        rows.append(dict(esperimento=name, granularita=granularity, record=count,
                         ordini_inseriti=summary["placed_orders"], ordini_cancellati=summary["cancelled_orders"],
                         fill_maker=summary["maker_fills"], fill_taker=summary["taker_fills"],
                         fill_totali=summary["total_fills"], equity_finale=summary["final_equity"],
                         commissioni=summary["total_fees"]))
        reports[key] = report
        if output is not None:
            save_report(report, Path(output) / key)
    table = pd.DataFrame(rows)
    if output is not None:
        table.to_csv(Path(output) / "small_experiment_summary.csv", index=False)
    return table, reports
