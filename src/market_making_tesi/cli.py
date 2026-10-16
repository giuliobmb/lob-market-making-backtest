import argparse
from pathlib import Path

from .config import StrategyConfig
from .engine import run_l2_backtest, run_order_level_backtest
from .execution import L2_FILL_MODELS
from .experiments import run_samples, save_report
from .parsers import iter_fi2010_snapshots, iter_hyperliquid_jsonl, iter_itch50_gzip

DATASETS = ["fi2010", "itch", "hyperliquid"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest di market making su dati L2, L3 e L4")
    sub = parser.add_subparsers(dest="mode", required=True)
    samples = sub.add_parser("samples", help="Controlli di correttezza sui campioni piccoli")
    samples.add_argument("--root", type=Path, default=Path.cwd())
    samples.add_argument("--output", type=Path, default=Path("results/samples"))
    ingest = sub.add_parser("ingest", help="Dati grezzi -> Parquet in data/interim")
    ingest.add_argument("--root", type=Path, default=Path.cwd())
    ingest.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    pipe = sub.add_parser("pipeline", help="Calibrazione, validazione, test e stress")
    pipe.add_argument("--root", type=Path, default=Path.cwd())
    stages = ["quality", "calibrate", "validate", "test", "ablation", "stress", "variants", "variant_stress"]
    pipe.add_argument("--stages", nargs="+", default=stages, choices=stages)
    pipe.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    pipe.add_argument("--workers", type=int, default=None)
    for mode in ("fi2010", "itch", "hyperliquid"):
        p = sub.add_parser(mode, help=f"Singolo backtest su un file {mode}")
        p.add_argument("input", type=Path)
        p.add_argument("--config", type=Path, required=True)
        p.add_argument("--symbol", required=True)
        p.add_argument("--output", type=Path, required=True)
        if mode == "fi2010":
            p.add_argument("--fill-model", choices=tuple(L2_FILL_MODELS), default="price-through")
    args = parser.parse_args()

    if args.mode == "samples":
        table, _ = run_samples(args.root, args.output)
        print(table.to_string(index=False))
        return
    if args.mode == "ingest":
        from .ingest import ingest_all
        ingest_all(args.root, args.datasets)
        return
    if args.mode == "pipeline":
        from .pipeline import run
        run(args.root, args.stages, args.datasets, args.workers)
        return
    config = StrategyConfig.from_json(args.config)
    if args.mode == "fi2010":
        report = run_l2_backtest(iter_fi2010_snapshots(args.input, symbol=args.symbol), config,
                                 L2_FILL_MODELS[args.fill_model]())
    else:
        adapter = iter_itch50_gzip if args.mode == "itch" else iter_hyperliquid_jsonl
        report = run_order_level_backtest(adapter(args.input, symbol=args.symbol), config)
    save_report(report, args.output)
    print(report.summary())


if __name__ == "__main__":
    main()
