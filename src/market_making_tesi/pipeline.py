"""Protocollo sperimentale: calibrazione, validazione, test e stress.

* **Sessioni.** Unita indipendenti e mai mescolate: titolo-giorno (FI-2010,
  ITCH) od ora di negoziazione (Hyperliquid).
* **Filtro di volatilita** (parte della strategia): la volatilita e la deviazione
  standard dei rendimenti del midpoint su una finestra mobile di
  ``volatility_window`` record; la strategia quota solo quando e sotto soglia
  (regime di bassa volatilita), altrimenti cancella, liquida e attende.
* **Calibrazione** (prime sessioni in ordine cronologico, per strumento): la
  soglia e il quantile ``vol_quantile`` della volatilita osservata nelle sessioni
  di calibrazione dello stesso strumento, per ogni finestra della griglia, ed e
  applicata alle sessioni successive. Un dataset senza sessioni di calibrazione
  userebbe il primo 20% di ogni sessione, in cui la strategia non quota.
* **Validazione** (sessioni successive alla calibrazione): si sceglie la
  combinazione della griglia (finestra e quantile della volatilita, durata degli
  ordini) con il P&L medio netto piu alto, usando il modello di riferimento.
* **Ablazione:** le sessioni di test sono ripetute senza filtro di volatilita
  per misurarne il contributo.
* **Test** (sessioni successive): la strategia selezionata, identica per tutti i
  modelli di fill, viene simulata con touch, price-through, cross e coda FIFO.
* **Stress (E5):** latenza crescente sul modello a coda; costi come scenari.
* **Varianti (estensione):** inventory skew e filtro di imbalance, con parametri
  scelti in validazione a parita di parametri base, confrontate con la strategia
  base sulle stesse sessioni di test.

Il test set non e mai usato per scegliere parametri. Commissioni e regola di
liquidazione non modificano le decisioni: sono valutate sui fill (vedi
``evaluation``).
"""
from __future__ import annotations

import itertools
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from . import storage
from .config import StrategyConfig
from .engine import BacktestReport, run_l2_backtest, run_order_level_backtest
from .evaluation import fees_from_fills
from .execution import L2_FILL_MODELS
from .models import BookSnapshot, OrderEvent
from .orderbook import MarketByOrderBook
from .volatility import VolatilityEstimator

CALIBRATION_FRACTION = 0.2
QUEUE = "queue"
SECOND = 10**9


@dataclass(frozen=True)
class FeeScenario:
    maker_bps: float = 0.0
    taker_bps: float = 0.0
    maker_per_unit: float = 0.0
    taker_per_unit: float = 0.0


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    kind: str                      # "l2" (solo snapshot) oppure "order" (eventi MBO/L4)
    config: str                    # file in config/
    models: tuple[str, ...]
    reference: str                 # modello usato in validazione
    validation: tuple[str, ...]    # sessioni di validazione
    calibration: tuple[str, ...] = ()  # sessioni di calibrazione delle soglie
    grid: dict = field(default_factory=dict)
    window_end_ns: int | None = None
    markout_records: int = 100
    fees: dict = field(default_factory=dict)
    params_from: str | None = None
    latencies: tuple[dict, ...] = ()
    save_fills_max: int = 200_000


L2_MODELS = ("touch", "price-through", "cross")
# Soglie di "bassa volatilita": si quota solo sotto la mediana, il terzo quartile
# o il 90-esimo percentile della volatilita di calibrazione.
VOL_QUANTILES = [0.50, 0.75, 0.90]
CALIBRATION_QUANTILES = (0.50, 0.75, 0.90, 0.95, 0.99)
NO_FILTER = 1e300  # soglia irraggiungibile: strategia sempre attiva (ablazione)
# Varianti della strategia: griglie di validazione (i parametri base restano quelli scelti).
VARIANTS = {
    "skew": {"inventory_mode": ["skew"], "skew_ticks": [0, 1], "max_hold_factor": [2, 10]},
    "imbalance": {"imbalance_threshold": [0.3, 0.6]},
    "skew_imbalance": {"inventory_mode": ["skew"], "skew_ticks": [0, 1], "max_hold_factor": [2, 10],
                       "imbalance_threshold": [0.3, 0.6]},
}
FLOAT_PARAMS = {"vol_quantile", "imbalance_threshold"}
TEXT_PARAMS = {"inventory_mode"}
NASDAQ_FEES = {"lordo": FeeScenario(),
               "nasdaq_base": FeeScenario(maker_per_unit=-0.0020, taker_per_unit=0.0030),
               "senza_rebate": FeeScenario(taker_per_unit=0.0030)}

DATASETS = {
    "fi2010": DatasetSpec(
        "fi2010", "l2", "fi2010.json", L2_MODELS, "price-through",
        calibration=("day01",), validation=("day02", "day03"),
        grid={"volatility_window": [20, 100], "vol_quantile": VOL_QUANTILES,
              "max_order_age_events": [50, 200]},
        markout_records=10, fees={"lordo": FeeScenario()}),
    "itch": DatasetSpec(
        "itch", "order", "itch.json", (*L2_MODELS, QUEUE), QUEUE,
        calibration=("2019-01-30",), validation=("2019-03-27",),
        # Volatilita in tempo di calendario: midpoint ogni secondo, finestre di 30 s e 2 min.
        grid={"volatility_window": [30, 120], "vol_quantile": VOL_QUANTILES,
              "max_order_age_ns": [5 * SECOND, 30 * SECOND]},
        window_end_ns=57_600 * SECOND,  # 16:00: sessione regolare completa
        markout_records=100, fees=NASDAQ_FEES,
        latencies=({"latency_ns": 0}, {"latency_ns": 10**6}, {"latency_ns": 10**7}, {"latency_ns": 10**8})),
    "hyperliquid": DatasetSpec(
        "hyperliquid", "order", "hyperliquid.json", (*L2_MODELS, QUEUE), QUEUE,
        calibration=("20251201T08",), validation=("20251201T14",),
        grid={"volatility_window": [500, 5000], "vol_quantile": VOL_QUANTILES,
              "max_order_age_events": [1000, 5000]},
        markout_records=1000,
        fees={"lordo": FeeScenario(), "tier0": FeeScenario(1.5, 4.5), "tier_alto": FeeScenario(-0.3, 2.4)},
        latencies=({"latency_events": 0}, {"latency_events": 50}, {"latency_events": 500})),
}


# --- accesso ai dati ---------------------------------------------------------------
def interim(root: Path) -> Path:
    return Path(root) / "data" / "interim"


def session_list(root: Path, spec: DatasetSpec) -> list[tuple[str, str, Path]]:
    return storage.sessions(interim(root), spec.name, "snapshots" if spec.kind == "l2" else "events")


def events(spec: DatasetSpec, path: Path, symbol: str) -> Iterator[OrderEvent]:
    return storage.read_events(path, symbol, stop_timestamp=spec.window_end_ns)


def l2_view(stream: Iterator[OrderEvent], depth: int = 1) -> Iterator[BookSnapshot]:
    """Vista L2 costruita dallo stesso feed order-level (stessi record del replay)."""
    book = MarketByOrderBook()
    for event in stream:
        book.apply(event)
        if event.metadata.get("bootstrap"):
            continue
        snapshot = book.snapshot(event.timestamp, event.event_index, depth=depth)
        if snapshot is not None:
            yield snapshot


def snapshots(spec: DatasetSpec, path: Path, symbol: str) -> Iterator[BookSnapshot]:
    if spec.kind == "l2":
        return storage.read_snapshots(path, symbol)
    return l2_view(events(spec, path, symbol))


# --- calibrazione --------------------------------------------------------------------
def calibrate_session(args: tuple[str, str, str, str]) -> dict:
    root, name, session, symbol = args
    spec = DATASETS[name]
    path = next(p for s, sym, p in session_list(Path(root), spec) if s == session and sym == symbol)
    config = StrategyConfig.from_json(Path(root) / "config" / spec.config)
    pairs = [(s.midpoint, s.timestamp) for s in snapshots(spec, path, symbol)]
    mids = np.fromiter((m for m, _ in pairs), dtype=float, count=len(pairs))
    times = np.fromiter((t for _, t in pairs), dtype=np.int64, count=len(pairs))
    del pairs
    # Con sessioni di calibrazione dedicate le soglie usano l'intera sessione; altrimenti
    # (ripiego) il primo 20%, che diventa riscaldamento senza quotazione.
    own = not spec.calibration
    warmup = int(len(mids) * CALIBRATION_FRACTION) if own else 0
    span = warmup if own else len(mids)
    row = {"dataset": name, "session": session, "symbol": symbol, "records": len(mids),
           "warmup_events": warmup, "mid_first": float(mids[0]), "mid_last": float(mids[-1])}
    for window in windows(spec, config):
        estimator = VolatilityEstimator(window, min(config.volatility_min_periods, window),
                                        config.volatility_method, config.return_type, config.ewma_lambda,
                                        config.volatility_sample_ns)
        sigmas = np.array([v for v in (estimator.update(m, t) for m, t in zip(mids[:span], times[:span]))
                           if v is not None])
        row[f"sigma_median_w{window}"] = float(np.median(sigmas))
        for q in CALIBRATION_QUANTILES:
            row[f"threshold_w{window}_q{round(q * 100)}"] = float(np.quantile(sigmas, q))
    return row


def windows(spec: DatasetSpec, config: StrategyConfig) -> list[int]:
    """Finestre di volatilita da calibrare: quelle della griglia piu quella di default."""
    return sorted({config.volatility_window, *spec.grid.get("volatility_window", [])})


# --- singola simulazione ----------------------------------------------------------------
def peak_memory_mb() -> float:
    try:
        import psutil
        info = psutil.Process().memory_info()
        return getattr(info, "peak_wset", getattr(info, "rss", 0)) / 1e6
    except Exception:  # pragma: no cover - dipende dalla piattaforma
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e3


def build_config(root: Path, spec: DatasetSpec, calibration: dict, params: dict) -> StrategyConfig:
    config = StrategyConfig.from_json(Path(root) / "config" / spec.config)
    overrides = {k: v for k, v in params.items() if k not in ("vol_quantile", "vol_filter")}
    window = int(params.get("volatility_window", config.volatility_window))
    overrides["volatility_window"] = window
    overrides["volatility_min_periods"] = min(config.volatility_min_periods, window)
    threshold = calibration[f"threshold_w{window}_q{round(params['vol_quantile'] * 100)}"]
    if not params.get("vol_filter", True):
        threshold = NO_FILTER
    return replace(config, volatility_threshold=threshold, warmup_events=calibration["warmup_events"],
                   **overrides)


def run_task(task: dict) -> dict:
    """Esegue una simulazione; pensata per un processo dedicato (misura della memoria)."""
    root, spec = Path(task["root"]), DATASETS[task["dataset"]]
    session, symbol, model = task["session"], task["symbol"], task["model"]
    path = next(p for s, sym, p in session_list(root, spec) if s == session and sym == symbol)
    config = build_config(root, spec, task["calibration"], task["params"])
    record_every = max(1, task["calibration"]["records"] // 2000)
    start = time.perf_counter()
    if model == QUEUE:
        report = run_order_level_backtest(events(spec, path, symbol), config, depth=1,
                                          record_every=record_every, record_fills=False,
                                          markout_records=spec.markout_records)
    else:
        report = run_l2_backtest(snapshots(spec, path, symbol), config, L2_FILL_MODELS[model](),
                                 record_every=record_every, record_fills=False,
                                 markout_records=spec.markout_records)
    seconds = time.perf_counter() - start
    summary = report.summary()
    row = {"dataset": spec.name, "stage": task["stage"], "variant": task.get("variant", "base"),
           "session": session, "symbol": symbol,
           "model": model, **{f"param_{k}": v for k, v in task["params"].items()},
           **summary, "seconds": seconds,
           "records_per_second": summary["records"] / seconds if seconds else None,
           "active_share": 1 - summary["halted_records"] / max(1, summary["records"] - config.warmup_events),
           "volatility_threshold": config.volatility_threshold,
           "peak_memory_mb": peak_memory_mb()}
    fills = report.fills_frame()
    for scenario, fee in spec.fees.items():
        cost = fees_from_fills(fills, fee.maker_bps, fee.taker_bps)
        if len(fills):
            unit = np.where(fills["liquidity"] == "maker", fee.maker_per_unit, fee.taker_per_unit)
            cost += float((fills["size"] * unit).sum())
        row[f"fees_{scenario}"] = cost
        row[f"net_{scenario}"] = summary["final_equity"] - cost
    if task.get("output"):
        out = Path(task["output"])
        out.mkdir(parents=True, exist_ok=True)
        report.equity_frame().to_csv(out / "equity.csv", index=False)
        if len(fills) <= spec.save_fills_max:
            fills.to_csv(out / "fills.csv.gz", index=False)
        (out / "summary.json").write_text(json.dumps(row, indent=2, default=str), encoding="utf-8")
    return row


def _pool(tasks: list, fn, workers: int | None) -> list:
    workers = workers or max(1, (os.cpu_count() or 2) - 2)
    with ProcessPoolExecutor(workers, max_tasks_per_child=1) as pool:
        return list(pool.map(fn, tasks))


# --- stadi --------------------------------------------------------------------------
def calibrate(root: Path, datasets: list[str], workers: int | None = None) -> pd.DataFrame:
    tasks = [(str(root), name, session, symbol) for name in datasets
             for session, symbol, _ in session_list(root, DATASETS[name])]
    frame = pd.DataFrame(_pool(tasks, calibrate_session, workers))
    out = Path(root) / "results" / "calibration.csv"
    if out.exists():
        frame = pd.concat([pd.read_csv(out).query("dataset not in @datasets"), frame])
    frame.to_csv(out, index=False)
    return frame


def _calibration(root: Path) -> dict:
    """Per ogni sessione: record e riscaldamento propri, soglie dalle sessioni di calibrazione.

    Le soglie di una sessione sono la media dei quantili delle sessioni di
    calibrazione dello stesso strumento (una sola nei dataset attuali); senza
    sessioni di calibrazione restano quelle stimate sul primo 20% della sessione.
    """
    frame = pd.read_csv(Path(root) / "results" / "calibration.csv")
    thresholds = [c for c in frame.columns if c.startswith("threshold_") or c.startswith("sigma_median")]
    out = {}
    for r in frame.to_dict("records"):
        spec = DATASETS[r["dataset"]]
        if spec.calibration:
            source = frame[(frame.dataset == r["dataset"]) & (frame.symbol == r["symbol"])
                           & frame.session.isin(spec.calibration)]
            r.update(source[thresholds].mean().to_dict())
            r["threshold_source"] = ",".join(spec.calibration)
        else:
            r["threshold_source"] = f"{r['session']} (primo 20%)"
        out[(r["dataset"], r["session"], r["symbol"])] = r
    return out


def _grid(spec: DatasetSpec) -> list[dict]:
    keys = list(spec.grid)
    return [dict(zip(keys, values)) for values in itertools.product(*spec.grid.values())]


def selection_metric(spec: DatasetSpec) -> str:
    base = [name for name in spec.fees if name != "lordo"]
    return f"net_{base[0]}" if base else "final_equity"


def validate(root: Path, datasets: list[str], workers: int | None = None) -> pd.DataFrame:
    cal = _calibration(root)
    tasks = []
    for name in datasets:
        spec = DATASETS[name]
        for session, symbol, _ in session_list(root, spec):
            if session not in spec.validation:
                continue
            for params in _grid(spec):
                tasks.append({"root": str(root), "dataset": name, "stage": "validation", "session": session,
                              "symbol": symbol, "model": spec.reference, "params": params,
                              "calibration": cal[(name, session, symbol)]})
    frame = pd.DataFrame(_pool(tasks, run_task, workers))
    out = Path(root) / "results" / "validation.csv"
    if out.exists():
        frame = pd.concat([pd.read_csv(out).query("dataset not in @datasets"), frame])
    frame.to_csv(out, index=False)
    select(root, frame)
    return frame


def select(root: Path, validation: pd.DataFrame) -> dict:
    """Combinazione con P&L netto medio piu alto in validazione, per dataset."""
    path = Path(root) / "results" / "selected_params.json"
    chosen = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    for name, group in validation.groupby("dataset"):
        spec = DATASETS[name]
        metric = selection_metric(spec)
        # Solo i parametri della griglia di questo dataset (le altre colonne sono vuote).
        params = [c for c in group.columns if c.startswith("param_") and group[c].notna().all()]
        means = group.groupby(params)[metric].mean().sort_values(ascending=False)
        best = dict(zip([p.removeprefix("param_") for p in params],
                        means.index[0] if len(params) > 1 else [means.index[0]]))
        chosen[name] = {"params": {k: _typed(k, v) for k, v in best.items()},
                        "metric": metric, "validation_mean": float(means.iloc[0]),
                        "grid": {str(k): float(v) for k, v in means.items()}}
    path.write_text(json.dumps(chosen, indent=2), encoding="utf-8")
    return chosen


def _typed(key: str, value):
    if key in TEXT_PARAMS:
        return str(value)
    return float(value) if key in FLOAT_PARAMS else int(value)


def _selected(root: Path, spec: DatasetSpec) -> dict:
    chosen = json.loads((Path(root) / "results" / "selected_params.json").read_text(encoding="utf-8"))
    return dict(chosen[spec.params_from or spec.name]["params"])


def test(root: Path, datasets: list[str], workers: int | None = None) -> pd.DataFrame:
    cal = _calibration(root)
    tasks = []
    for name in datasets:
        spec = DATASETS[name]
        params = _selected(root, spec)
        for session, symbol, _ in session_list(root, spec):
            if session in spec.validation or session in spec.calibration:
                continue
            for model in spec.models:
                tasks.append({"root": str(root), "dataset": name, "stage": "test", "session": session,
                              "symbol": symbol, "model": model, "params": params,
                              "calibration": cal[(name, session, symbol)],
                              "output": str(Path(root) / "results" / "runs" / name / f"{session}_{symbol}" / model)})
    frame = pd.DataFrame(_pool(tasks, run_task, workers))
    out = Path(root) / "results" / "test.csv"
    if out.exists():
        frame = pd.concat([pd.read_csv(out).query("dataset not in @datasets"), frame])
    frame.to_csv(out, index=False)
    return frame


def ablation(root: Path, datasets: list[str], workers: int | None = None) -> pd.DataFrame:
    """Sessioni di test senza filtro di volatilita (strategia sempre attiva)."""
    cal = _calibration(root)
    tasks = []
    for name in datasets:
        spec = DATASETS[name]
        params = {**_selected(root, spec), "vol_filter": False}
        for session, symbol, _ in session_list(root, spec):
            if session in spec.validation or session in spec.calibration:
                continue
            for model in spec.models:
                tasks.append({"root": str(root), "dataset": name, "stage": "ablation", "session": session,
                              "symbol": symbol, "model": model, "params": params,
                              "calibration": cal[(name, session, symbol)]})
    frame = pd.DataFrame(_pool(tasks, run_task, workers))
    out = Path(root) / "results" / "ablation.csv"
    if out.exists():
        frame = pd.concat([pd.read_csv(out).query("dataset not in @datasets"), frame])
    frame.to_csv(out, index=False)
    return frame


def stress(root: Path, datasets: list[str], workers: int | None = None) -> pd.DataFrame:
    cal = _calibration(root)
    tasks = []
    for name in datasets:
        spec = DATASETS[name]
        for session, symbol, _ in session_list(root, spec):
            if session in spec.validation or session in spec.calibration:
                continue
            for latency in spec.latencies:
                tasks.append({"root": str(root), "dataset": name, "stage": "stress", "session": session,
                              "symbol": symbol, "model": QUEUE, "params": {**_selected(root, spec), **latency},
                              "calibration": cal[(name, session, symbol)]})
    frame = pd.DataFrame(_pool(tasks, run_task, workers))
    out = Path(root) / "results" / "stress.csv"
    if out.exists():
        frame = pd.concat([pd.read_csv(out).query("dataset not in @datasets"), frame])
    frame.to_csv(out, index=False)
    return frame


def quality_session(args: tuple[str, str, str, str]) -> dict:
    from .quality import check_events, check_snapshots
    root, name, session, symbol = args
    spec = DATASETS[name]
    path = next(p for s, sym, p in session_list(Path(root), spec) if s == session and sym == symbol)
    start = time.perf_counter()
    if spec.kind == "l2":
        row = check_snapshots(storage.read_snapshots(path, symbol))
    else:
        row = check_events(events(spec, path, symbol))
    return {"dataset": name, "session": session, "symbol": symbol, **row,
            "seconds": time.perf_counter() - start}


def quality(root: Path, datasets: list[str], workers: int | None = None) -> pd.DataFrame:
    tasks = [(str(root), name, session, symbol) for name in datasets
             for session, symbol, _ in session_list(root, DATASETS[name])]
    frame = pd.DataFrame(_pool(tasks, quality_session, workers))
    out = Path(root) / "results" / "quality.csv"
    if out.exists():
        frame = pd.concat([pd.read_csv(out).query("dataset not in @datasets"), frame])
    frame.to_csv(out, index=False)
    return frame


def _variant_grid(name: str) -> list[dict]:
    grid = VARIANTS[name]
    return [dict(zip(grid, values)) for values in itertools.product(*grid.values())]


def variants(root: Path, datasets: list[str], workers: int | None = None) -> pd.DataFrame:
    """Varianti: selezione in validazione (modello di riferimento) e test con tutti i modelli."""
    cal = _calibration(root)
    path = Path(root) / "results" / "selected_params.json"
    tasks = []
    for name in datasets:
        spec = DATASETS[name]
        if not spec.validation:
            continue
        base = _selected(root, spec)
        for session, symbol, _ in session_list(root, spec):
            if session not in spec.validation:
                continue
            for variant in VARIANTS:
                for combo in _variant_grid(variant):
                    tasks.append({"root": str(root), "dataset": name, "stage": "variant_validation",
                                  "variant": variant, "session": session, "symbol": symbol,
                                  "model": spec.reference, "params": {**base, **combo},
                                  "calibration": cal[(name, session, symbol)]})
    validation = pd.DataFrame(_pool(tasks, run_task, workers)) if tasks else pd.DataFrame()
    out = Path(root) / "results" / "variants_validation.csv"
    if out.exists():
        old = pd.read_csv(out)
        validation = pd.concat([old[~old.dataset.isin(datasets)], validation])
    validation.to_csv(out, index=False)

    chosen = json.loads(path.read_text(encoding="utf-8"))
    store = chosen.setdefault("variants", {})
    for (name, variant), group in validation[validation.dataset.isin(datasets)].groupby(["dataset", "variant"]):
        metric = selection_metric(DATASETS[name])
        keys = [f"param_{k}" for k in VARIANTS[variant]]
        means = group.groupby(keys)[metric].mean().sort_values(ascending=False)
        best = means.index[0] if len(keys) > 1 else (means.index[0],)
        store.setdefault(name, {})[variant] = {
            "params": {k.removeprefix("param_"): _typed(k.removeprefix("param_"), v) for k, v in zip(keys, best)},
            "validation_mean": float(means.iloc[0]), "metric": metric}
    path.write_text(json.dumps(chosen, indent=2), encoding="utf-8")

    tasks = []
    for name in datasets:
        spec = DATASETS[name]
        source = spec.params_from or name
        if source not in store:
            continue
        base = _selected(root, spec)
        for session, symbol, _ in session_list(root, spec):
            if session in spec.validation or session in spec.calibration:
                continue
            for variant, info in store[source].items():
                for model in spec.models:
                    tasks.append({"root": str(root), "dataset": name, "stage": "variant_test", "variant": variant,
                                  "session": session, "symbol": symbol, "model": model,
                                  "params": {**base, **info["params"]},
                                  "calibration": cal[(name, session, symbol)]})
    frame = pd.DataFrame(_pool(tasks, run_task, workers))
    out = Path(root) / "results" / "variants.csv"
    if out.exists():
        old = pd.read_csv(out)
        frame = pd.concat([old[~old.dataset.isin(datasets)], frame])
    frame.to_csv(out, index=False)
    return frame


def variant_stress(root: Path, datasets: list[str], workers: int | None = None) -> pd.DataFrame:
    """Latenza sulla variante migliore in validazione (coda FIFO, sessioni di test)."""
    cal = _calibration(root)
    store = json.loads((Path(root) / "results" / "selected_params.json").read_text(encoding="utf-8"))["variants"]
    tasks = []
    for name in datasets:
        spec = DATASETS[name]
        if not spec.latencies or name not in store:
            continue
        variant, info = max(store[name].items(), key=lambda item: item[1]["validation_mean"])
        base = _selected(root, spec)
        for session, symbol, _ in session_list(root, spec):
            if session in spec.validation or session in spec.calibration:
                continue
            for latency in spec.latencies:
                tasks.append({"root": str(root), "dataset": name, "stage": "variant_stress", "variant": variant,
                              "session": session, "symbol": symbol, "model": QUEUE,
                              "params": {**base, **info["params"], **latency},
                              "calibration": cal[(name, session, symbol)]})
    frame = pd.DataFrame(_pool(tasks, run_task, workers))
    out = Path(root) / "results" / "variants_stress.csv"
    if out.exists():
        old = pd.read_csv(out)
        frame = pd.concat([old[~old.dataset.isin(datasets)], frame])
    frame.to_csv(out, index=False)
    return frame


STAGES = {"quality": quality, "calibrate": calibrate, "validate": validate, "test": test,
          "ablation": ablation, "stress": stress, "variants": variants, "variant_stress": variant_stress}


def run(root: Path, stages: list[str], datasets: list[str], workers: int | None = None) -> None:
    for stage in stages:
        names = [d for d in datasets if stage not in ("stress", "variant_stress") or DATASETS[d].latencies]
        names = [d for d in names if stage != "validate" or DATASETS[d].validation]
        start = time.perf_counter()
        STAGES[stage](Path(root), names, workers)
        print(f"{stage}: {', '.join(names)} in {time.perf_counter() - start:.0f}s", flush=True)
