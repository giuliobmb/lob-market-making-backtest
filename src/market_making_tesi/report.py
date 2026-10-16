"""Supporto ai notebook: percorsi, palette, formattazione e tabelle ricorrenti."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .evaluation import session_statistics

# Palette categoriale validata (ordine fisso): il colore segue il modello, mai il rango.
COLORS = {"touch": "#2a78d6", "price-through": "#eb6834", "cross": "#1baf7a", "queue": "#eda100"}
LABELS = {"touch": "touch", "price-through": "price-through", "cross": "cross", "queue": "coda FIFO"}
MODEL_ORDER = ["touch", "price-through", "cross", "queue"]
DATASET_LABELS = {"fi2010": "FI-2010", "itch": "Nasdaq ITCH", "hyperliquid": "Hyperliquid BTC"}
INK = "#52514e"


def find_root() -> Path:
    for path in (Path.cwd(), *Path.cwd().parents):
        if (path / "src" / "market_making_tesi").is_dir():
            if str(path / "src") not in sys.path:
                sys.path.insert(0, str(path / "src"))
            return path
    raise RuntimeError("Aprire il notebook dalla cartella del progetto")


def setup_plots() -> None:
    plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.15, "font.size": 10, "lines.linewidth": 2,
                         "axes.titleweight": "bold", "axes.titlesize": 11})


def it(value, decimals: int = 0) -> str:
    """Numero in formato italiano: punto per le migliaia, virgola decimale."""
    if value is None or value != value:
        return "n.d."
    return f"{value:,.{decimals}f}".replace(",", "_").replace(".", ",").replace("_", ".")


def pct(value, decimals: int = 1) -> str:
    return "n.d." if value is None or value != value else f"{it(100 * value, decimals)}%"


def load(root: Path, name: str) -> pd.DataFrame:
    return pd.read_csv(Path(root) / "results" / f"{name}.csv")


def load_json(root: Path, name: str) -> dict:
    return json.loads((Path(root) / "results" / f"{name}.json").read_text(encoding="utf-8"))


def model_table(test: pd.DataFrame, dataset: str, value: str = "final_equity") -> pd.DataFrame:
    """Statistiche per sessione (media, IC bootstrap, quota positive) per modello."""
    frame = test[test.dataset == dataset]
    stats = session_statistics(frame, value, ["model"]).set_index("model")
    order = [m for m in MODEL_ORDER if m in stats.index]
    stats = stats.loc[order]
    stats.index = [LABELS[m] for m in order]
    return stats


def fill_bias(test: pd.DataFrame, dataset: str, reference: str = "queue") -> pd.DataFrame:
    """Rapporto tra fill maker di ogni regola L2 e del replay a coda, per sessione."""
    frame = test[test.dataset == dataset]
    pivot = frame.pivot_table(index=["session", "symbol"], columns="model", values="maker_fills")
    ratios = pivot.div(pivot[reference].replace(0, float("nan")), axis=0)
    return ratios[[m for m in MODEL_ORDER if m in ratios]]


DECOMPOSITION = ["maker_edge", "liquidation_cost", "inventory_pnl"]
DECOMPOSITION_LABELS = {"maker_edge": "edge maker", "liquidation_cost": "costo di liquidazione",
                        "inventory_pnl": "P&L di inventario", "fees": "commissioni"}


def decomposition(test: pd.DataFrame, dataset: str, fee_column: str | None = None) -> pd.DataFrame:
    frame = test[test.dataset == dataset]
    out = frame.groupby("model")[DECOMPOSITION + ["final_equity", "adverse_selection"]].mean()
    if fee_column:
        out["fees"] = -frame.groupby("model")[fee_column].mean()
        out["net"] = out["final_equity"] + out["fees"]
    order = [m for m in MODEL_ORDER if m in out.index]
    out = out.loc[order]
    out.index = [LABELS[m] for m in order]
    return out


def fill_rates(test: pd.DataFrame, dataset: str) -> pd.DataFrame:
    frame = test[test.dataset == dataset]
    sums = frame.groupby("model")[["pairs_placed", "pairs_bilateral", "pairs_unilateral",
                                   "pairs_unfilled", "pairs_partial"]].sum()
    rates = pd.DataFrame({
        "coppie": sums.pairs_placed,
        "bilaterali": sums.pairs_bilateral / sums.pairs_placed,
        "unilaterali": sums.pairs_unilateral / sums.pairs_placed,
        "senza fill": sums.pairs_unfilled / sums.pairs_placed,
        "con fill parziale": sums.pairs_partial / sums.pairs_placed,
    })
    order = [m for m in MODEL_ORDER if m in rates.index]
    rates = rates.loc[order]
    rates.index = [LABELS[m] for m in order]
    return rates


def bar_by_model(ax, stats: pd.DataFrame, title: str, ylabel: str) -> None:
    """Media per modello con intervallo bootstrap al 95%."""
    colors = [COLORS[next(k for k, v in LABELS.items() if v == name)] for name in stats.index]
    ax.bar(stats.index, stats["media"], color=colors, width=0.6)
    ax.errorbar(range(len(stats)), stats["media"],
                yerr=[stats["media"] - stats["ic95_basso"], stats["ic95_alto"] - stats["media"]],
                fmt="none", ecolor=INK, capsize=4, linewidth=1)
    ax.axhline(0, color=INK, linewidth=0.8)
    ax.set(title=title, ylabel=ylabel)
    ax.grid(axis="x", visible=False)


def validation_grid(validation: pd.DataFrame, dataset: str, metric: str) -> pd.DataFrame:
    """P&L medio di validazione per finestra e quantile (righe) e durata degli ordini (colonne)."""
    frame = validation[validation.dataset == dataset]
    age = next(c for c in ("param_max_order_age_ns", "param_max_order_age_events") if frame[c].notna().all())
    grid = frame.pivot_table(index=["param_volatility_window", "param_vol_quantile"], columns=age,
                             values=metric, aggfunc="mean")
    grid.index.names = ["finestra (record)", "quantile soglia"]
    grid.columns = [f"durata {int(c / 1e9)} s" if "ns" in age else f"durata {int(c)} eventi" for c in grid.columns]
    return grid


def filter_effect(test: pd.DataFrame, ablation: pd.DataFrame, dataset: str, value: str) -> pd.DataFrame:
    """Confronto appaiato per sessione: strategia con filtro di volatilità e sempre attiva."""
    from .evaluation import bootstrap_mean_ci
    keys = ["session", "symbol", "model"]
    on = test[test.dataset == dataset].set_index(keys)
    off = ablation[ablation.dataset == dataset].set_index(keys)
    rows = []
    for model in [m for m in MODEL_ORDER if m in on.index.get_level_values("model")]:
        a = on.xs(model, level="model")
        b = off.xs(model, level="model").reindex(a.index)
        mean, low, high = bootstrap_mean_ci(a[value] - b[value])
        rows.append({"modello": LABELS[model], "sessioni": len(a), "con filtro": a[value].mean(),
                     "sempre attiva": b[value].mean(),
                     "differenza": mean, "IC95 basso": low, "IC95 alto": high,
                     "sessioni migliori con filtro": float((a[value] > b[value]).mean()),
                     "quota record attivi": a["active_share"].mean(),
                     "fill con filtro": a["maker_fills"].mean(), "fill sempre attiva": b["maker_fills"].mean(),
                     "liquidazioni per volatilità": a["volatility_liquidations"].mean()})
    return pd.DataFrame(rows).set_index("modello")


def plot_regime(axes, equity: pd.DataFrame, threshold: float, x: pd.Series,
                warmup_x: float | None = None, xlabel: str = "") -> None:
    """Midpoint e volatilità stimata, con soglia e periodi di sospensione ombreggiati."""
    low = equity["low_volatility"].fillna(0).astype(bool).to_numpy()
    for ax in axes:
        ax.fill_between(x, 0, 1, where=~low, transform=ax.get_xaxis_transform(),
                        color=INK, alpha=0.12, linewidth=0, step="post")
    axes[0].plot(x, equity["midpoint"], color=COLORS["touch"], linewidth=1)
    axes[0].set(ylabel="Midpoint")
    axes[1].plot(x, equity["volatility"], color=COLORS["queue"], linewidth=1)
    axes[1].axhline(threshold, color=INK, linewidth=1, linestyle="--")
    axes[1].text(x.iloc[-1], threshold, "soglia ", va="bottom", ha="right", color=INK)
    axes[1].set(ylabel="Volatilità stimata", xlabel=xlabel)
    if warmup_x is not None:
        for ax in axes:
            ax.axvline(warmup_x, color=INK, linewidth=0.8)
        axes[0].text(warmup_x, 1, " fine calibrazione", va="top", color=INK,
                     transform=axes[0].get_xaxis_transform())


VARIANT_LABELS = {"base": "base", "skew": "inventory skew", "imbalance": "filtro di imbalance",
                  "skew_imbalance": "skew + imbalance"}


def variant_effect(test: pd.DataFrame, variants: pd.DataFrame, dataset: str, value: str,
                   model: str = "queue", fee_column: str | None = None) -> pd.DataFrame:
    """Varianti contro strategia base sulle stesse sessioni (differenza appaiata, IC bootstrap)."""
    from .evaluation import bootstrap_mean_ci
    keys = ["session", "symbol"]
    base = test[(test.dataset == dataset) & (test.model == model)].set_index(keys)
    rows = []
    for name in VARIANT_LABELS:
        if name == "base":
            frame = base
        else:
            frame = variants[(variants.dataset == dataset) & (variants.model == model)
                             & (variants.variant == name)].set_index(keys)
            if frame.empty:
                continue
        frame = frame.reindex(base.index)
        mean, low, high = bootstrap_mean_ci(frame[value])
        diff, dlow, dhigh = bootstrap_mean_ci(frame[value] - base[value])
        fees = -frame[fee_column].mean() if fee_column else 0.0
        rows.append({
            "variante": VARIANT_LABELS[name], "sessioni": len(frame), "P&L netto medio": mean,
            "IC95 basso": low, "IC95 alto": high, "sessioni positive": float((frame[value] > 0).mean()),
            "differenza vs base": diff if name != "base" else 0.0,
            "diff IC95 basso": dlow if name != "base" else 0.0, "diff IC95 alto": dhigh if name != "base" else 0.0,
            "edge maker": frame["maker_edge"].mean(), "costo di liquidazione": frame["liquidation_cost"].mean(),
            "P&L di inventario": frame["inventory_pnl"].mean(), "commissioni": fees,
            "fill maker": frame["maker_fills"].mean(), "coppie bilaterali": frame["fill_rate_pairs"].mean(),
            "liquidazioni a mercato": frame["taker_fills"].mean(),
        })
    return pd.DataFrame(rows).set_index("variante")


# --- presentazione leggibile ------------------------------------------------------------------
UNITS = {"fi2010": "osservazioni", "itch": "secondi", "hyperliquid": "eventi"}


def show_box(title: str, lines: list[str]) -> None:
    """Riquadro di sintesi in testa ai notebook."""
    from IPython.display import Markdown, display
    body = "\n".join(f"> - {line}" for line in lines)
    display(Markdown(f"> **{title}**\n>\n{body}"))


def ci_text(mean: float, low: float, high: float, decimals: int = 0) -> str:
    """Media con intervallo di confidenza: "-541 (-660; -435)"."""
    return f"{it(mean, decimals)} ({it(low, decimals)}; {it(high, decimals)})"


def params_text(dataset: str, params: dict) -> str:
    """Parametri della strategia descritti a parole."""
    unit = UNITS.get(dataset, "record")
    parts = []
    if "volatility_window" in params:
        parts.append(f"volatilità su una finestra di {it(params['volatility_window'])} {unit}")
    if "vol_quantile" in params:
        parts.append(f"si opera solo sotto il {round(100 * params['vol_quantile'])}° percentile della volatilità di calibrazione")
    if "max_order_age_ns" in params:
        parts.append(f"ogni quota resta nel book al massimo {it(params['max_order_age_ns'] / 1e9)} s")
    if "max_order_age_events" in params:
        parts.append(f"ogni quota resta nel book al massimo {it(params['max_order_age_events'])} {unit}")
    if params.get("inventory_mode") == "skew":
        ticks = params.get("skew_ticks", 0)
        where = "al miglior prezzo corrente" if ticks == 0 else f"migliorate di {ticks} tick"
        parts.append(f"inventario smaltito con quote passive {where}, liquidazione a mercato solo dopo "
                     f"{params.get('max_hold_factor')} volte la durata di una quota")
    if params.get("imbalance_threshold") is not None:
        parts.append(f"si quota solo con imbalance |I| ≤ {it(params['imbalance_threshold'], 1)}")
    return "; ".join(parts)


def results_table(stats: pd.DataFrame, decimals: int = 0) -> pd.DataFrame:
    """Da ``model_table``: P&L medio con IC e quota di sessioni positive."""
    return pd.DataFrame({
        "P&L medio per sessione (IC 95%)": [ci_text(r.media, r.ic95_basso, r.ic95_alto, decimals)
                                            for r in stats.itertuples()],
        "sessioni positive": [pct(r.quota_positive, 0) for r in stats.itertuples()],
        "sessioni": [int(r.sessioni) for r in stats.itertuples()],
    }, index=stats.index)


def decomposition_table(dec: pd.DataFrame, decimals: int = 2) -> pd.DataFrame:
    """Da ``decomposition``: da dove viene il P&L, con nomi espliciti."""
    out = pd.DataFrame(index=dec.index)
    out["spread incassato (edge maker)"] = dec["maker_edge"]
    out["costo di liquidazione"] = dec["liquidation_cost"]
    out["movimento del prezzo sull'inventario"] = dec["inventory_pnl"]
    out["= P&L lordo"] = dec["final_equity"]
    if "fees" in dec:
        out["commissioni e rebate"] = dec["fees"]
        out["= P&L netto"] = dec["net"]
    return out.map(lambda v: it(v, decimals))


def filter_table(effect: pd.DataFrame, decimals: int = 1) -> pd.DataFrame:
    """Da ``filter_effect``: con filtro di volatilità contro strategia sempre attiva."""
    return pd.DataFrame({
        "con filtro": effect["con filtro"].map(lambda v: it(v, decimals)),
        "sempre attiva": effect["sempre attiva"].map(lambda v: it(v, decimals)),
        "differenza (IC 95%)": [ci_text(r["differenza"], r["IC95 basso"], r["IC95 alto"], decimals)
                                for _, r in effect.iterrows()],
        "sessioni migliori con filtro": effect["sessioni migliori con filtro"].map(lambda v: pct(v, 0)),
        "tempo in cui opera": effect["quota record attivi"].map(lambda v: pct(v, 0)),
    })


def variants_table(effect: pd.DataFrame, decimals: int = 1) -> pd.DataFrame:
    """Da ``variant_effect``: varianti contro strategia base."""
    return pd.DataFrame({
        "P&L netto medio (IC 95%)": [ci_text(r["P&L netto medio"], r["IC95 basso"], r["IC95 alto"], decimals)
                                     for _, r in effect.iterrows()],
        "sessioni positive": effect["sessioni positive"].map(lambda v: pct(v, 0)),
        "differenza vs base (IC 95%)": ["—" if name == "base" else
                                        ci_text(r["differenza vs base"], r["diff IC95 basso"], r["diff IC95 alto"], decimals)
                                        for name, r in effect.iterrows()],
        "liquidazioni a mercato per sessione": effect["liquidazioni a mercato"].map(lambda v: it(v, 0)),
    })


def fill_rates_table(rates: pd.DataFrame) -> pd.DataFrame:
    """Da ``fill_rates``: esito delle coppie di quote."""
    out = pd.DataFrame({"coppie quotate": rates["coppie"].map(lambda v: it(v))})
    for col, label in [("bilaterali", "entrambi i lati eseguiti"), ("unilaterali", "un solo lato eseguito"),
                       ("senza fill", "nessun lato eseguito"), ("con fill parziale", "con fill parziale")]:
        out[label] = rates[col].map(lambda v: pct(v, 0))
    return out


def nel(value, decimals: int = 0) -> str:
    """Percentuale con la preposizione corretta: "nell'85%", "nel 56%"."""
    text = pct(value, decimals)
    elided = text.startswith(("8", "11", "1%", "1,")) and not text.startswith(("18", "1%0"))
    return f"nell'{text}" if elided else f"nel {text}"
