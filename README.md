# Valutazione di una strategia di market making mediante dati LOB L2, L3 e L4

Codice e notebook della tesi di Giulio Bombarda. La stessa strategia di
market making passivo viene simulata con regole di esecuzione euristiche su dati
L2 (`touch`, `price-through` e, come baseline aggiuntiva, `cross`) e con un
modello a coda FIFO su dati order-level L3 (Nasdaq TotalView-ITCH) e L4
(Hyperliquid), per misurare quanto la profittabilità stimata dipenda dal
modello di esecuzione.

## Dati

| Dataset | Livello | Sessioni | Ruolo |
|---|---|---|---|
| FI-2010 (10 giorni × 5 titoli) | L2 normalizzato | 50 titolo-giorno | EDA, volatilità, baseline L2 (E1, E2) |
| Nasdaq ITCH 5.0 (7 giornate 2019-2020, file completi) | L3 MBO | 21 titolo-giorno (AAPL, MSFT, INTC), 09:30-16:00 | stima principale con coda FIFO (E3) |
| Hyperliquid BTC-PERP (6 ore, 1-3/12/2025) | L4 + trade | 6 ore | robustezza su un secondo mercato (E4) |

I file ITCH completi occupano circa 31 GB; gli altri dataset meno di 1 GB per file.
Fonti, date di accesso e SHA-256 sono in
[`data/manifest.json`](data/manifest.json); le fonti bibliografiche in
[`docs/fonti.md`](docs/fonti.md).

## Esecuzione (Windows, PowerShell)

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e ".[notebooks]"
.venv/Scripts/python.exe scripts/download_data.py --fi2010-zip <zip scaricato da Etsin>
.venv/Scripts/mm-backtest.exe ingest       # grezzi -> Parquet in data/interim (~3 min)
.venv/Scripts/mm-backtest.exe pipeline     # qualità, calibrazione, validazione, test, ablazione, stress, varianti (~1 ora)
.venv/Scripts/python.exe scripts/validate_notebooks.py   # esegue i notebook 00-06
```

`.venv/Scripts/python.exe -m unittest discover -s tests -v` esegue i test.

## Architettura

| Componente | Modulo | Output |
|---|---|---|
| parsers | `parsers/` (FI-2010, ITCH, Hyperliquid), `ingest.py`, `storage.py` | eventi e snapshot normalizzati in Parquet partizionato |
| orderbook | `orderbook.py` | book MBO incrementale e vista L2 |
| strategy | `strategy.py` | intenti di ordine (quota, cancella, liquida) |
| execution | `execution.py` | regole L2, coda FIFO, latenza e ordini post-only |
| accounting | `accounting.py` | cassa, inventario, commissioni, mark to market |
| evaluation | `evaluation.py`, `pipeline.py`, `quality.py` | metriche, scomposizione del P&L, protocollo, controlli |

`engine.py` collega i componenti; `report.py` contiene le funzioni comuni ai notebook.

## Notebook

| Notebook | Contenuto |
|---|---|
| `00_setup` | ambiente, verifica del manifesto, configurazioni, sessioni, test |
| `01_fi2010_eda` | qualità, confini tra titoli, distribuzioni, soglie di volatilità |
| `02_l2_backtest` | E1 e E2 su FI-2010 |
| `03_itch_replay` | parsing ITCH, ricostruzione MBO, controlli di qualità |
| `04_l3_backtest` | E3: coda FIFO contro regole L2 sullo stesso feed |
| `05_hyperliquid` | E4: ciclo di vita L4, classificazione fill/cancellazioni |
| `06_results` | confronto, costi, liquidazione, latenza (E5), prestazioni, risposta alla domanda |

## Strategia

Market making passivo con una coppia bid/ask al best, quantità fissa, una
coppia alla volta, scadenza e liquidazione dell'inventario residuo. La strategia
**opera solo a bassa volatilità**: la volatilità è la deviazione standard dei
rendimenti del midpoint su una finestra mobile (*W* record per FI-2010 e
Hyperliquid, *W* secondi per ITCH, con midpoint campionato ogni
secondo); sopra soglia cancella le quote, liquida e resta ferma finché la stima
non rientra. Finestra e soglia (quantile 50%, 75% o 90% della volatilità di
calibrazione) sono scelte in validazione; l'ablazione misura il contributo del
filtro ripetendo il test con la strategia sempre attiva.

### Varianti (estensione)

- **Inventory skew:** dopo un fill su un solo lato non si liquida a mercato alla
  scadenza; si mantiene una quota passiva solo sul lato che riduce l'inventario,
  al best migliorato di *k* tick se lo spread lo consente, e si liquida a mercato
  solo dopo un tempo massimo di detenzione (o con volatilità sopra soglia).
- **Filtro di imbalance:** con I = (q_bid − q_ask)/(q_bid + q_ask) al best, si
  apre una coppia solo se |I| ≤ θ e si cancella una quota quando l'imbalance la
  minaccia (I > θ per una vendita, I < −θ per un acquisto).

I parametri (*k*, tempo massimo, θ) sono scelti in validazione a parità di
parametri base; lo stadio `variants` confronta le varianti con la strategia base
sulle stesse sessioni di test (`results/variants.csv`).

## Protocollo

- **Sessioni** indipendenti: titolo-giorno o ora, mai mescolate.
- **Calibrazione:** la prima sessione di ogni strumento; la soglia di volatilità
  è un quantile della volatilità mobile osservata in quella sessione e viene
  applicata alle sessioni successive.
- **Validazione:** le sessioni successive alla calibrazione scelgono finestra e
  quantile della volatilità e durata degli ordini (griglia 2×3×2) con il
  modello di riferimento.
- **Test:** sessioni successive, mai usate per scegliere parametri; la strategia
  selezionata è identica per tutti i modelli di fill.
- **Ablazione:** il test ripetuto senza filtro di volatilità.
- **Stress:** latenza crescente sul modello a coda; costi e liquidazione al
  midpoint come scenari sui fill, perché non cambiano le decisioni.

I risultati sono in `results/` (`test.csv`, `validation.csv`, `stress.csv`,
`ablation.csv`, `quality.csv`, `calibration.csv`, `selected_params.json`, figure e singole
simulazioni in `results/runs/`).
