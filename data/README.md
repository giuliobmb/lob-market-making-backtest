# Dati

| Cartella | Contenuto | Versionata |
|---|---|---|
| `raw/` | file grezzi così come scaricati (ITCH: ~31 GB in totale) | no |
| `interim/` | eventi e snapshot normalizzati in Parquet partizionato | no |
| `samples/` | campioni piccoli per i test | sì |
| `manifest.json` | fonte, data di accesso, byte e SHA-256 di ogni file grezzo | sì |

`scripts/download_data.py` popola `raw/` e verifica gli hash contro il manifesto;
`mm-backtest ingest` produce `interim/`.

## File grezzi

| File | Origine |
|---|---|
| `raw/fi2010/fi2010_dayNN.txt` | FI-2010 NoAuction DecPre: giorno 1 = `Train_Dst_..._CF_1`, giorno k+1 = `Test_Dst_..._CF_k` (zip Etsin, download manuale) |
| `raw/itch/MMDDYYYY.NASDAQ_ITCH50.gz` | file giornalieri completi Nasdaq TotalView-ITCH 5.0 |
| `raw/hyperliquid/book_diffs_YYYYMMDD_HH.gz`, `trades_YYYYMMDD_HH.gz` | file orari estratti dagli archivi Zenodo 10.5281/zenodo.18184441 |

## Dati intermedi

`interim/<dataset>/session=<id>/symbol=<simbolo>/events.parquet` (flusso
order-level) oppure `snapshots.parquet` (book L2 a 10 livelli). Schema comune
degli eventi: `timestamp`, `event_index`, `event_type` (ADD, EXECUTE, CANCEL,
DELETE, UPDATE, REPLACE), `order_id`, `side`, `price`, `size`, `new_order_id`,
`new_price`, `new_size`, `bootstrap` (eventi che ricostruiscono solo lo stato
iniziale del book, senza attivare la strategia).

## Note per dataset

- **FI-2010.** Ogni file giornaliero concatena cinque titoli senza
  identificativo; i confini sono dove il midpoint salta di oltre il 5%. Prezzi
  normalizzati, un'osservazione ogni 10 eventi. Righe 1-40: per ogni livello
  ask price, ask volume, bid price, bid volume.
- **ITCH.** Il file contiene i messaggi dalla mezzanotte; il filtro usa lo
  *stock locate* assegnato dai messaggi `R`. Prezzi in decimillesimi di
  dollaro. Gli eventi prima delle 09:30 sono `bootstrap`.
- **Hyperliquid.** Nei trade `side` è l'aggressore e `side_info` è
  [acquirente, venditore]: il maker è il venditore se `side == "B"`. Le
  riduzioni di un ordine nei book diff sono attribuite prima al volume eseguito
  osservato per quell'`oid`, poi a una cancellazione. I diff non hanno timestamp.
