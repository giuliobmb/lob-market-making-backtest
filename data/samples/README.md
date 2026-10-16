# Campioni piccoli per i test

| File | Origine | Scopo |
|---|---|---|
| `fi2010_real_60.txt` | 60 osservazioni reali estratte da `Train_Dst_NoAuction_DecPre_CF_7.txt` | test dell'adattatore L2 e dei due modelli di fill |
| `hyperliquid_book_diffs_real_500.jsonl` | 500 eventi BTC reali dal primo file dell'archivio Zenodo | test dello schema `raw_book_diff` e ricostruzione L4 |
| `itch_aapl_synthetic_10.gz` | campione controllato conforme a ITCH 5.0 | test binario riproducibile della coda prezzo-tempo |
| `itch_aapl_synthetic_10_decoded.csv` | versione leggibile del file precedente | ispezione manuale |
| `hyperliquid_queue_synthetic_10.jsonl` | campione controllato | verifica deterministica dei fill queue-aware |
| `snapshots_preview.csv` | prodotto dai tre adattatori | confronto immediato del primo book ricostruibile |

## Perché due campioni sono sintetici

Il sample pubblico Nasdaq completo occupa diversi gigabyte. Il file ITCH qui
incluso usa messaggi binari conformi alla specifica ufficiale ma valori creati
appositamente per esercitare `ADD` ed `EXECUTE` in pochi byte.

Nel book diff pubblico Hyperliquid una rimozione non distingue cancellazione e
fill. Il campione reale serve quindi alla ricostruzione del book; il piccolo
campione sintetico, che classifica esplicitamente i fill, serve a verificare la
logica della coda senza attribuire esecuzioni non osservate.

## Rigenerazione

Il file FI-2010 reale e quello Hyperliquid reale restano invariati. Gli altri
file e `snapshots_preview.csv` possono essere rigenerati dalla radice del
progetto con:

```powershell
.venv/Scripts/python.exe scripts/create_test_samples.py
```
