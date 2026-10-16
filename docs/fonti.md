# Fonti e scelte rispetto al piano di lavoro

## Fonti bibliografiche (piano, §7)

- Avellaneda, M., Stoikov, S. (2008). *High-frequency trading in a limit order book*. Quantitative Finance 8(3).
- Guéant, O., Lehalle, C.-A., Fernandez-Tapia, J. (2013). *Dealing with the inventory risk: a solution to the market making problem*. Mathematics and Financial Economics 7(4).
- Huang, W., Lehalle, C.-A., Rosenbaum, M. (2015). *Simulating and analyzing order book data: the queue-reactive model*. JASA 110(509).
- Moallemi, C., Yuan, K. (2017). *A model for queue position valuation in a limit order book*. Working paper.
- Ntakaris, A., Magris, M., Kanniainen, J., Gabbouj, M., Iosifidis, A. (2018). *Benchmark dataset for mid-price forecasting of limit order book data with machine learning methods*. Journal of Forecasting 37(8).

## Fonti dei dati

| Dataset | Origine | Riferimento |
|---|---|---|
| FI-2010 | Etsin/Fairdata, dataset 73eb48d7-4dbc-4a10-a52a-da745b47a649 | Ntakaris et al. (2018) |
| Nasdaq TotalView-ITCH 5.0 | file campione pubblici `emi.nasdaq.com/ITCH/Nasdaq ITCH/` | specifica Nasdaq TotalView-ITCH 5.0 |
| Hyperliquid L4 | Zenodo, DOI 10.5281/zenodo.18184441 (licenza CC BY 4.0) | Albers, J., Cucuringu, M., Howison, S., Shestopaloff, A. Y. (2026). *An Open Book: Level 4 Order Book Data from the Hyperliquid Exchange* |

Le date di accesso e gli hash di ogni file sono in `data/manifest.json`.

## Scelte e integrazioni rispetto al piano

- **ITCH completo.** Si usano i sette file giornalieri pubblici interi
  (3,5-5,6 GB ciascuno) e la sessione regolare 09:30-16:00. Una prima versione
  usava solo il primo GB di ogni file (finestra 09:30-10:10) per un limite di
  spazio poi rimosso.
- **Regola `cross`.** Aggiunta come terza baseline L2 (fill quando il best opposto
  raggiunge la quota), perché è la convenzione più diffusa in letteratura; non
  sostituisce `touch` e `price-through`.
- **LOBSTER escluso.** Un campione LOBSTER (AAPL, 21/06/2012) era stato usato in
  una versione precedente come giornata di test aggiuntiva; è stato rimosso perché
  una sola giornata non permette valutazioni statistiche e i dati ITCH completi
  coprono già lo stesso mercato.
- **Hyperliquid.** I book diff non distinguono fill e cancellazioni: si uniscono
  ai trade pubblici tramite l'`oid` del maker. I diff non hanno timestamp, per cui
  durata degli ordini e latenza sono espresse in eventi.
- **Costi.** Tariffe indicative: Nasdaq rebate 0,20 cent/azione al maker e
  0,30 cent/azione al taker; Hyperliquid tier 0 (1,5 bps maker, 4,5 bps taker) e
  tier alto (-0,3 bps maker, 2,4 bps taker). Da verificare con le tabelle
  ufficiali del periodo prima della stesura finale.
- **Latenza.** Ritardo costante su inserimento e cancellazione; gli ordini sono
  post-only (rifiutati se all'attivazione attraverserebbero il book).
- **Volatilità in tempo di calendario per ITCH.** Misurata per
  messaggio, la volatilità dipende dall'intensità del flusso: all'apertura
  migliaia di messaggi non muovono il prezzo e la soglia calibrata risultava
  artificialmente bassa (in una sessione AAPL la strategia sarebbe stata attiva
  nell'1,7% del tempo). Per i dati con timestamp il midpoint è campionato ogni
  secondo e la finestra è in secondi; FI-2010 (campionamento fisso) e
  Hyperliquid (senza timestamp) restano in tempo di evento.
- **Calibrazione cronologica delle soglie.** Calibrare la soglia sul primo 20%
  di ogni sessione significava, per ITCH, calibrarla sull'apertura, il tratto più
  volatile della giornata: in tempo di calendario la soglia risultava alta e il
  filtro quasi inattivo (strategia attiva nel 98% del tempo). Come previsto dal
  piano, la soglia è ora stimata su una sessione di calibrazione precedente dello
  stesso strumento (FI-2010 giorno 1, ITCH 30/01/2019, Hyperliquid 1/12 ore 08) e
  applicata alle sessioni di validazione e test.
- **Varianti della strategia.** Inventory skew e filtro di imbalance non sono
  nel piano iniziale: sono estensioni aggiunte dopo aver osservato che la
  strategia base perde per selezione avversa e costi di liquidazione. La strategia
  base resta quella principale; le varianti usano gli stessi parametri base e sono
  valutate sulle stesse sessioni di test, con parametri propri scelti in validazione.
