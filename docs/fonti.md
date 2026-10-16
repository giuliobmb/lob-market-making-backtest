# Fonti e scelte rispetto al piano di lavoro

## Fonti bibliografiche (piano, §7)

- Avellaneda, M., Stoikov, S. (2008). *High-frequency trading in a limit order book*. Quantitative Finance 8(3).
- Guéant, O., Lehalle, C.-A., Fernandez-Tapia, J. (2013). *Dealing with the inventory risk: a solution to the market making problem*. Mathematics and Financial Economics 7(4).
- Huang, W., Lehalle, C.-A., Rosenbaum, M. (2015). *Simulating and analyzing order book data: the queue-reactive model*. JASA 110(509).
- Moallemi, C., Yuan, K. (2017). *A model for queue position valuation in a limit order book*. Working paper.
- Ntakaris, A., Magris, M., Kanniainen, J., Gabbouj, M., Iosifidis, A. (2018). *Benchmark dataset for mid-price forecasting of limit order book data with machine learning methods*. Journal of Forecasting 37(8).

## Fonti aggiunte (citate nei notebook)

| Riferimento | Dove e perché è citato |
|---|---|
| Ho, T., Stoll, H. R. (1981). *Optimal dealer pricing under transactions and return uncertainty*. Journal of Financial Economics 9(1). | 00: origine del rischio di inventario del market maker. |
| Cartea, Á., Jaimungal, S., Penalva, J. (2015). *Algorithmic and High-Frequency Trading*. Cambridge University Press. | 00: liquidazione dell'inventario residuo a fine orizzonte. |
| Glosten, L. R., Milgrom, P. R. (1985). *Bid, ask and transaction prices in a specialist market with heterogeneously informed traders*. Journal of Financial Economics 14(1). | 00, 04, glossario: selezione avversa, causa principale delle perdite. |
| Cont, R., Stoikov, S., Talreja, R. (2010). *A stochastic model for order book dynamics*. Operations Research 58(3). | 02, 04: l'esecuzione dipende dalla coda, assente nei dati L2. |
| Gould, M. D. et al. (2013). *Limit order books*. Quantitative Finance 13(11). | Bibliografia: rassegna di riferimento sui book. |
| Bouchaud, J.-P., Bonart, J., Donier, J., Gould, M. (2018). *Trades, Quotes and Prices*. Cambridge University Press. | Bibliografia: microstruttura e dinamica della coda. |
| Cont, R., Kukanov, A., Stoikov, S. (2014). *The price impact of order book events*. Journal of Financial Econometrics 12(1). | 04, 06: base del filtro di imbalance. |
| Cartea, Á., Donnelly, R., Jaimungal, S. (2018). *Enhancing trading strategies with order book signals*. Applied Mathematical Finance 25(1). | 04, 06: uso dell'imbalance nelle strategie. |
| Hasbrouck, J., Saar, G. (2013). *Low-latency trading*. Journal of Financial Markets 16(4). | 06: importanza della velocità. |
| Budish, E., Cramton, P., Shim, J. (2015). *The high-frequency trading arms race*. Quarterly Journal of Economics 130(4). | 06: competizione sulla latenza. |
| Menkveld, A. J. (2013). *High frequency trading and the new market makers*. Journal of Financial Markets 16(4). | 06: comportamento di un market maker reale. |
| Bailey, D. H., Borwein, J. M., López de Prado, M., Zhu, Q. J. (2017). *The probability of backtest overfitting*. Journal of Computational Finance 20(4). | 00: motivazione della divisione cronologica dei dati. |
| Efron, B., Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall. | Glossario: intervalli di confidenza bootstrap. |

La strategia non implementa i modelli ottimi di Avellaneda e Stoikov (2008) o di
Guéant, Lehalle e Fernandez-Tapia (2013): ne riprende solo due risultati qualitativi
(il rischio di inventario cresce con la volatilità; le quote vanno spostate in base
all'inventario), tradotti nel filtro di volatilità e nella variante di inventory skew.

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
