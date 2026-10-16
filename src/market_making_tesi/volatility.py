from __future__ import annotations

import math
from collections import deque


class VolatilityEstimator:
    """Volatilita su finestra mobile; stima causale (usa solo midpoint gia osservati).

    La volatilita e la deviazione standard campionaria degli ultimi ``window``
    rendimenti del midpoint. I rendimenti possono essere misurati:

    * **in tempo di evento** (``sample_ns=None``): un rendimento per record;
      adatto a dati campionati a passo fisso (FI-2010) o privi di timestamp
      (book diff Hyperliquid);
    * **in tempo di calendario** (``sample_ns`` impostato): il midpoint e
      campionato alla fine di ogni intervallo di ``sample_ns`` nanosecondi e la
      finestra copre ``window`` intervalli. Gli intervalli senza record hanno
      rendimento nullo. Rende la stima indipendente dall'intensita dei
      messaggi, che nei dati order-level varia di ordini di grandezza (es.
      raffiche di inserimenti e cancellazioni all'apertura).

    Tra due chiusure di intervallo la stima restituita resta quella dell'ultimo
    intervallo completato.
    """

    def __init__(
        self,
        window: int,
        min_periods: int,
        method: str = "rolling",
        return_type: str = "log",
        ewma_lambda: float = 0.94,
        sample_ns: int | None = None,
    ) -> None:
        self.window = window
        self.min_periods = min_periods
        self.method = method
        self.return_type = return_type
        self.ewma_lambda = ewma_lambda
        self.sample_ns = sample_ns
        self._returns: deque[float] = deque(maxlen=window)
        self._last_mid: float | None = None
        self._ewma_variance: float | None = None
        self._sigma: float | None = None
        self._bucket: int | None = None
        self._pending: float | None = None
        # Somme mobili per la deviazione standard in O(1), ricalcolate
        # periodicamente per evitare l'accumulo di errori di arrotondamento.
        self._sum = 0.0
        self._sum_sq = 0.0
        self._updates = 0

    def _return(self, midpoint: float) -> float:
        assert self._last_mid is not None
        if self.return_type == "log":
            if midpoint <= 0 or self._last_mid <= 0:
                raise ValueError(
                    "Il rendimento logaritmico richiede prezzi positivi; "
                    "per FI-2010 normalizzato usare return_type='diff'."
                )
            return math.log(midpoint / self._last_mid)
        if self.return_type == "diff":
            return midpoint - self._last_mid
        raise ValueError(f"return_type non supportato: {self.return_type}")

    def update(self, midpoint: float, timestamp: int | None = None) -> float | None:
        if not math.isfinite(midpoint):
            return self._sigma
        if self.sample_ns is None:
            return self._observe(midpoint)
        if timestamp is None:
            raise ValueError("La volatilita in tempo di calendario richiede il timestamp")
        bucket = timestamp // self.sample_ns
        if self._bucket is None:
            self._bucket, self._pending = bucket, midpoint
            return self._sigma
        if bucket > self._bucket:
            # Chiusura dell'intervallo precedente; gli intervalli vuoti hanno rendimento nullo.
            self._observe(self._pending)
            for _ in range(min(bucket - self._bucket - 1, self.window)):
                self._observe(self._pending)
            self._bucket = bucket
        self._pending = midpoint
        return self._sigma

    def _observe(self, midpoint: float) -> float | None:
        if self._last_mid is None:
            self._last_mid = midpoint
            return None

        value = self._return(midpoint)
        self._last_mid = midpoint
        if len(self._returns) == self.window:
            dropped = self._returns[0]
            self._sum -= dropped
            self._sum_sq -= dropped * dropped
        self._returns.append(value)
        self._sum += value
        self._sum_sq += value * value
        self._updates += 1
        if self._updates % 10_000 == 0:
            self._sum = math.fsum(self._returns)
            self._sum_sq = math.fsum(r * r for r in self._returns)

        if self.method == "ewma":
            squared = value * value
            if self._ewma_variance is None:
                self._ewma_variance = squared
            else:
                lam = self.ewma_lambda
                self._ewma_variance = lam * self._ewma_variance + (1 - lam) * squared
            self._sigma = (math.sqrt(max(self._ewma_variance, 0.0))
                           if len(self._returns) >= self.min_periods else None)
            return self._sigma

        if self.method != "rolling":
            raise ValueError(f"Metodo di volatilità non supportato: {self.method}")
        n = len(self._returns)
        if n < self.min_periods:
            self._sigma = None
            return None
        variance = (self._sum_sq - self._sum * self._sum / n) / (n - 1)
        self._sigma = math.sqrt(max(variance, 0.0))
        return self._sigma
