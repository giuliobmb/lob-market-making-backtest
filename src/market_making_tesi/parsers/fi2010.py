from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np

from ..models import BookSnapshot


def _load_matrix(path: str | Path | np.ndarray, delimiter: str | None = None) -> np.ndarray:
    """Carica uno dei file testuali della distribuzione FI-2010.

    La versione comunemente distribuita ha 149 righe (feature/label) e una
    osservazione per colonna. Alcune copie sono già trasposte: il controllo
    seguente gestisce entrambe le disposizioni senza modificare i dati.
    """

    if isinstance(path, np.ndarray):
        return path
    path = Path(path)
    if delimiter is None:
        # Lettura veloce per i file completi (centinaia di MB): stessi valori di
        # ``np.loadtxt`` con separatori di spazi bianchi.
        with path.open("r", encoding="ascii") as stream:
            columns = len(stream.readline().split())
        values = np.fromfile(path, sep=" ")
        if columns == 0 or values.size % columns:
            raise ValueError(f"Matrice FI-2010 irregolare in {path}")
        matrix = values.reshape(-1, columns)
    else:
        matrix = np.loadtxt(path, delimiter=delimiter, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    if matrix.shape[0] not in {144, 149} and matrix.shape[1] in {144, 149}:
        matrix = matrix.T
    if matrix.shape[0] < 40:
        raise ValueError(
            f"FI-2010 richiede almeno 40 righe LOB; forma trovata: {matrix.shape}"
        )
    return matrix


def fi2010_stock_segments(
    path: str | Path | np.ndarray,
    jump: float = 0.05,
    delimiter: str | None = None,
) -> list[tuple[int, int]]:
    """Individua i blocchi contigui dei cinque titoli in un file FI-2010.

    I file concatenano le osservazioni dei titoli senza una colonna
    identificativa. Il passaggio da un titolo al successivo produce un salto
    relativo del midpoint molto superiore a quelli intraday: si separa dove
    ``|mid[t+1] / mid[t] - 1| > jump``. Restituisce intervalli ``[start, stop)``.
    """

    matrix = _load_matrix(path, delimiter)
    mid = (matrix[0] + matrix[2]) / 2
    jumps = np.flatnonzero(np.abs(mid[1:] / mid[:-1] - 1) > jump) + 1
    bounds = [0, *jumps.tolist(), mid.size]
    return list(zip(bounds[:-1], bounds[1:]))


def load_fi2010_matrix(path: str | Path, delimiter: str | None = None) -> np.ndarray:
    """Matrice ``(righe, osservazioni)``, da caricare una volta per piu blocchi."""

    return _load_matrix(path, delimiter)


def iter_fi2010_snapshots(
    path: str | Path | np.ndarray,
    symbol: str = "FI2010",
    max_observations: int | None = None,
    delimiter: str | None = None,
    start: int = 0,
    stop: int | None = None,
) -> Iterator[BookSnapshot]:
    """Converte le prime 40 feature FI-2010 in snapshot a 10 livelli.

    Per ogni livello l'ordine originale è: ask price, ask volume, bid price,
    bid volume. I prezzi sono normalizzati: il P&L risultante è quindi un
    indicatore comparativo, non un profitto monetario. ``start``/``stop``
    selezionano un blocco (vedi ``fi2010_stock_segments``) per non unire
    titoli diversi nello stesso replay.
    """

    matrix = _load_matrix(path, delimiter)
    stop = matrix.shape[1] if stop is None else min(stop, matrix.shape[1])
    if max_observations is not None:
        stop = min(stop, start + max_observations)

    for index in range(start, stop):
        lob = matrix[:40, index].reshape(10, 4)
        yield BookSnapshot(
            timestamp=index,
            event_index=index,
            symbol=symbol,
            ask_prices=tuple(float(value) for value in lob[:, 0]),
            ask_sizes=tuple(float(value) for value in lob[:, 1]),
            bid_prices=tuple(float(value) for value in lob[:, 2]),
            bid_sizes=tuple(float(value) for value in lob[:, 3]),
        )


def load_fi2010_labels(
    path: str | Path,
    delimiter: str | None = None,
) -> np.ndarray | None:
    """Restituisce le cinque righe di label, quando presenti."""

    matrix = _load_matrix(path, delimiter)
    return matrix[-5:, :].copy() if matrix.shape[0] >= 149 else None
