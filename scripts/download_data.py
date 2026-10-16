"""Scarica o estrae i dataset grezzi in ``data/raw`` e li registra nel manifesto.

Il manifesto ``data/manifest.json`` (versionato) riporta per ogni file URL di
origine, data di accesso, dimensione e SHA-256. Alle esecuzioni successive i
file presenti vengono verificati contro il manifesto; quelli mancanti vengono
riscaricati e devono riprodurre lo stesso hash.

* Nasdaq TotalView-ITCH 5.0: i sette file giornalieri pubblici completi
  (3,5-5,6 GB ciascuno), scaricati a blocchi paralleli.
* Hyperliquid (Zenodo 10.5281/zenodo.18184441): dagli archivi tar da decine di
  GB si scaricano solo i file orari richiesti (book diff e trade).
* FI-2010: lo zip va scaricato a mano da Etsin e passato con ``--fi2010-zip``;
  si estraggono i dieci giorni NoAuction DecPre (giorno 1 = Train CF_1,
  giorno k+1 = Test CF_k).

Uso: python scripts/download_data.py [--fi2010-zip PERCORSO] [--only itch hyperliquid ...]
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import http.client
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
MANIFEST = ROOT / "data" / "manifest.json"

ITCH_URL = "https://emi.nasdaq.com/ITCH/Nasdaq%20ITCH/{}.NASDAQ_ITCH50.gz"
ITCH_DAYS = ["01302019", "03272019", "07302019", "08302019", "10302019", "12302019", "01302020"]

ZENODO = "https://zenodo.org/api/records/18184441/files/{}/content"
# (data, ora UTC): stesso orario su tre giorni consecutivi e un secondo orario.
HYPERLIQUID_HOURS = [("20251201", 8), ("20251201", 14), ("20251202", 8),
                     ("20251202", 14), ("20251203", 8), ("20251203", 14)]

FI2010_URL = "https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649"
FI2010_BASE = "BenchmarkDatasets/NoAuction/3.NoAuction_DecPre/NoAuction_DecPre_{}/{}"


def fi2010_members() -> dict[str, str]:
    """Nome locale -> membro dello zip, per i dieci giorni."""
    members = {"fi2010_day01.txt": FI2010_BASE.format("Training", "Train_Dst_NoAuction_DecPre_CF_1.txt")}
    for k in range(1, 10):
        members[f"fi2010_day{k + 1:02d}.txt"] = FI2010_BASE.format(
            "Testing", f"Test_Dst_NoAuction_DecPre_CF_{k}.txt")
    return members


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class Manifest:
    def __init__(self) -> None:
        self.entries = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {}

    def check(self, path: Path) -> bool:
        """True se il file esiste ed e coerente con il manifesto."""
        key = path.relative_to(ROOT).as_posix()
        if not path.exists():
            return False
        if key in self.entries:
            if sha256(path) != self.entries[key]["sha256"]:
                raise SystemExit(f"Hash diverso da quello del manifesto: {key}")
            print(f"OK  {key}")
            return True
        return False

    def record(self, path: Path, source: str, note: str = "") -> None:
        key = path.relative_to(ROOT).as_posix()
        digest = sha256(path)
        previous = self.entries.get(key)
        if previous and previous["sha256"] != digest:
            raise SystemExit(f"Il file riscaricato non riproduce l'hash registrato: {key}")
        self.entries[key] = previous or {
            "source": source, "accessed": dt.date.today().isoformat(),
            "bytes": path.stat().st_size, "sha256": digest, "note": note}
        MANIFEST.write_text(json.dumps(self.entries, indent=2, sort_keys=True), encoding="utf-8")
        print(f"REG {key}")


def _get(url: str, start: int | None = None, length: int | None = None, attempts: int = 5):
    """Richiesta HTTP (eventualmente Range) con timeout e tentativi ripetuti."""
    import time

    headers = {} if start is None else {"Range": f"bytes={start}-{start + length - 1}"}
    for attempt in range(attempts):
        try:
            response = urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60)
            break
        except (OSError, http.client.HTTPException):
            if attempt == attempts - 1:
                raise
            time.sleep(2 ** attempt)
    if start is not None and response.status != 206:
        raise SystemExit(f"Il server non supporta richieste Range: {url}")
    return response


def download(url: str, target: Path, start: int | None = None, length: int | None = None,
             parallel: int = 8, chunk: int = 8 << 20) -> None:
    """Scarica un file o un intervallo di byte; gli intervalli lunghi a blocchi paralleli
    (Zenodo limita la banda per connessione)."""
    from concurrent.futures import ThreadPoolExecutor

    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    if start is None or length <= chunk:
        with _get(url, start, length) as response, partial.open("wb") as out:
            shutil.copyfileobj(response, out, 1 << 20)
    else:
        with partial.open("wb") as out:
            out.truncate(length)

        def fetch(offset: int) -> None:
            size = min(chunk, length - offset)
            for attempt in range(5):
                try:
                    with _get(url, start + offset, size) as response:
                        data = response.read()
                    if len(data) == size:
                        break
                except (OSError, http.client.HTTPException):
                    if attempt == 4:
                        raise
            else:
                raise SystemExit(f"Blocco incompleto a {offset} in {url}")
            with partial.open("r+b") as out:
                out.seek(offset)
                out.write(data)

        with ThreadPoolExecutor(parallel) as pool:
            list(pool.map(fetch, range(0, length, chunk)))
    partial.replace(target)


def tar_index(url: str, wanted: set[str]) -> dict[str, tuple[int, int]]:
    """Posizione e dimensione dei membri richiesti, leggendo solo le intestazioni del tar."""
    found, offset = {}, 0
    while wanted - found.keys():
        with _get(url, offset, 512) as response:
            header = response.read()
        if header == b"\0" * 512:
            break
        name = header[:100].rstrip(b"\0").decode()
        size = int(header[124:136].strip(b"\0 ") or b"0", 8)
        if header[156:157] == b"L":  # nome lungo GNU
            with _get(url, offset + 512, size) as response:
                name = response.read().rstrip(b"\0").decode()
            offset += 512 + (size + 511) // 512 * 512
            with _get(url, offset, 512) as response:
                header = response.read()
            size = int(header[124:136].strip(b"\0 ") or b"0", 8)
        if name in wanted:
            found[name] = (offset + 512, size)
        offset += 512 + (size + 511) // 512 * 512
    missing = wanted - found.keys()
    if missing:
        raise SystemExit(f"Membri non trovati in {url}: {sorted(missing)}")
    return found


def get_itch(manifest: Manifest) -> None:
    for day in ITCH_DAYS:
        target = RAW / "itch" / f"{day}.NASDAQ_ITCH50.gz"
        if manifest.check(target):
            continue
        if not target.exists():
            with _get(ITCH_URL.format(day), 0, 1) as response:
                size = int(response.headers["Content-Range"].rsplit("/", 1)[1])
            print(f"Scarico {day} ({size / 1e9:.1f} GB)", flush=True)
            download(ITCH_URL.format(day), target, 0, size, parallel=6, chunk=64 << 20)
        manifest.record(target, ITCH_URL.format(day), "file giornaliero completo")


def get_hyperliquid(manifest: Manifest) -> None:
    jobs = {"book_diffs_202512.tar": {}, "trades_2025_12.tar": {}}
    for day, hour in HYPERLIQUID_HOURS:
        jobs["book_diffs_202512.tar"][f"{day}/ex{hour}.gz"] = RAW / "hyperliquid" / f"book_diffs_{day}_{hour:02d}.gz"
        jobs["trades_2025_12.tar"][f"{day}/{hour}.gz"] = RAW / "hyperliquid" / f"trades_{day}_{hour:02d}.gz"
    for archive, members in jobs.items():
        url = ZENODO.format(archive)
        todo = {m: t for m, t in members.items() if not manifest.check(t)}
        missing = {m for m, t in todo.items() if not t.exists()}
        index = tar_index(url, missing) if missing else {}
        for member, target in todo.items():
            if member in index:
                start, size = index[member]
                print(f"Scarico {archive}:{member} ({size / 1e6:.0f} MB)")
                download(url, target, start, size)
            manifest.record(target, url, f"membro {member} del tar")


def get_fi2010(manifest: Manifest, archive: Path | None) -> None:
    todo = {name: member for name, member in fi2010_members().items()
            if not manifest.check(RAW / "fi2010" / name)}
    missing = {n: m for n, m in todo.items() if not (RAW / "fi2010" / n).exists()}
    if missing:
        if archive is None:
            raise SystemExit("FI-2010 assente: scaricare lo zip da Etsin e usare --fi2010-zip")
        (RAW / "fi2010").mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as outer:
            info = next(i for i in outer.infolist() if i.filename.endswith("BenchmarkDatasets.zip"))
            with zipfile.ZipFile(outer.open(info)) as inner:
                for name, member in missing.items():
                    print(f"Estraggo {member}")
                    with inner.open(member) as src, (RAW / "fi2010" / name).open("wb") as out:
                        shutil.copyfileobj(src, out, 1 << 20)
    for name, member in todo.items():
        manifest.record(RAW / "fi2010" / name, FI2010_URL, f"BenchmarkDatasets.zip:{member}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fi2010-zip", type=Path, help="zip scaricato da Etsin")
    parser.add_argument("--only", nargs="+", choices=["itch", "hyperliquid", "fi2010"],
                        default=["itch", "hyperliquid", "fi2010"])
    args = parser.parse_args()
    manifest = Manifest()
    if "fi2010" in args.only:
        get_fi2010(manifest, args.fi2010_zip)
    if "hyperliquid" in args.only:
        get_hyperliquid(manifest)
    if "itch" in args.only:
        get_itch(manifest)


if __name__ == "__main__":
    main()
