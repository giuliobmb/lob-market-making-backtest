"""Esegue i notebook (tutti, o quelli indicati per nome) nel kernel della .venv e salva gli output."""
from pathlib import Path
import json
import sys
import tempfile

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager


def main():
    root = Path(__file__).resolve().parents[1]
    wanted = set(sys.argv[1:])
    paths = [p for p in sorted((root / "notebooks").glob("*.ipynb")) if not wanted or p.stem in wanted]
    if not paths:
        raise SystemExit("Nessun notebook trovato")
    with tempfile.TemporaryDirectory() as directory:
        kernel = Path(directory) / "tesi"
        kernel.mkdir()
        (kernel / "kernel.json").write_text(json.dumps({
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "Tesi", "language": "python",
        }), encoding="utf-8")
        for path in paths:
            notebook = nbformat.read(path, as_version=4)
            nbformat.validate(notebook)
            manager = KernelManager(kernel_name="tesi",
                                    kernel_spec_manager=KernelSpecManager(kernel_dirs=[directory]))
            client = NotebookClient(notebook, km=manager, timeout=1800,
                                    resources={"metadata": {"path": str(root)}})
            try:
                client.execute()
            finally:
                if manager.has_kernel:
                    manager.shutdown_kernel(now=True)
            nbformat.validate(notebook)
            nbformat.write(notebook, path)
            print(f"OK {path.name}: eseguito e salvato con output")


if __name__ == "__main__":
    main()
