"""Run the series notebooks locally in order (for testing), then store their outputs in the Colab versions.
Usage: py -3.13 run_series_locally.py [first_number] [--flag name=value ...]"""
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

HERE = Path(__file__).parent
LOCAL = (HERE.parent / "DataSets_for_Drive" / "ThisTimeIsDifferent").as_posix() + "/"
SETUP_LINES = ["from google.colab import drive", "drive.mount('/content/drive')",
               "folder = '/content/drive/MyDrive/DataSets/ThisTimeIsDifferent/'"]


def patched(nb, flags):
    test = nbformat.from_dict(nbformat.reads(nbformat.writes(nb), as_version=4))
    for c in test.cells:
        if c.cell_type != "code":
            continue
        if SETUP_LINES[0] in c.source:
            src = c.source
            for line in SETUP_LINES:
                src = src.replace(line, "")
            c.source = f"folder = '{LOCAL}'\n" + src
        if c.source.startswith("!pip"):
            c.source = "pass"
        for name, value in flags.items():
            c.source = c.source.replace(f"{name} = False", f"{name} = {value}")
    return test


def main():
    first = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    flags = dict(a.split("=") for a in sys.argv[2:])
    keep = not flags                       # outputs are stored only for normal runs
    for path in sorted(HERE.glob("[0-9][0-9]_*.ipynb")):
        if int(path.name[:2]) < first:
            continue
        nb = nbformat.read(path, as_version=4)
        test = patched(nb, flags)
        t0 = time.time()
        client = NotebookClient(test, timeout=7200, kernel_name="py313", resources={"metadata": {"path": str(HERE)}})
        try:
            client.execute()
            status = "ok"
        except Exception as e:
            status = "ERROR " + type(e).__name__
        print(f"{path.name:45s} {status:20s} {time.time() - t0:6.0f} s", flush=True)
        if status != "ok":
            for c in test.cells:
                for o in c.get("outputs", []):
                    if o.output_type == "error":
                        print("   ", o.ename, o.evalue[:400])
            return
        if keep:
            for orig, run in zip(nb.cells, test.cells):
                if orig.cell_type == "code":
                    is_setup = SETUP_LINES[0] in orig.source or orig.source.startswith("!pip")
                    orig.outputs = [] if is_setup else run.outputs
                    orig.execution_count = None if is_setup else run.execution_count
            nbformat.write(nb, path)


if __name__ == "__main__":
    main()
