"""Execute ooc_package_analysis.ipynb code cells (non-interactive) to export thesis figures."""
import json
import os
import sys
import traceback
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

os.chdir(Path(__file__).resolve().parent)
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt  # noqa: E402

nb_path = Path("ooc_package_analysis.ipynb")
nb = json.loads(nb_path.read_text(encoding="utf-8"))

globs = {"display": lambda x=None, **k: None, "__name__": "__main__"}


def run():
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell.get("source", []))
        if not src.strip():
            continue
        print(f"--- cell {i} ---", flush=True)
        try:
            exec(compile(src, f"cell_{i}", "exec"), globs)
        except Exception:
            traceback.print_exc()
            sys.exit(1)
    plt.close("all")
    root = globs.get("FIG_ROOT")
    if root and Path(root).is_dir():
        n = sum(1 for _ in Path(root).rglob("*.png"))
        print(f"Exported {n} PNG files under {root}")


if __name__ == "__main__":
    run()
