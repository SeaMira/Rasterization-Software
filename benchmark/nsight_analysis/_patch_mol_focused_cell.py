"""Append focused-export cells to benchmark_mol_yaml.ipynb if missing."""
from __future__ import annotations

import json
from pathlib import Path

NB = Path(__file__).resolve().parent / "benchmark_mol_yaml.ipynb"
MARKER = "export_thesis_focused_figures"

MD_CELL = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## Export focused thesis figures (molecular scenes)\n",
        "\n",
        "Regenerates PNGs under `img/chapter3/results/{fst_gpu|scnd_gpu|std|hybrid}/**/focused/` "
        "with only the subcharts cited in the thesis. Requires `df_detailed_metrics` from above.\n",
    ],
}

CODE_CELL = {
    "cell_type": "code",
    "metadata": {},
    "outputs": [],
    "source": [
        "import sys\n",
        "from pathlib import Path\n",
        "\n",
        "_ns = Path.cwd()\n",
        "if str(_ns) not in sys.path:\n",
        "    sys.path.insert(0, str(_ns))\n",
        "\n",
        "from _thesis_focused_export import export_thesis_focused_figures\n",
        "\n",
        "export_thesis_focused_figures(df_detailed_metrics)\n",
    ],
}


def main() -> None:
    raw = NB.read_text(encoding="utf-8")
    nb = json.loads(raw)
    if any(MARKER in "".join(c.get("source", [])) for c in nb.get("cells", [])):
        print("Notebook already has focused export cell")
        return
    nb["cells"].extend([MD_CELL, CODE_CELL])
    NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Patched {NB}")


if __name__ == "__main__":
    main()
