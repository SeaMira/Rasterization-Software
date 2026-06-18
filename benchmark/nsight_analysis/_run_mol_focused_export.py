"""Run molecular focused thesis figure export (non-interactive)."""
from __future__ import annotations

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("MPLBACKEND", "Agg")

from _thesis_focused_export import (  # noqa: E402
    consolidar_metricas_detalladas,
    export_thesis_focused_figures,
    load_molecular_todos_los_datos,
)

if __name__ == "__main__":
    df = consolidar_metricas_detalladas(load_molecular_todos_los_datos())
    print(f"Registros: {len(df)}")
    export_thesis_focused_figures(df)
