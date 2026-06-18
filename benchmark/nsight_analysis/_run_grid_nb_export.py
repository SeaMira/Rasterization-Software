"""Execute benchmark_grid_yaml.ipynb code cells (non-interactive) to export compact thesis figures."""
import json
import os
import re
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

nb_path = Path("benchmark_grid_yaml.ipynb")
nb = json.loads(nb_path.read_text(encoding="utf-8"))

globs = {"display": lambda x=None, **k: None, "__name__": "__main__"}


def run():
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell.get("source", []))
        if not src.strip():
            continue
        if "pip install" in src:
            continue
        if "analizar_con_tiempos_reales(" in src and "def analizar_con_tiempos_reales" not in src:
            continue
        if "for regla in rules_to_analyze" in src:
            continue
        if "plotear_resultados_por_rango(" in src and "def plotear_resultados_por_rango" not in src:
            continue
        if "graficar_metrica_comparativa(" in src:
            continue
        if re.search(r"\bcs_1st_grid1_ranges\b", src) and "_fallback_todos_los_datos" not in src:
            continue
        if "rearrange[" in src and "guardar_resultados_json" in src:
            continue
        if "datos_para_analisis = {" in src:
            continue
        if "df_cs_1st_master" in src or "heatmap_data" in src:
            continue
        print(f"--- cell {i} ---", flush=True)
        try:
            exec(compile(src, f"cell_{i}", "exec"), globs)
        except Exception:
            traceback.print_exc()
            sys.exit(1)
    if "export_thesis_metric_figures" in globs and "df_detailed_metrics" in globs:
        print("--- export metrics ---", flush=True)
        globs["export_thesis_metric_figures"](globs["df_detailed_metrics"])
    if "export_all_time_per_mark" in globs and "df_summary_compact" not in globs:
        if "consolidar_datos_en_dataframe" in globs and "todos_los_datos" in globs:
            globs["df_summary_compact"] = globs["consolidar_datos_en_dataframe"](
                globs["todos_los_datos"]
            )
    if "export_all_time_per_mark" in globs and "df_summary_compact" in globs:
        print("--- export time_per_mark ---", flush=True)
        globs["export_all_time_per_mark"](globs["df_summary_compact"])

    plt.close("all")
    root = globs.get("FIG_ROOT")
    if root and Path(root).is_dir():
        n = sum(1 for _ in Path(root).rglob("*.png"))
        print(f"Exported {n} PNG files under {root}")


if __name__ == "__main__":
    run()
