"""Patch benchmark_grid_yaml.ipynb for compact-scene thesis export.

Focused molecular-scene figures (subchart filtering) live in _thesis_focused_export.py;
run via _run_mol_focused_export.py or the export cell in benchmark_mol_yaml.ipynb.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

NB_PATH = Path(__file__).resolve().parent / "benchmark_grid_yaml.ipynb"

JSON_CACHE_CELL = r'''
JSON_SUMMARIES_DIR = "json_summaries"
USE_JSON_SUMMARIES_CACHE = True


def ruta_json_resultados(experimento, carpeta=None):
    carpeta = carpeta or JSON_SUMMARIES_DIR
    return os.path.join(carpeta, f"resultados_{experimento.lower()}_ranges.json")


def cargar_experimento_desde_json_summaries(experimento, carpeta=None):
    path = ruta_json_resultados(experimento, carpeta)
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def construir_dict_experimentos_json_o_ram(keys_en_orden, proveedor_fallback_dict, usar_json=None, carpeta=None):
    usar_json = USE_JSON_SUMMARIES_CACHE if usar_json is None else usar_json
    carpeta = carpeta or JSON_SUMMARIES_DIR
    desde_json, desde_ram, faltantes = [], [], []
    fb_cache = None

    def fallback_dict():
        nonlocal fb_cache
        if fb_cache is None:
            fb_cache = proveedor_fallback_dict()
        return fb_cache

    out = {}
    for clave in keys_en_orden:
        if usar_json:
            datos = cargar_experimento_desde_json_summaries(clave, carpeta)
            if datos is not None:
                out[clave] = datos
                desde_json.append(clave)
                continue
            faltantes.append(clave)
        out[clave] = fallback_dict()[clave]
        desde_ram.append(clave)

    print(f"Experimentos desde JSON: {len(desde_json)} | desde RAM: {len(desde_ram)}")
    if faltantes:
        print(f"  JSON faltante (RAM): {faltantes}")
    return out


def parse_nombre_experimento(nombre_experimento):
    """Version + grid desde claves como CS_1st_grid1 o CU_HYB_BIN_grid2."""
    if nombre_experimento.startswith("CU_HYB_BIN_"):
        return "CU_HYB_BIN", nombre_experimento.replace("CU_HYB_BIN_", "")
    partes = nombre_experimento.split("_")
    if len(partes) >= 3:
        return f"{partes[0]}_{partes[1]}", partes[2]
    return partes[0], "Unknown"
'''

PLOT_EXPORT_CELL = r'''
from pathlib import Path
import re

FIG_ROOT = Path(r"D:\Users\Escritorio\U\Ot2026-2doSemM\tesis-magister-latex\img\chapter3\results\compact")
SAVE_FIGURES = True
FIG_DPI = 150
SHOW_PLOTS = False

VERSION_FOLDER = {
    "CS_1st": "fst_gpu",
    "CS_2nd": "scnd_gpu",
    "CS_std": "std",
    "CU_HYB_BIN": "hybrid",
}

RULE_FOLDER = {
    "GPU Engines Active [%]": "gpu_engines_active_pcnt",
    "GPU Engines Active": "gpu_engines_active",
    "Graphics/Compute Idle [%]": "graphics_compute_idle",
    "L1TEX L2 Hit Rates": "l2_hit_rates",
    "L1TEX Miss Sectors [%]": "l1_miss_sectors",
    "L1TEX Sectors [%]": "l1_sectors",
    "Unit Throughputs": "unit_throughput",
    "SM Warp Occupancy [Warps Per Cycle]": "sm_warp_occ",
    "SM Warp Occupancy [%]": "sm_warp_occ_pcnt",
    "SM Warp Issue Stalls [%]": "sm_warp_issue_stalls",
    "Cumulative Warp Latencies [%]": "cumulative_warp_lat_pcnt",
    "Cumulative Warp Latencies": "cumulative_warp_lat",
    "Active Threads Per Warp": "active_threads",
    "Warp Launch Stalled by Reasons [%]": "launch_stalled_reasons",
    "SM Throughputs": "sm_throughput",
    "Shader Pixels Coverage Kill": "shader_pix_coverage_kill",
}

SHADER_SLUG = {
    "Sphere Shader": "sphere_shader",
    "Cylinder Shader": "cylinder_shader",
    "Sphere Bbox Extraction Shader": "sphere_extr",
    "Sphere Bbox Intersection Shader": "sphere_intr",
    "Cylinder Bbox Extraction Shader": "cylinder_extr",
    "Cylinder Bbox Intersection Shader": "cylinder_intr",
    "Draw Spheres": "draw_spheres",
    "Draw Cylinders": "draw_cylinders",
    "Sphere: Classify": "sphere_classify",
    "Sphere: Sort+RLE+TileOffsets": "sphere_sort_rle_off",
    "Sphere: Expand WorkGroups": "sphere_expand_wg",
    "Sphere: Tiled Raster WG": "sphere_tile_rast",
    "Sphere: Small Raster": "sphere_small_rast",
    "Cylinder: Classify": "cylinder_classify",
    "Cylinder: Sort+RLE+TileOffsets": "cylinder_sort_rle_off",
    "Cylinder: Expand WorkGroups": "cylinder_expand_wg",
    "Cylinder: Tiled Raster WG": "cylinder_tile_rast",
}

EXPORT_MANIFEST = {
    "CS_1st": ["Sphere Shader", "Cylinder Shader"],
    "CS_2nd": [
        "Sphere Bbox Extraction Shader",
        "Sphere Bbox Intersection Shader",
        "Cylinder Bbox Extraction Shader",
        "Cylinder Bbox Intersection Shader",
    ],
    "CS_std": ["Draw Spheres", "Draw Cylinders"],
    "CU_HYB_BIN": [
        "Sphere: Classify",
        "Sphere: Sort+RLE+TileOffsets",
        "Sphere: Expand WorkGroups",
        "Sphere: Tiled Raster WG",
        "Sphere: Small Raster",
        "Cylinder: Classify",
        "Cylinder: Sort+RLE+TileOffsets",
        "Cylinder: Expand WorkGroups",
        "Cylinder: Tiled Raster WG",
    ],
}

EXPORT_RULES = list(RULE_FOLDER.keys())


def thesis_figure_path(version_target, rule_name, shader_target):
    vf = VERSION_FOLDER.get(version_target)
    rf = RULE_FOLDER.get(rule_name)
    sf = SHADER_SLUG.get(shader_target)
    if not (vf and rf and sf):
        return None
    return FIG_ROOT / vf / rf / f"{sf}.png"


def save_figure(fig, path):
    if not SAVE_FIGURES or path is None:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    return path
'''

GRAFICAR_CELL = r'''
def filtrar_grid_insuficientes(df, min_rangos=4):
    conteo_rangos = df.groupby(
        ['Version', 'Shader', 'Rule_Name', 'Metric_Name', 'Grid']
    )['Dataset'].transform('nunique')
    return df[conteo_rangos >= min_rangos].copy()


def graficar_regla_por_columnas(
    df,
    version_target,
    shader_target,
    rule_name='L1TEX L2 Hit Rates',
    min_rangos=4,
    save_path=None,
    show=None,
):
    mask = (df['Version'] == version_target) & \
           (df['Shader'] == shader_target) & \
           (df['Rule_Name'] == rule_name)
    data_plot = df[mask].copy()
    if data_plot.empty:
        print(f"⚠️ No hay datos iniciales para: {rule_name} / {shader_target}")
        return None

    skip_quality_filters = version_target == "CU_HYB_BIN"
    effective_min = 3 if version_target == "CS_2nd" else min_rangos

    if not skip_quality_filters:
        data_plot = filtrar_grid_insuficientes(data_plot, min_rangos=effective_min)
        if data_plot.empty:
            print(f"⚠️ Se filtraron todos los datos (mínimo {effective_min} rangos por grid).")
            return None
        metric_max_values = data_plot.groupby('Metric_Name')['Value_Avg'].apply(lambda x: x.abs().max())
        metrics_to_keep = metric_max_values[metric_max_values > 1e-9].index.tolist()
        data_plot = data_plot[data_plot['Metric_Name'].isin(metrics_to_keep)].copy()
        if data_plot.empty:
            print("ℹ️ Todas las métricas restantes son cero.")
            return None
    else:
        metrics_to_keep = sorted(data_plot['Metric_Name'].unique(), key=str)

    data_plot['Dataset_Num'] = data_plot['Dataset'].str.extract(r'(\d+)').astype(int)
    data_plot = data_plot.sort_values('Dataset_Num')

    g = sns.catplot(
        data=data_plot,
        x="Dataset",
        y="Value_Avg",
        hue="Grid",
        col="Metric_Name",
        col_wrap=3,
        kind="point",
        sharey=False,
        height=8,
        aspect=1.2,
        palette="viridis",
        markers="o",
        linestyles="-",
    )
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle(
        f"Análisis: '{rule_name}' ({version_target})\nShader {shader_target}",
        fontsize=15,
    )
    g.set_axis_labels("Rango", "Valor")
    g.set_titles("{col_name}")

    if save_path is None and SAVE_FIGURES:
        save_path = thesis_figure_path(version_target, rule_name, shader_target)
    saved = save_figure(g.fig, save_path)
    if saved:
        print(f"  → guardado {saved}")

    do_show = SHOW_PLOTS if show is None else show
    if do_show:
        plt.show()
    else:
        plt.close(g.fig)
    return g.fig


def export_thesis_metric_figures(df):
    if not SAVE_FIGURES:
        print("SAVE_FIGURES=False; omitiendo exportación de métricas.")
        return
    n = 0
    for version, shaders in EXPORT_MANIFEST.items():
        for shader in shaders:
            for regla in EXPORT_RULES:
                fig = graficar_regla_por_columnas(
                    df, version, shader, regla, save_path=None, show=False
                )
                if fig is not None:
                    n += 1
    print(f"Exportadas {n} figuras de métricas bajo {FIG_ROOT}")


def plot_time_per_mark_compact(df_summary, version_target, save_path=None, show=None):
    """Frame time por shader, todas las grids en un panel (hue=Grid)."""
    sub = df_summary[df_summary["Version"] == version_target].copy()
    if sub.empty:
        print(f"Sin datos de tiempo para {version_target}")
        return None
    sub["Dataset_Num"] = sub["Dataset"].str.extract(r"(\d+)").astype(int)
    sub = sub.sort_values("Dataset_Num")
    g = sns.catplot(
        data=sub,
        x="Dataset",
        y="Duration_ms",
        hue="Grid",
        col="Shader_Name",
        col_wrap=3,
        kind="point",
        sharey=False,
        height=6,
        aspect=1.2,
        palette="viridis",
        markers="o",
        linestyles="-",
    )
    g.fig.subplots_adjust(top=0.88)
    g.fig.suptitle(f"Frame time per marked range — {version_target} (compact grids)", fontsize=14)
    g.set_axis_labels("Range", "Duration (ms)")
    g.set_titles("{col_name}")

    if save_path is None and SAVE_FIGURES:
        vf = VERSION_FOLDER.get(version_target, version_target.lower())
        save_path = FIG_ROOT / vf / "time_per_mark.png"
    saved = save_figure(g.fig, save_path)
    if saved:
        print(f"  → time_per_mark {saved}")

    do_show = SHOW_PLOTS if show is None else show
    if do_show:
        plt.show()
    else:
        plt.close(g.fig)
    return g.fig


def export_all_time_per_mark(df_summary):
    for version in EXPORT_MANIFEST:
        plot_time_per_mark_compact(df_summary, version, show=False)
'''

CU_HYB_GRID_CELL = r'''
cu_hyb_bin_grid1_ranges = []
for i in range(5):
    cu_hyb_bin_grid1_ranges.append(
        analizar_con_tiempos_reales(
            path_archivos["CU_HYB_BIN"]["grid1"][f"range{i+1}"],
            path_frames["CU_HYB_BIN"]["grid1"][f"range{i+1}"],
            6,
        )
    )
rearrange = {f"range{k+1}": cu_hyb_bin_grid1_ranges[k] for k in range(5)}
guardar_resultados_json(rearrange, nombre_archivo="resultados_cu_hyb_bin_grid1_ranges.json")
'''

CU_HYB_GRID2_CELL = r'''
cu_hyb_bin_grid2_ranges = []
for i in range(5):
    cu_hyb_bin_grid2_ranges.append(
        analizar_con_tiempos_reales(
            path_archivos["CU_HYB_BIN"]["grid2"][f"range{i+1}"],
            path_frames["CU_HYB_BIN"]["grid2"][f"range{i+1}"],
            6,
        )
    )
rearrange = {f"range{k+1}": cu_hyb_bin_grid2_ranges[k] for k in range(5)}
guardar_resultados_json(rearrange, nombre_archivo="resultados_cu_hyb_bin_grid2_ranges.json")
'''

CU_HYB_GRID3_CELL = r'''
cu_hyb_bin_grid3_ranges = []
for i in range(5):
    cu_hyb_bin_grid3_ranges.append(
        analizar_con_tiempos_reales(
            path_archivos["CU_HYB_BIN"]["grid3"][f"range{i+1}"],
            path_frames["CU_HYB_BIN"]["grid3"][f"range{i+1}"],
            6,
        )
    )
rearrange = {f"range{k+1}": cu_hyb_bin_grid3_ranges[k] for k in range(5)}
guardar_resultados_json(rearrange, nombre_archivo="resultados_cu_hyb_bin_grid3_ranges.json")
'''

CU_HYB_GRID4_CELL = r'''
cu_hyb_bin_grid4_ranges = []
for i in range(5):
    cu_hyb_bin_grid4_ranges.append(
        analizar_con_tiempos_reales(
            path_archivos["CU_HYB_BIN"]["grid4"][f"range{i+1}"],
            path_frames["CU_HYB_BIN"]["grid4"][f"range{i+1}"],
            6,
        )
    )
rearrange = {f"range{k+1}": cu_hyb_bin_grid4_ranges[k] for k in range(5)}
guardar_resultados_json(rearrange, nombre_archivo="resultados_cu_hyb_bin_grid4_ranges.json")
'''

TODOS_LOS_DATOS_CELL = r'''
_KEYS_ALL = [
    "CS_1st_grid1", "CS_1st_grid2", "CS_1st_grid3", "CS_1st_grid4",
    "CS_2nd_grid1", "CS_2nd_grid2", "CS_2nd_grid3",
    "CS_std_grid1", "CS_std_grid2", "CS_std_grid3", "CS_std_grid4",
    "CU_HYB_BIN_grid1", "CU_HYB_BIN_grid2", "CU_HYB_BIN_grid3", "CU_HYB_BIN_grid4",
]


def _fallback_todos_los_datos():
    return {
        "CS_1st_grid1": cs_1st_grid1_ranges,
        "CS_1st_grid2": cs_1st_grid2_ranges,
        "CS_1st_grid3": cs_1st_grid3_ranges,
        "CS_1st_grid4": cs_1st_grid4_ranges,
        "CS_2nd_grid1": cs_2nd_grid1_ranges,
        "CS_2nd_grid2": cs_2nd_grid2_ranges,
        "CS_2nd_grid3": cs_2nd_grid3_ranges,
        "CS_std_grid1": cs_std_grid1_ranges,
        "CS_std_grid2": cs_std_grid2_ranges,
        "CS_std_grid3": cs_std_grid3_ranges,
        "CS_std_grid4": cs_std_grid4_ranges,
        "CU_HYB_BIN_grid1": cu_hyb_bin_grid1_ranges,
        "CU_HYB_BIN_grid2": cu_hyb_bin_grid2_ranges,
        "CU_HYB_BIN_grid3": cu_hyb_bin_grid3_ranges,
        "CU_HYB_BIN_grid4": cu_hyb_bin_grid4_ranges,
    }


todos_los_datos = construir_dict_experimentos_json_o_ram(_KEYS_ALL, _fallback_todos_los_datos)
df_detailed_metrics = consolidar_metricas_detalladas(todos_los_datos)
print(f"DataFrame generado con {len(df_detailed_metrics)} registros de métricas.")
display(df_detailed_metrics.head())
display(df_detailed_metrics["Rule_Name"].unique())
'''

CONSOLIDAR_CELL = r'''
# --- EXTENSIÓN: ANÁLISIS AVANZADO CON PANDAS ---

def consolidar_datos_en_dataframe(diccionario_datos_cargados):
    filas = []
    for nombre_experimento, lista_rangos in diccionario_datos_cargados.items():
        version, grid = parse_nombre_experimento(nombre_experimento)
        if isinstance(lista_rangos, dict):
            iter_rangos = enumerate(lista_rangos.values())
        else:
            iter_rangos = enumerate(lista_rangos)
        for i, datos_rango in iter_rangos:
            nombre_dataset = f"range{i+1}"
            for item in datos_rango:
                filas.append({
                    'Experimento': nombre_experimento,
                    'Version': version,
                    'Grid': grid,
                    'Dataset': nombre_dataset,
                    'Shader_Name': item.get('Range', 'Unknown'),
                    'Duration_ms': item.get('Real_Duration_ms_Avg', 0),
                    'Rel_Duration': item.get('RelativeFrameDuration_Avg', 0),
                    'Muestras': item.get('Total_Muestras', 0),
                })
    return pd.DataFrame(filas)


def consolidar_metricas_detalladas(diccionario_datos_cargados):
    filas = []
    for nombre_experimento, data_rangos in diccionario_datos_cargados.items():
        version, grid = parse_nombre_experimento(nombre_experimento)
        if isinstance(data_rangos, list):
            iterador = enumerate(data_rangos)
        elif isinstance(data_rangos, dict):
            iterador = enumerate(data_rangos.values())
        else:
            continue
        for i, lista_shaders in iterador:
            nombre_range = f"range{i+1}"
            for shader_data in lista_shaders:
                shader_name = shader_data.get('Range', 'Unknown')
                for regla in shader_data.get('Rules', []):
                    regla_nombre = regla.get('Name')
                    for metrica in regla.get('Metrics', []):
                        filas.append({
                            'Experimento': nombre_experimento,
                            'Version': version,
                            'Grid': grid,
                            'Dataset': nombre_range,
                            'Shader': shader_name,
                            'Rule_Name': regla_nombre,
                            'Metric_Name': metrica.get('Name'),
                            'Metric_Id': metrica.get('Id'),
                            'Value_Avg': metrica.get('Value_Avg', 0),
                            'Value_Min': metrica.get('Value_Min', 0),
                            'Value_Max': metrica.get('Value_Max', 0),
                        })
    return pd.DataFrame(filas)
'''.strip()

EXPORT_RUN_CELL = r'''
df_summary_compact = consolidar_datos_en_dataframe(todos_los_datos)
export_thesis_metric_figures(df_detailed_metrics)
export_all_time_per_mark(df_summary_compact)
print("Thesis compact export complete:", FIG_ROOT)
'''


def _cell_src(cell) -> str:
    return "".join(cell.get("source", []))


def _set_cell_src(cell, src: str) -> None:
    cell["source"] = [line + "\n" for line in src.split("\n")]
    if cell["source"]:
        cell["source"][-1] = cell["source"][-1].rstrip("\n")


def patch_notebook() -> None:
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    cells = nb["cells"]

    # versiones
    for cell in cells:
        src = _cell_src(cell)
        if 'versiones = ["CS_1st"' in src or "versiones = " in src and "path_to_files" in src:
            src = re.sub(
                r'versiones = \[.*?\]',
                'versiones = ["CS_1st", "CS_2nd", "CS_std", "CU_HYB_BIN"]',
                src,
                count=1,
                flags=re.DOTALL,
            )
            _set_cell_src(cell, src)

    # JSON cache helpers after guardar_resultados_json cell
    inserted = False
    for i, cell in enumerate(cells):
        if "def guardar_resultados_json" in _cell_src(cell):
            if "USE_JSON_SUMMARIES_CACHE" not in _cell_src(cells[i + 1] if i + 1 < len(cells) else cell):
                cells.insert(i + 1, {"cell_type": "code", "metadata": {}, "outputs": [], "source": []})
                _set_cell_src(cells[i + 1], JSON_CACHE_CELL.strip())
                inserted = True
            break
    if not inserted:
        print("WARN: JSON cache cell not inserted")

    # CU_HYB_BIN ingestion after cs_std_grid4 save
    for i, cell in enumerate(cells):
        if "resultados_cs_std_grid4_ranges.json" in _cell_src(cell):
            if not any("cu_hyb_bin_grid1_ranges" in _cell_src(c) for c in cells):
                insert_at = i + 1
                for block in (CU_HYB_GRID_CELL, CU_HYB_GRID2_CELL, CU_HYB_GRID3_CELL, CU_HYB_GRID4_CELL):
                    cells.insert(insert_at, {"cell_type": "code", "metadata": {}, "outputs": [], "source": []})
                    _set_cell_src(cells[insert_at], block.strip())
                    insert_at += 1
            break

    # consolidar helpers: replace whole cell (parse_nombre lives in JSON-cache cell)
    for cell in cells:
        if "def consolidar_datos_en_dataframe" in _cell_src(cell):
            _set_cell_src(cell, CONSOLIDAR_CELL)
            break

    # todos_los_datos cell
    for cell in cells:
        if 'todos_los_datos = {' in _cell_src(cell) and "df_detailed_metrics" in _cell_src(cell):
            _set_cell_src(cell, TODOS_LOS_DATOS_CELL.strip())
            break

    # plot export config + graficar before first graficar cell
    graficar_idx = None
    for i, cell in enumerate(cells):
        if "def graficar_regla_por_columnas" in _cell_src(cell):
            graficar_idx = i
            break
    if graficar_idx is not None:
        if "FIG_ROOT" not in _cell_src(cells[graficar_idx - 1] if graficar_idx > 0 else cells[graficar_idx]):
            cells.insert(graficar_idx, {"cell_type": "code", "metadata": {}, "outputs": [], "source": []})
            _set_cell_src(cells[graficar_idx], PLOT_EXPORT_CELL.strip())
            graficar_idx += 1
        _set_cell_src(cells[graficar_idx], GRAFICAR_CELL.strip())

    # export run cell at end
    if not any("export_thesis_metric_figures" in _cell_src(c) for c in cells):
        cells.append({"cell_type": "code", "metadata": {}, "outputs": [], "source": []})
        _set_cell_src(cells[-1], EXPORT_RUN_CELL.strip())

    NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Patched {NB_PATH}")


if __name__ == "__main__":
    patch_notebook()
