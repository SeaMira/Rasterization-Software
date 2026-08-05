"""Export thesis figures with focused subcharts (molecular scenes)."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

THESIS_FIG_ROOT = Path(
    r"D:\Users\Escritorio\U\Ot2026-2doSemM\tesis-magister-latex\img\chapter3\results"
)
SAVE_FIGURES = True
FIG_DPI = 150
SHOW_PLOTS = False

from _thesis_chart_style import apply_thesis_chart_style, panel_col_wrap  # noqa: E402

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
    "Sphere Bbox Extraction Shader": "sphere_extr",
    "Sphere Bbox Intersection Shader": "sphere_intr",
    "Draw Spheres": "draw_spheres",
    "Sphere: Classify": "sphere_classify",
    "Sphere: Sort+RLE+TileOffsets": "sphere_sort_rle_off",
    "Sphere: Expand WorkGroups": "sphere_expand_wg",
    "Sphere: Tiled Raster WG": "sphere_tile_rast",
    "Sphere: Small Raster": "sphere_small_rast",
}

_GPU_ENGINES_PCNT = ["GR Cycles Active [%]", "Engine Active Copy Async [%]"]
_SM_WARP_OCC_COMPUTE = ["Active Warps per Cycle All"]
_SM_WARP_OCC_DRAW = [
    "Active Warps per Cycle All",
    "Active Warps per Cycle VTG",
    "Active Warps per Cycle PS",
]
_WARP_LAUNCH_HYBRID_CS = ["CS Warp Launch Stalled Warp Slot Allocation [%]"]
_WARP_LAUNCH_STD_DRAW = [
    "PS Warp Launch Stalled TRAM Fill [%]",
    "VTG Warp Launch Stalled ISBE Allocation [%]",
    "PS Warp Launch Stalled OOO Warp Completion [%]",
]
_SM_THROUGHPUT_DRAW = [
    "SM Pipe FMA Active [%]",
    "SM Pipe SFU Active [%]",
    "SM Issue Active [%]",
]
_SM_THROUGHPUT_INTR = [
    "SM Pipe SFU Active [%]",
    "SM Issue Active [%]",
    "SM Pipe ALU Active [%]",
]
_SM_THROUGHPUT_HYBRID_HEAVY = ["SM Issue Active [%]", "SM Pipe FMA Active [%]"]
_L1_SECTORS_1ST = [
    "L1TEX Tag-Stage Sectors Global Atom [%]",
    "L1TEX Tag-Stage Sectors Surface Store [%]",
    "L1TEX Tag-Stage Sectors Global Load [%]",
]
_UNIT_TP_1ST = [
    "PCIe Throughput [%]",
    "VRAM Throughput [%]",
    "SM Pipe FMA Active [%]",
    "SM Issue Active [%]",
]
_ISSUE_STALLS_1ST = [
    "Warps Issue Stalled Long Scoreboard L1 [%]",
    "Warps Issue Stalled Not Selected [%]",
    "Warps Issue Stalled LG Throttle [%]",
]
_ISSUE_STALLS_2ND_EXTR = [
    "Warps Issue Stalled Long Scoreboard L1 [%]",
    "Warps Issue Stalled LG Throttle [%]",
]
_ISSUE_STALLS_2ND_INTR = [
    "Warps Issue Stalled Wait [%]",
    "Warps Issue Stalled Short Scoreboard [%]",
    "Warps Issue Stalled Not Selected [%]",
]
_CUMULATIVE_WARP_DRAW = [
    "Cumulative Warp Latency VTG [%]",
    "Cumulative Warp Latency PS [%]",
]
_SM_THROUGHPUT_1ST = ["SM Pipe FMA Active [%]", "SM Issue Active [%]"]

_HYBRID_SPHERE_STAGES = [
    "Sphere: Classify",
    "Sphere: Sort+RLE+TileOffsets",
    "Sphere: Expand WorkGroups",
    "Sphere: Tiled Raster WG",
    "Sphere: Small Raster",
]


def _entries(
    version: str,
    shader: str,
    rule: str,
    metrics: list[str] | None,
) -> tuple[tuple[str, str, str], list[str] | None]:
    return (version, shader, rule), metrics


def build_thesis_focused() -> dict[tuple[str, str, str], list[str] | None]:
    out: dict[tuple[str, str, str], list[str] | None] = {}

    def add(version, shader, rule, metrics):
        out[(version, shader, rule)] = metrics

    # --- First GPGPU ---
    add("CS_1st", "Sphere Shader", "GPU Engines Active [%]", _GPU_ENGINES_PCNT)
    add("CS_1st", "Sphere Shader", "L1TEX L2 Hit Rates", None)
    add("CS_1st", "Sphere Shader", "L1TEX Sectors [%]", _L1_SECTORS_1ST)
    add("CS_1st", "Sphere Shader", "Unit Throughputs", _UNIT_TP_1ST)
    add("CS_1st", "Sphere Shader", "SM Warp Occupancy [Warps Per Cycle]", _SM_WARP_OCC_COMPUTE)
    add("CS_1st", "Sphere Shader", "SM Warp Issue Stalls [%]", _ISSUE_STALLS_1ST)
    add("CS_1st", "Sphere Shader", "Active Threads Per Warp", None)
    add("CS_1st", "Sphere Shader", "Warp Launch Stalled by Reasons [%]", None)
    add("CS_1st", "Sphere Shader", "SM Throughputs", _SM_THROUGHPUT_1ST)

    # --- Second GPGPU ---
    for shader in ("Sphere Bbox Extraction Shader", "Sphere Bbox Intersection Shader"):
        add("CS_2nd", shader, "GPU Engines Active [%]", _GPU_ENGINES_PCNT)
        add("CS_2nd", shader, "L1TEX L2 Hit Rates", None)
        add("CS_2nd", shader, "L1TEX Sectors [%]", None)
        add("CS_2nd", shader, "Unit Throughputs", None)
        add("CS_2nd", shader, "SM Warp Occupancy [Warps Per Cycle]", _SM_WARP_OCC_COMPUTE)
        add("CS_2nd", shader, "Active Threads Per Warp", None)
        add("CS_2nd", shader, "Warp Launch Stalled by Reasons [%]", None)
    add("CS_2nd", "Sphere Bbox Extraction Shader", "SM Warp Issue Stalls [%]", _ISSUE_STALLS_2ND_EXTR)
    add("CS_2nd", "Sphere Bbox Intersection Shader", "SM Warp Issue Stalls [%]", _ISSUE_STALLS_2ND_INTR)
    add("CS_2nd", "Sphere Bbox Intersection Shader", "SM Throughputs", _SM_THROUGHPUT_INTR)

    # --- Standard (Draw Spheres only) ---
    add("CS_std", "Draw Spheres", "GPU Engines Active [%]", _GPU_ENGINES_PCNT)
    add("CS_std", "Draw Spheres", "L1TEX L2 Hit Rates", None)
    add("CS_std", "Draw Spheres", "L1TEX Sectors [%]", None)
    add("CS_std", "Draw Spheres", "Unit Throughputs", None)
    add("CS_std", "Draw Spheres", "SM Warp Occupancy [Warps Per Cycle]", _SM_WARP_OCC_DRAW)
    add("CS_std", "Draw Spheres", "Cumulative Warp Latencies [%]", _CUMULATIVE_WARP_DRAW)
    add("CS_std", "Draw Spheres", "Active Threads Per Warp", None)
    add("CS_std", "Draw Spheres", "Shader Pixels Coverage Kill", None)
    add("CS_std", "Draw Spheres", "Warp Launch Stalled by Reasons [%]", _WARP_LAUNCH_STD_DRAW)
    add("CS_std", "Draw Spheres", "SM Throughputs", _SM_THROUGHPUT_DRAW)

    # --- Hybrid (sphere stages) ---
    for stage in _HYBRID_SPHERE_STAGES:
        if stage != "Sphere: Expand WorkGroups":
            add("CU_HYB_BIN", stage, "GPU Engines Active [%]", _GPU_ENGINES_PCNT)
            add("CU_HYB_BIN", stage, "L1TEX L2 Hit Rates", None)
        add("CU_HYB_BIN", stage, "L1TEX Sectors [%]", None)
        add("CU_HYB_BIN", stage, "Unit Throughputs", None)
        add("CU_HYB_BIN", stage, "SM Warp Occupancy [Warps Per Cycle]", _SM_WARP_OCC_COMPUTE)
        add("CU_HYB_BIN", stage, "Active Threads Per Warp", None)

    for stage in ("Sphere: Tiled Raster WG", "Sphere: Small Raster"):
        add("CU_HYB_BIN", stage, "Graphics/Compute Idle [%]", None)

    for stage in (
        "Sphere: Classify",
        "Sphere: Sort+RLE+TileOffsets",
        "Sphere: Expand WorkGroups",
        "Sphere: Tiled Raster WG",
    ):
        add("CU_HYB_BIN", stage, "Warp Launch Stalled by Reasons [%]", _WARP_LAUNCH_HYBRID_CS)

    for stage in (
        "Sphere: Sort+RLE+TileOffsets",
        "Sphere: Expand WorkGroups",
        "Sphere: Tiled Raster WG",
        "Sphere: Small Raster",
    ):
        add("CU_HYB_BIN", stage, "SM Throughputs", _SM_THROUGHPUT_HYBRID_HEAVY)

    return out


THESIS_FOCUSED = build_thesis_focused()


def parse_nombre_experimento(nombre_experimento: str) -> tuple[str, str]:
    key = nombre_experimento.lower()
    prefixes = (
        ("cu_hyb_bin_", "CU_HYB_BIN"),
        ("cs_1st_", "CS_1st"),
        ("cs_2nd_", "CS_2nd"),
        ("cs_std_", "CS_std"),
    )
    for prefix, version in prefixes:
        if key.startswith(prefix):
            return version, nombre_experimento[len(prefix) :]
    if nombre_experimento.startswith("CU_HYB_BIN_"):
        return "CU_HYB_BIN", nombre_experimento.replace("CU_HYB_BIN_", "")
    partes = nombre_experimento.split("_")
    if len(partes) >= 3:
        return f"{partes[0]}_{partes[1]}", partes[2]
    return partes[0], "Unknown"


def consolidar_metricas_detalladas(diccionario_datos_cargados: dict) -> pd.DataFrame:
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
                shader_name = shader_data.get("Range", "Unknown")
                for regla in shader_data.get("Rules", []):
                    regla_nombre = regla.get("Name")
                    for metrica in regla.get("Metrics", []):
                        filas.append(
                            {
                                "Experimento": nombre_experimento,
                                "Version": version,
                                "Grid": grid,
                                "Dataset": nombre_range,
                                "Shader": shader_name,
                                "Rule_Name": regla_nombre,
                                "Metric_Name": metrica.get("Name"),
                                "Metric_Id": metrica.get("Id"),
                                "Value_Avg": metrica.get("Value_Avg", 0),
                                "Value_Min": metrica.get("Value_Min", 0),
                                "Value_Max": metrica.get("Value_Max", 0),
                            }
                        )
    return pd.DataFrame(filas)


def load_molecular_todos_los_datos(json_dir: Path | None = None) -> dict:
    json_dir = json_dir or Path(__file__).resolve().parent / "json_summaries"
    out = {}
    for mol in ("1aga", "1c0o", "2mjq", "8wql"):
        for ver in ("cs_1st", "cs_2nd", "cs_std", "cu_hyb_bin"):
            key = f"{ver}_{mol}"
            path = json_dir / f"resultados_{key}_ranges.json"
            if path.is_file():
                out[key] = json.loads(path.read_text(encoding="utf-8"))
    return out


def thesis_figure_path(
    version_target: str,
    rule_name: str,
    shader_target: str,
    *,
    subdir: str = "focused",
) -> Path | None:
    vf = VERSION_FOLDER.get(version_target)
    rf = RULE_FOLDER.get(rule_name)
    sf = SHADER_SLUG.get(shader_target)
    if not (vf and rf and sf):
        return None
    base = THESIS_FIG_ROOT / vf / rf
    if subdir:
        base = base / subdir
    return base / f"{sf}.png"


def save_figure(fig, path: Path | None):
    if not SAVE_FIGURES or path is None:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    return path


def filtrar_grid_insuficientes(df: pd.DataFrame, min_rangos: int = 4) -> pd.DataFrame:
    conteo_rangos = df.groupby(
        ["Version", "Shader", "Rule_Name", "Metric_Name", "Grid"]
    )["Dataset"].transform("nunique")
    return df[conteo_rangos >= min_rangos].copy()


def _resolve_metrics(
    data_plot: pd.DataFrame,
    metric_names: list[str] | None,
    *,
    min_pct: float = 0.0,
) -> list[str]:
    available = set(data_plot["Metric_Name"].unique())
    if metric_names is not None:
        keep = [m for m in metric_names if m in available]
        if not keep:
            metric_max = data_plot.groupby("Metric_Name")["Value_Avg"].apply(
                lambda x: x.abs().max()
            )
            keep = metric_max[metric_max > 1e-9].index.tolist()
        return sorted(keep, key=str)

    metric_max = data_plot.groupby("Metric_Name")["Value_Avg"].apply(lambda x: x.abs().max())
    mask = metric_max > 1e-9
    if min_pct > 0:
        mask &= metric_max >= min_pct
    return sorted(metric_max[mask].index.tolist(), key=str)


def graficar_regla_por_columnas(
    df: pd.DataFrame,
    version_target: str,
    shader_target: str,
    rule_name: str = "L1TEX L2 Hit Rates",
    min_rangos: int = 4,
    metric_names: list[str] | None = None,
    save_path: Path | None = None,
    show: bool | None = None,
):
    mask = (
        (df["Version"] == version_target)
        & (df["Shader"] == shader_target)
        & (df["Rule_Name"] == rule_name)
    )
    data_plot = df[mask].copy()
    if data_plot.empty:
        print(f"WARN: No hay datos: {rule_name} / {shader_target}")
        return None

    skip_quality_filters = version_target == "CU_HYB_BIN"
    effective_min = 3 if version_target == "CS_2nd" else min_rangos

    if not skip_quality_filters:
        data_plot = filtrar_grid_insuficientes(data_plot, min_rangos=effective_min)
        if data_plot.empty:
            print(f"WARN: Filtrado por rangos ({effective_min}): {rule_name} / {shader_target}")
            return None

    metrics_to_keep = _resolve_metrics(data_plot, metric_names)
    if not metrics_to_keep:
        print(f"INFO: Sin metricas utiles: {rule_name} / {shader_target}")
        return None
    data_plot = data_plot[data_plot["Metric_Name"].isin(metrics_to_keep)].copy()

    data_plot["Dataset_Num"] = data_plot["Dataset"].str.extract(r"(\d+)").astype(int)
    data_plot = data_plot.sort_values("Dataset_Num")

    n_panels = len(metrics_to_keep)
    col_wrap, legend_out = panel_col_wrap(n_panels)
    g = sns.catplot(
        data=data_plot,
        x="Dataset",
        y="Value_Avg",
        hue="Grid",
        col="Metric_Name",
        col_wrap=col_wrap,
        kind="point",
        sharey=False,
        height=8,
        aspect=1.2,
        palette="viridis",
        markers="o",
        linestyles="-",
        legend_out=legend_out,
    )
    apply_thesis_chart_style(
        g,
        n_panels=n_panels,
        col_wrap=col_wrap,
        legend_out=legend_out,
        suptitle=f"Analysis: {rule_name} ({version_target})\nShader: {shader_target}",
    )

    if save_path is None and SAVE_FIGURES:
        save_path = thesis_figure_path(version_target, rule_name, shader_target)
    saved = save_figure(g.fig, save_path)
    if saved:
        print(f"  -> {saved} ({len(metrics_to_keep)} paneles)")

    do_show = SHOW_PLOTS if show is None else show
    if do_show:
        plt.show()
    else:
        plt.close(g.fig)
    return g.fig


def export_thesis_focused_figures(df: pd.DataFrame) -> int:
    if not SAVE_FIGURES:
        print("SAVE_FIGURES=False; omitiendo export focused.")
        return 0
    n = 0
    for (version, shader, rule), metric_names in THESIS_FOCUSED.items():
        fig = graficar_regla_por_columnas(
            df,
            version,
            shader,
            rule,
            metric_names=metric_names,
            save_path=thesis_figure_path(version, rule, shader),
            show=False,
        )
        if fig is not None:
            n += 1
    print(f"Exportadas {n} figuras focused bajo {THESIS_FIG_ROOT}")
    return n


def main():
    os.environ.setdefault("MPLBACKEND", "Agg")
    datos = load_molecular_todos_los_datos()
    df = consolidar_metricas_detalladas(datos)
    print(f"Registros: {len(df)}")
    export_thesis_focused_figures(df)


if __name__ == "__main__":
    main()
