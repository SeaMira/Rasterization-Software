"""Generate LaTeX appendix with all OOC result figures under FIG_ROOT."""
from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

FIG_ROOT = Path(
    r"D:\Users\Escritorio\U\Ot2026-2doSemM\tesis-magister-latex\img\chapter3\results\ooc"
)
OUTPUT = Path(
    r"D:\Users\Escritorio\U\Ot2026-2doSemM\tesis-magister-latex\chapters\apendix_ooc_results.tex"
)

SCENE_LABELS = {
    "g10m": "G10M",
    "g50m": "G50M",
    "g100m": "G100M",
    "g500m": "G500M",
    "8wql": "8WQL",
}

SECTION_TITLES = {
    "nsight/stage_fraction": "Pipeline stage fractions",
    "nsight/stage_heatmap": "Pipeline stage heatmaps",
    "nsight/gr_active": "GR cycles active by stage and ring",
    "nsight/gr_idle": "GPU idle by stage and ring",
    "nsight/l1tex_hit": "L1TEX hit rate by stage and ring",
    "nsight/pcie": "PCIe throughput by stage and ring",
    "nsight/warp_occupancy": "Warp occupancy by stage and ring",
    "nsight/stall_breakdown": "Warp stall breakdown by ring",
    "nsight/metrics_vs_ring": "GPU metrics vs camera ring",
    "nsight/correlation": "Metric correlation by pipeline stage",
    "nsight/memory_bound": "Memory-bound vs compute-bound scatter",
    "nsight/frame_ms_by_ring": "Nsight frame time by camera ring",
    "nsight/panels": "Composite vertical panels",
    "cross_scene": "Cross-scene metric comparisons",
    "batch": "Batch runtime summaries",
    "batch/streaming_by_visibility": "Peak block requests vs batch\\_index",
    "batch/occlusion_by_visibility": "Occlusion ratio at fixed views (by avg_numVisible)",
    "batch/timeseries": "Batch timeseries",
    "preprocess": "Preprocess metrics",
    "package": "PACKAGE scale analysis",
}


def _slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()


def _scene_from_stem(stem: str) -> str | None:
    for key, label in SCENE_LABELS.items():
        if stem == key or stem.startswith(f"{key}_") or stem.endswith(f"_{key}"):
            return label
        if f"_{key}_" in stem or stem.endswith(f"_{key}"):
            return label
    for key, label in SCENE_LABELS.items():
        if key in stem:
            return label
    return None


def _caption(rel: Path) -> str:
    parts = rel.as_posix().split("/")
    stem = rel.stem
    scene = _scene_from_stem(stem)

    if "stage_fraction" in parts:
        return f"Renormalised pipeline-stage fractions for {scene or stem} by camera ring."
    if "stage_heatmap" in parts:
        return f"Stage-fraction heatmap for {scene or stem}."
    if "gr_active" in parts:
        return f"GR cycles active [{scene or stem}] by pipeline stage and camera ring."
    if "gr_idle" in parts:
        return f"GPU idle [{scene or stem}] by pipeline stage and camera ring."
    if "l1tex_hit" in parts:
        return f"L1TEX hit rate [{scene or stem}] by pipeline stage and camera ring."
    if "pcie" in parts:
        return f"PCIe throughput [{scene or stem}] by pipeline stage and camera ring."
    if "warp_occupancy" in parts:
        return f"Warp occupancy [{scene or stem}] by pipeline stage and camera ring."
    if "stall_breakdown" in parts:
        m = re.search(r"ring(\d+)", stem)
        ring = m.group(1) if m else "?"
        return f"Warp stall breakdown [{scene or stem}], camera ring {ring}."
    if "metrics_vs_ring" in parts and "panels" not in parts:
        metric = stem.split("_", 1)[-1].replace("_", " ") if "_" in stem else stem
        return f"Metric trend vs camera ring [{scene or stem}]: {metric}."
    if "correlation" in parts and "panels" not in parts:
        stage = stem.split("_", 1)[-1].replace("_", " ") if "_" in stem else stem
        return f"Metric correlation [{scene or stem}] for stage {stage}."
    if "memory_bound" in parts:
        return f"PCIe throughput vs SM issue active [{scene or stem}]."
    if "frame_ms_by_ring" in parts:
        if stem == "cross_scene":
            return (
                "Mean Nsight Graphics frame time by camera ring; grouped bars compare "
                "all scenes (error bars: std over frames and positional replicas)."
            )
        return (
            f"Mean Nsight Graphics frame time by camera ring [{scene or stem}] "
            f"(error bars: std over frames and positional replicas)."
        )
    if "panels" in parts:
        return f"Composite vertical panel [{stem.replace('_', ' ')}]."
    if "cross_scene" in parts:
        return f"Cross-scene comparison: {stem.replace('_', ' ')}."
    if "timeseries" in parts:
        return f"Batch timeseries [{scene or stem}]: FPS, frame time, and logical counters."
    if rel.name == "summary_boxplots.png":
        return "Batch summary boxplots across scenes."
    if rel.name == "performance_distributions.png":
        return "Batch performance distributions across scenes."
    if "streaming_by_visibility" in parts:
        return (
            f"Peak block requests per batch vs. batch\\_index [{scene or stem}]."
        )
    if "occlusion_by_visibility" in parts:
        if (scene or stem).lower() == "g500m":
            return (
                f"Occlusion efficiency ratio vs batch_index for top post-occlusion block levels "
                f"[{scene or stem}] (10 groups)."
            )
        return (
            f"Occlusion efficiency ratio vs batch_index for top visible-block levels "
            f"[{scene or stem}]."
        )
    if rel.name == "streaming_requests_grid.png":
        return "Streaming request decay across PACKAGE scenes."
    if rel.name == "occlusion_ratio_grid.png":
        return "Post-occlusion efficiency across PACKAGE scenes."
    if rel.name == "pool_and_atom_fraction.png":
        return "Pool residency and atom-fraction metrics (coverage, not VRAM)."
    if rel.name == "correlation_grid.png":
        return "Batch metric correlation grid across scenes."
    if rel.name == "preprocess_overview.png":
        return (
            "Preprocess phase timings, throughput, stacked GPU buffer allocations at init "
            "(pipeline, streaming, depth, HiZ, color), and cudaMemGetInfo usage."
        )
    if rel.name == "scaling_blocks_octree.png":
        return "Block count, octree node count, and block-file size vs scale."
    if rel.name == "raster_fraction_vs_scale.png":
        return "Raster stage fraction vs PACKAGE scale."
    if rel.name == "nsight_vs_profiler_frame_ms.png":
        return (
            "Left: mean batch avg_frame_ms from profiler CSV (not Nsight range*_frames.txt). "
            "Right: Nsight Build Active Atom List fraction vs mean activeCount."
        )

    return f"OOC result figure: {rel.as_posix()}."


def _section_key(rel: Path) -> str:
    parent = rel.parent.relative_to(FIG_ROOT).as_posix()
    if parent == ".":
        return rel.stem
    return parent


def _section_title(key: str) -> str:
    if key in SECTION_TITLES:
        return SECTION_TITLES[key]
    return key.replace("/", " — ").replace("_", " ").title()


def _sort_key(rel: Path) -> tuple:
    scene_order = ["g10m", "g50m", "g100m", "g500m", "8wql"]
    stem = rel.stem
    scene_idx = next((i for i, s in enumerate(scene_order) if s in stem), 99)
    ring_m = re.search(r"ring(\d+)", stem)
    ring_idx = int(ring_m.group(1)) if ring_m else 0
    return (_section_key(rel), scene_idx, ring_idx, rel.name)


def generate() -> int:
    pngs = sorted(FIG_ROOT.rglob("*.png"), key=_sort_key)
    if not pngs:
        raise SystemExit(f"No PNG files found under {FIG_ROOT}")

    grouped: dict[str, list[Path]] = defaultdict(list)
    for p in pngs:
        grouped[_section_key(p)].append(p)

    lines = [
        "% Auto-generated by benchmark/nsight_analysis/_generate_ooc_appendix_tex.py",
        "% Do not edit manually; regenerate after re-exporting OOC figures.",
        "",
        r"\subsection{Out-of-Core Results Figures}\label{app:ooc_results_figures}",
        "",
        r"Complete figure set for Section~\ref{sec:ooc_results}. "
        r"Chapter~3 presents representative plots for selected scenes; "
        r"the remaining per-scene and per-metric breakdowns are listed here.",
        "",
    ]

    section_order = sorted(grouped.keys(), key=lambda k: (_sort_key(grouped[k][0])[0], k))
    for sec in section_order:
        files = grouped[sec]
        lines.append(rf"\subsubsection{{{_section_title(sec)}}}")
        lines.append("")
        for rel in files:
            posix = rel.relative_to(
                Path(r"D:\Users\Escritorio\U\Ot2026-2doSemM\tesis-magister-latex")
            ).as_posix()
            label = "fig:app_ooc_" + _slug(rel.relative_to(FIG_ROOT).as_posix())
            cap = _caption(rel.relative_to(FIG_ROOT))
            lines.extend([
                r"\begin{figure}[H]",
                r"    \centering",
                rf"    \includegraphics[width=1\textwidth]{{{posix}}}",
                rf"    \caption{{{cap}}}",
                rf"    \label{{{label}}}",
                r"\end{figure}",
                "",
            ])

    OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {len(pngs)} figures in {len(grouped)} sections to {OUTPUT}")
    return len(pngs)


if __name__ == "__main__":
    generate()
