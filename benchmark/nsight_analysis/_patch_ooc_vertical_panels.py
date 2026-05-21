"""Patch ooc_package_analysis.ipynb: vertical thesis panels + plot helpers."""
from __future__ import annotations

import json
from pathlib import Path

NB = Path(__file__).resolve().parent / "ooc_package_analysis.ipynb"

HELPERS_INSERT = '''
THESIS_PANEL_DPI = 200
PANEL_WIDTH = 14
PANEL_ROW_H = 4.5
PANEL_STALL_ROW_H = 3.5
PANEL_METRIC_ROW_H = 4.0
PANEL_CORR_ROW_H = 5.0


def save_panel_fig(fig, rel_path: str):
    """Save thesis panel at higher DPI."""
    if not SAVE_FIGURES:
        return None
    path = FIG_ROOT / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=THESIS_PANEL_DPI, bbox_inches="tight")
    return path


def plot_metric_by_ring_on_ax(
    ax, df, metric_col, title, ylabel,
    threshold=None, threshold_label=None, palette="viridis",
):
    ordered = analysis_stage_order(df["pipeline_range"].values)
    sub = df[df["pipeline_range"].isin(ordered)].copy()
    sub["pipeline_range"] = pd.Categorical(
        sub["pipeline_range"], categories=ordered, ordered=True
    )
    sns.barplot(
        data=sub, x="pipeline_range", y=metric_col, hue="ring",
        palette=palette, ax=ax, errorbar="sd",
    )
    if threshold is not None:
        ax.axhline(
            threshold, ls="--", color="red", lw=1.5,
            label=threshold_label or str(threshold),
        )
    ax.set_title(title, fontsize=11)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Pipeline stage")
    ax.tick_params(axis="x", rotation=45)
    leg = ax.legend(title="Ring", fontsize=8, title_fontsize=9)
    if leg is not None:
        leg._legend_box.set_alpha(0.9)


'''

PLOT_METRIC_CELL = r'''def plot_metric_by_ring(df, metric_col, title, ylabel, rel_path,
                        threshold=None, threshold_label=None,
                        palette='viridis', figsize=(16, 7)):
    fig, ax = plt.subplots(figsize=figsize)
    plot_metric_by_ring_on_ax(
        ax, df, metric_col, title, ylabel,
        threshold=threshold, threshold_label=threshold_label, palette=palette,
    )
    plt.tight_layout()
    show_and_save(fig, rel_path)


for scene_key, df in dfs_ring.items():
    cfg = SCENES[scene_key]
    plot_metric_by_ring(df, 'gr_active_pct', f"{cfg['label']} — GR Cycles Active [%]", 'GR Active %',
                        f"nsight/gr_active/{scene_key}.png", threshold=80, threshold_label='80% (high utilization)')
    plot_metric_by_ring(df, 'gr_idle_pct', f"{cfg['label']} — GPU Idle [%]", 'GR Idle %',
                        f"nsight/gr_idle/{scene_key}.png", threshold=10, threshold_label='10% (low idle)')
'''

THESIS_PANELS_CELL = r'''## 10b. Thesis panels — vertical stacks for LaTeX (full-width figures)

focus_ranges = ['Sphere Raster OOC', 'Octree BFS Frustum Culling',
                'Occlusion Culling', 'Build Active Atom List']
focus_metrics = ['gr_active_pct', 'gr_idle_pct', 'l1tex_hit_pct',
                 'pcie_throughput', 'warp_occ_pct', 'sm_issue_active']
stall_cols = ['stall_short_scoreboard', 'stall_drain', 'stall_wait']


def _export_gr_active_idle_panel(scene_key='g500m'):
    df = dfs_ring[scene_key]
    cfg = SCENES[scene_key]
    n = 2
    fig, axes = plt.subplots(n, 1, figsize=(PANEL_WIDTH, PANEL_ROW_H * n), squeeze=False)
    plot_metric_by_ring_on_ax(
        axes[0, 0], df, 'gr_active_pct',
        f"{cfg['label']} — GR Cycles Active [%]", 'GR Active %',
        threshold=80, threshold_label='80% (high utilization)',
    )
    plot_metric_by_ring_on_ax(
        axes[1, 0], df, 'gr_idle_pct',
        f"{cfg['label']} — GPU Idle [%]", 'GR Idle %',
        threshold=10, threshold_label='10% (low idle)',
    )
    fig.tight_layout()
    save_panel_fig(fig, f"nsight/panels/gr_active_idle_{scene_key}.png")
    plt.close(fig)


def _export_l1tex_pcie_panel(scene_key='g500m'):
    df = dfs_ring[scene_key]
    cfg = SCENES[scene_key]
    n = 2
    fig, axes = plt.subplots(n, 1, figsize=(PANEL_WIDTH, PANEL_ROW_H * n), squeeze=False)
    plot_metric_by_ring_on_ax(
        axes[0, 0], df, 'l1tex_hit_pct',
        f"{cfg['label']} — L1TEX Hit Rate [%]", 'L1TEX Hit %',
        threshold=70, threshold_label='70%',
    )
    plot_metric_by_ring_on_ax(
        axes[1, 0], df, 'pcie_throughput',
        f"{cfg['label']} — PCIe Throughput [%]", 'PCIe %', palette='magma',
    )
    fig.tight_layout()
    save_panel_fig(fig, f"nsight/panels/l1tex_pcie_{scene_key}.png")
    plt.close(fig)


def _export_stall_rings_panel(scene_key):
    df = dfs_ring[scene_key]
    cfg = SCENES[scene_key]
    rings = sorted(df['ring'].unique())
    stall_data = df.groupby(['ring', 'pipeline_range'])[stall_cols].mean().reset_index()
    ordered = analysis_stage_order(stall_data['pipeline_range'].values)
    n = len(rings)
    fig, axes = plt.subplots(n, 1, figsize=(PANEL_WIDTH, PANEL_STALL_ROW_H * n), squeeze=False)
    for i, ring_id in enumerate(rings):
        ax = axes[i, 0]
        sub = stall_data[(stall_data['ring'] == ring_id) & stall_data['pipeline_range'].isin(ordered)].copy()
        sub['pipeline_range'] = pd.Categorical(sub['pipeline_range'], categories=ordered, ordered=True)
        sub = sub.sort_values('pipeline_range').set_index('pipeline_range')[stall_cols]
        if sub.dropna(how='all').empty:
            ax.set_visible(False)
            continue
        sub.plot(kind='bar', stacked=True, ax=ax, color=['#e74c3c', '#3498db', '#2ecc71'], legend=False)
        ax.set_title(f"Ring {ring_id}", fontsize=11)
        ax.set_ylabel('Stall %')
        ax.set_xlabel('Pipeline stage')
        ax.tick_params(axis='x', rotation=45)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if not handles:
        from matplotlib.patches import Patch
        handles = [Patch(facecolor=c) for c in ['#e74c3c', '#3498db', '#2ecc71']]
        labels = ['Short Scoreboard', 'Drain', 'Wait']
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.98, 0.98), fontsize=9)
    fig.suptitle(f"{cfg['label']} — Stall breakdown by ring", fontsize=12, y=1.002)
    fig.tight_layout()
    save_panel_fig(fig, f"nsight/panels/stall_rings_{scene_key}.png")
    plt.close(fig)


def _export_metrics_vs_ring_panel(scene_key):
    df = dfs[scene_key]
    cfg = SCENES[scene_key]
    sub = df[df['pipeline_range'].isin(focus_ranges)].copy()
    metrics = [m for m in focus_metrics if not sub[m].dropna().empty]
    if not metrics:
        return
    n = len(metrics)
    fig, axes = plt.subplots(n, 1, figsize=(PANEL_WIDTH, PANEL_METRIC_ROW_H * n), squeeze=False)
    rings = sorted(df['ring'].unique())
    for i, met in enumerate(metrics):
        ax = axes[i, 0]
        for rng in focus_ranges:
            rng_data = sub[sub['pipeline_range'] == rng].groupby('ring')[met].agg(['mean', 'std']).reset_index()
            if rng_data['mean'].dropna().empty:
                continue
            ax.errorbar(rng_data['ring'], rng_data['mean'], yerr=rng_data['std'].fillna(0),
                        marker='o', capsize=4, label=rng)
        ax.set_title(f"{met}", fontsize=11)
        ax.set_xlabel('Ring (1=near, 5=far)')
        ax.set_ylabel(met)
        ax.legend(fontsize=7, loc='best')
        ax.set_xticks(rings)
    fig.suptitle(f"{cfg['label']} — Metrics vs camera ring", fontsize=12, y=1.002)
    fig.tight_layout()
    save_panel_fig(fig, f"nsight/panels/metrics_vs_ring_{scene_key}.png")
    plt.close(fig)


def _export_correlation_panel(scene_key='g10m'):
    df = dfs[scene_key]
    cfg = SCENES[scene_key]
    ranges = [r for r in focus_ranges if len(df[df['pipeline_range'] == r]) > 0]
    n = len(ranges)
    fig, axes = plt.subplots(n, 1, figsize=(PANEL_WIDTH, PANEL_CORR_ROW_H * n), squeeze=False)
    for i, rng in enumerate(ranges):
        sub = df[df['pipeline_range'] == rng]
        cols = [c for c in focus_metrics if not sub[c].dropna().empty]
        if len(cols) < 2:
            axes[i, 0].set_visible(False)
            continue
        corr = sub[cols].corr()
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1,
                    ax=axes[i, 0], cbar_kws={'shrink': 0.8})
        axes[i, 0].set_title(f"{rng}", fontsize=11)
    fig.suptitle(f"{cfg['label']} — Metric correlation by stage", fontsize=12, y=1.002)
    fig.tight_layout()
    save_panel_fig(fig, f"nsight/panels/correlation_{scene_key}.png")
    plt.close(fig)


if 'dfs_ring' in dir() and 'g500m' in dfs_ring:
    _export_gr_active_idle_panel('g500m')
    _export_l1tex_pcie_panel('g500m')
if 'dfs_ring' in dir():
    for sk in ('g100m', '8wql'):
        if sk in dfs_ring:
            _export_stall_rings_panel(sk)
if 'dfs' in dir():
    for sk in ('g10m', 'g500m'):
        if sk in dfs:
            _export_metrics_vs_ring_panel(sk)
    if 'g10m' in dfs:
        _export_correlation_panel('g10m')

print('Thesis panels exported under', FIG_ROOT / 'nsight' / 'panels')
'''

OLD_PLOT_START = "def plot_metric_by_ring(df, metric_col, title, ylabel, rel_path,"


def patch():
    nb = json.loads(NB.read_text(encoding="utf-8"))
    changed = []

    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell.get("source", []))

        if "def show_and_save(fig, rel_path: str):" in src and "THESIS_PANEL_DPI" not in src:
            src = src.replace(
                "def show_and_save(fig, rel_path: str):\n"
                "    save_fig(fig, rel_path)\n"
                "    plt.show()\n\n\n",
                "def show_and_save(fig, rel_path: str):\n"
                "    save_fig(fig, rel_path)\n"
                "    plt.show()\n\n"
                + HELPERS_INSERT,
            )
            cell["source"] = [line + "\n" for line in src.splitlines()]
            if src and not src.endswith("\n"):
                cell["source"][-1] = cell["source"][-1].rstrip("\n") + "\n"
            changed.append(f"helpers@{i}")

        if src.strip().startswith(OLD_PLOT_START) and "plot_metric_by_ring_on_ax" not in src:
            cell["source"] = [line + "\n" for line in PLOT_METRIC_CELL.splitlines()]
            if PLOT_METRIC_CELL and not PLOT_METRIC_CELL.endswith("\n"):
                pass
            changed.append(f"plot_metric@{i}")

    # Insert thesis panels after correlation cell (## 10.)
    insert_idx = None
    for i, cell in enumerate(nb["cells"]):
        src = "".join(cell.get("source", []))
        if "## 10. Metric correlation" in src or (
            cell["cell_type"] == "code"
            and "nsight/correlation/" in src
            and "_export_correlation_panel" not in src
        ):
            insert_idx = i + 1

    if insert_idx is None:
        for i, cell in enumerate(nb["cells"]):
            if cell["cell_type"] == "code" and "nsight/correlation/" in "".join(cell.get("source", [])):
                insert_idx = i + 1
                break

    has_panels = any("## 10b. Thesis panels" in "".join(c.get("source", [])) for c in nb["cells"])
    if not has_panels and insert_idx is not None:
        md, code = THESIS_PANELS_CELL.split("\n\n", 1)
        nb["cells"].insert(
            insert_idx,
            {"cell_type": "markdown", "metadata": {}, "source": [md + "\n"]},
        )
        nb["cells"].insert(
            insert_idx + 1,
            {
                "cell_type": "code",
                "metadata": {},
                "outputs": [],
                "source": [line + "\n" for line in code.splitlines()],
            },
        )
        changed.append(f"panels@{insert_idx}")

    NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("Patched:", changed)


if __name__ == "__main__":
    patch()
