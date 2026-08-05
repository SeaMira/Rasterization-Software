"""Patch ooc_package_analysis.ipynb for thesis figure export and analysis-stage filter."""
from __future__ import annotations

import json
from pathlib import Path

NB_PATH = Path(__file__).resolve().parent / "ooc_package_analysis.ipynb"

CONFIG_INSERT = r'''
FIG_ROOT = Path(r"D:\Users\Escritorio\U\Ot2026-2doSemM\tesis-magister-latex\img\chapter3\results\ooc")
SAVE_FIGURES = True
FIG_DPI = 150

OOC_PIPELINE_EXCLUDED = {
    "Screen Clear", "Blit Framebuffer", "Swap Window",
    "cub::DeviceRadixSort",
}
OOC_PIPELINE_ANALYSIS = [r for r in OOC_PIPELINE_RANGES if r not in OOC_PIPELINE_EXCLUDED]


def _slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()


def save_fig(fig, rel_path: str):
    if not SAVE_FIGURES:
        return None
    path = FIG_ROOT / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    return path


def show_and_save(fig, rel_path: str):
    save_fig(fig, rel_path)
    plt.show()


def analysis_stage_order(values):
    present = set(values)
    return [r for r in OOC_PIPELINE_ANALYSIS if r in present]


def filter_analysis_df(df):
    return df[df["pipeline_range"].isin(OOC_PIPELINE_ANALYSIS)].copy()


def renormalize_rel_duration(df, group_cols):
    out = df.copy()
    totals = out.groupby(group_cols, observed=True)["rel_frame_duration"].transform("sum")
    out["rel_frame_duration"] = out["rel_frame_duration"] / totals.replace(0, np.nan)
    return out


def prepare_df_for_analysis(df):
    return renormalize_rel_duration(filter_analysis_df(df), ["sample", "ring", "position"])


def prepare_ring_df_for_analysis(df_ring):
    return renormalize_rel_duration(filter_analysis_df(df_ring), ["ring"])
'''

CELL_SOURCES: dict[int, str] = {}


def _load_cells():
    global CELL_SOURCES
    # populated below in exec of strings - use read from separate block file
    pass


# --- cell sources (index -> source) ---
CELL_SOURCES = {
10: """### Notes on CU_OOC metrics

- **`ring`** = camera distance (`range1`…`range5` in the filename); **`position`** = YAML replica (1–3).
- **`RelativeFrameDuration`** is the fraction of frame time per pipeline stage.
- **Analysis stages** exclude `Screen Clear`, `Blit Framebuffer`, `Swap Window`, and `cub::DeviceRadixSort` (eight core stages); remaining fractions are **renormalized** to sum to 1 per capture.
- **`rangeN_frames.txt`** files (under each scene's YAML folder) hold Nsight Graphics frame times in ms, split by `------` into three positional replicas per ring.
- **`HiZ Downsample`** often has no GPU counters (NaN); duration share is still valid.
- Bar charts in §6–8 use **`dfs_ring`**. §9 keeps **`dfs`** for error bars across replicas.
- With **`SAVE_FIGURES = True`**, plots are saved under `img/chapter3/results/ooc/` in the thesis tree.
""",
14: r'''def build_metrics_dataframe(scene_data, key_metrics=KEY_METRIC_IDS):
    rows = []
    for sample_label, ranges in scene_data.items():
        parts = sample_label.replace('range', '').split('_')
        ring, pos = int(parts[0]), int(parts[1])
        for rname, mdict in ranges.items():
            row = {
                'sample': sample_label, 'ring': ring, 'position': pos,
                'pipeline_range': rname,
                'rel_frame_duration': mdict.get('RelativeFrameDuration', 0.0),
            }
            for friendly, mid in key_metrics.items():
                row[friendly] = mdict.get(mid, np.nan)
            rows.append(row)
    return pd.DataFrame(rows)


dfs = {}
dfs_raw = {}
for scene_key, scene_data in all_scene_data.items():
    df_raw = build_metrics_dataframe(scene_data)
    dfs_raw[scene_key] = df_raw
    df = prepare_df_for_analysis(df_raw)
    dfs[scene_key] = df
    print(f"\n=== {SCENES[scene_key]['label']} ===")
    print(f"Rows: {len(df)}, unique samples: {df['sample'].nunique()}, ranges: {df['pipeline_range'].nunique()}")
    display(df.head(16))

dfs_ring = {sk: prepare_ring_df_for_analysis(aggregate_metrics_by_ring(dfs_raw[sk])) for sk in dfs_raw}
print("Per-ring aggregates (mean over 3 YAML replicas per camera range):")
for sk, dr in dfs_ring.items():
    print(f"  {sk}: {len(dr)} rows")
''',
16: r'''def _rel_col(stage):
    return "rel_" + stage.replace(" ", "_").replace("+", "plus")


def nsight_scene_summary(dfs_ring_map):
    rows = []
    for sk, dr in dfs_ring_map.items():
        row = {"scene_id": sk}
        for stage in OOC_NSIGHT_FOCUS_STAGES:
            row[_rel_col(stage)] = dr.loc[dr["pipeline_range"] == stage, "rel_frame_duration"].mean()
        row["rel_ooc_core"] = dr.loc[dr["pipeline_range"].isin(OOC_NSIGHT_FOCUS_STAGES), "rel_frame_duration"].sum()
        rows.append(row)
    return pd.DataFrame(rows)


if df_batch_all is not None and not df_batch_all.empty and "dfs_ring" in dir():
    prof = (
        df_batch_all.groupby("scene_id", observed=True)
        .agg(
            frame_ms_mean=("avg_frame_ms", "mean"), fps_mean=("avg_fps", "mean"),
            visible_mean=("avg_numVisible", "mean"), filtered_mean=("avg_numFiltered", "mean"),
            requests_mean=("avg_numRequests", "mean"), active_mean=("avg_activeCount", "mean"),
        ).reset_index()
    )
    joined = nsight_scene_summary(dfs_ring).merge(prof, on="scene_id")
    display(joined.round(4))
    pkg = joined[joined["scene_id"].str.match(r"^g\d+m$", na=False)].copy()
    if len(pkg) >= 2:
        pkg["sphere_M"] = pkg["scene_id"].str.extract(r"^g(\d+)m$")[0].astype(float)
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].scatter(pkg["sphere_M"], pkg["frame_ms_mean"], s=90)
        ax[0].set_xscale("log")
        ax[0].set_xlabel("Spheres (millions)")
        ax[0].set_ylabel("Mean frame time (ms)")
        ax[1].scatter(pkg[_rel_col("Build Active Atom List")], pkg["active_mean"], s=90)
        ax[1].set_xlabel("Nsight rel. Build Active Atom List")
        ax[1].set_ylabel("Mean active atoms / batch")
        plt.tight_layout()
        show_and_save(fig, "package/nsight_vs_profiler_frame_ms.png")
''',
18: r'''def infer_sphere_millions(scene_id: str):
    m = re.match(r'^g(\d+)m$', scene_id, re.IGNORECASE)
    return int(m.group(1)) if m else None


pkg_scenes = [(k, infer_sphere_millions(k)) for k in dfs if infer_sphere_millions(k) is not None]
if len(pkg_scenes) >= 2:
    target_rng = 'Sphere Raster OOC'
    all_rings = set()
    for sk, _ in pkg_scenes:
        all_rings.update(dfs[sk]['ring'].unique())
    fig, ax = plt.subplots(figsize=(11, 5))
    for ring_id in sorted(all_rings):
        xs, ys = [], []
        for sk, millions in sorted(pkg_scenes, key=lambda t: t[1]):
            sub = dfs[sk][(dfs[sk]['pipeline_range'] == target_rng) & (dfs[sk]['ring'] == ring_id)]
            if sub.empty:
                continue
            xs.append(millions)
            ys.append(sub['rel_frame_duration'].mean())
        if len(xs) >= 2:
            ax.plot(xs, ys, marker='o', label=f'Ring {ring_id}')
    if ax.lines:
        ax.set_xscale('log')
        ax.set_xlabel('Spheres (millions, log scale)')
        ax.set_ylabel(f'Mean frame fraction — {target_rng}')
        ax.set_title('PACKAGE scales: relative OOC raster cost vs scene size')
        ax.legend(title='Distance (ring)')
        plt.tight_layout()
        show_and_save(fig, "package/raster_fraction_vs_scale.png")
    else:
        print('Not enough points for', target_rng, 'in g*m scenes.')
else:
    print('Fewer than 2 g*m folders with data; skipping comparative overview.')
''',
20: r'''for scene_key, df in dfs.items():
    cfg = SCENES[scene_key]
    pivot = df.pivot_table(index='pipeline_range', columns='ring', values='rel_frame_duration', aggfunc='mean')
    ordered = analysis_stage_order(pivot.index)
    pivot = pivot.loc[ordered]

    fig, ax = plt.subplots(figsize=(10, 6))
    pivot.T.plot(kind='bar', stacked=True, ax=ax, colormap='tab20')
    ax.set_title(f"{cfg['label']} — Frame fraction by stage")
    ax.set_xlabel('Ring (distance)')
    ax.set_ylabel('Frame fraction')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    ax.set_xticklabels([f'Ring {c}' for c in pivot.columns], rotation=0)
    plt.tight_layout()
    show_and_save(fig, f"nsight/stage_fraction/{scene_key}.png")

    fig, ax = plt.subplots(figsize=(12, max(6, len(ordered) * 0.45)))
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Frame fraction'})
    ax.set_title(f"{cfg['label']} — Relative duration heatmap")
    ax.set_ylabel('Pipeline stage')
    ax.set_xlabel('Ring')
    plt.tight_layout()
    show_and_save(fig, f"nsight/stage_heatmap/{scene_key}.png")
''',
22: r'''for scene_key, df in dfs.items():
    cfg = SCENES[scene_key]
    grp = df.groupby(['ring', 'pipeline_range'])['rel_frame_duration'].mean().reset_index()
    idx = grp.groupby('ring')['rel_frame_duration'].idxmax()
    bottleneck = grp.loc[idx][['ring', 'pipeline_range', 'rel_frame_duration']]
    bottleneck.columns = ['Ring', 'Bottleneck stage', 'Frame fraction']
    print(f"\n=== {cfg['label']} — Bottleneck by distance ===")
    display(bottleneck.reset_index(drop=True))
''',
24: r'''def plot_metric_by_ring(df, metric_col, title, ylabel, rel_path,
                        threshold=None, threshold_label=None,
                        palette='viridis', figsize=(16, 7)):
    fig, ax = plt.subplots(figsize=figsize)
    ordered = analysis_stage_order(df['pipeline_range'].values)
    sub = df[df['pipeline_range'].isin(ordered)].copy()
    sub['pipeline_range'] = pd.Categorical(sub['pipeline_range'], categories=ordered, ordered=True)
    sns.barplot(data=sub, x='pipeline_range', y=metric_col, hue='ring', palette=palette, ax=ax, errorbar='sd')
    if threshold is not None:
        ax.axhline(threshold, ls='--', color='red', lw=1.5, label=threshold_label or str(threshold))
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Pipeline stage')
    ax.tick_params(axis='x', rotation=45)
    ax.legend(title='Ring')
    plt.tight_layout()
    show_and_save(fig, rel_path)


for scene_key, df in dfs_ring.items():
    cfg = SCENES[scene_key]
    plot_metric_by_ring(df, 'gr_active_pct', f"{cfg['label']} — GR Cycles Active [%]", 'GR Active %',
                        f"nsight/gr_active/{scene_key}.png", threshold=80, threshold_label='80% (high utilization)')
    plot_metric_by_ring(df, 'gr_idle_pct', f"{cfg['label']} — GPU Idle [%]", 'GR Idle %',
                        f"nsight/gr_idle/{scene_key}.png", threshold=10, threshold_label='10% (low idle)')
''',
26: r'''for scene_key, df in dfs_ring.items():
    cfg = SCENES[scene_key]
    plot_metric_by_ring(df, 'l1tex_hit_pct', f"{cfg['label']} — L1TEX Hit Rate [%]", 'L1TEX Hit %',
                        f"nsight/l1tex_hit/{scene_key}.png", threshold=70, threshold_label='70%')
    plot_metric_by_ring(df, 'pcie_throughput', f"{cfg['label']} — PCIe Throughput [%]", 'PCIe %',
                        f"nsight/pcie/{scene_key}.png", palette='magma')
''',
28: r'''stall_cols = ['stall_short_scoreboard', 'stall_drain', 'stall_wait']

for scene_key, df in dfs_ring.items():
    cfg = SCENES[scene_key]
    plot_metric_by_ring(df, 'warp_occ_pct', f"{cfg['label']} — SM Warp Occupancy [%]", 'Warp Occupancy %',
                        f"nsight/warp_occupancy/{scene_key}.png", palette='crest')
    stall_data = df.groupby(['ring', 'pipeline_range'])[stall_cols].mean().reset_index()
    ordered = analysis_stage_order(stall_data['pipeline_range'].values)
    for ring_id in sorted(df['ring'].unique()):
        sub = stall_data[(stall_data['ring'] == ring_id) & stall_data['pipeline_range'].isin(ordered)].copy()
        sub['pipeline_range'] = pd.Categorical(sub['pipeline_range'], categories=ordered, ordered=True)
        sub = sub.sort_values('pipeline_range').set_index('pipeline_range')[stall_cols]
        if sub.dropna(how='all').empty:
            continue
        fig, ax = plt.subplots(figsize=(14, 5))
        sub.plot(kind='bar', stacked=True, ax=ax, color=['#e74c3c', '#3498db', '#2ecc71'])
        ax.set_title(f"{cfg['label']} — Stall Breakdown, Ring {ring_id}")
        ax.set_ylabel('Stall %')
        ax.set_xlabel('Pipeline stage')
        ax.tick_params(axis='x', rotation=45)
        ax.legend(['Short Scoreboard', 'Drain', 'Wait'])
        plt.tight_layout()
        show_and_save(fig, f"nsight/stall_breakdown/{scene_key}_ring{ring_id}.png")
''',
30: r'''focus_ranges = ['Sphere Raster OOC', 'Octree BFS Frustum Culling',
                'Occlusion Culling', 'Build Active Atom List']
focus_metrics = ['gr_active_pct', 'gr_idle_pct', 'l1tex_hit_pct',
                 'pcie_throughput', 'warp_occ_pct', 'sm_issue_active']

for scene_key, df in dfs.items():
    cfg = SCENES[scene_key]
    sub = df[df['pipeline_range'].isin(focus_ranges)].copy()
    if sub.empty:
        continue
    for met in focus_metrics:
        if sub[met].dropna().empty:
            continue
        fig, ax = plt.subplots(figsize=(12, 5))
        for rng in focus_ranges:
            rng_data = sub[sub['pipeline_range'] == rng].groupby('ring')[met].agg(['mean', 'std']).reset_index()
            if rng_data['mean'].dropna().empty:
                continue
            ax.errorbar(rng_data['ring'], rng_data['mean'], yerr=rng_data['std'],
                        marker='o', capsize=4, label=rng)
        ax.set_title(f"{cfg['label']} — {met} vs distance")
        ax.set_xlabel('Ring (1=near, 5=far)')
        ax.set_ylabel(met)
        ax.legend(fontsize=8)
        ax.set_xticks(sorted(df['ring'].unique()))
        plt.tight_layout()
        show_and_save(fig, f"nsight/metrics_vs_ring/{scene_key}_{_slug(met)}.png")
''',
32: r'''for scene_key, df in dfs.items():
    cfg = SCENES[scene_key]
    for rng in focus_ranges:
        sub = df[df['pipeline_range'] == rng]
        cols = [c for c in focus_metrics if not sub[c].dropna().empty]
        if len(cols) < 2:
            continue
        corr = sub[cols].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, ax=ax)
        ax.set_title(f"{cfg['label']} — Metric correlation: {rng}")
        plt.tight_layout()
        show_and_save(fig, f"nsight/correlation/{scene_key}_{_slug(rng)}.png")
''',
34: r'''if df_prep is None and PREPROCESS_CSV.is_file():
    df_prep = load_preprocess_table()
    df_prep['total_preprocess_ms'] = df_prep[['ms_morton_pipeline','ms_blocks_and_file','ms_octree_build']].sum(axis=1)
    df_prep['atoms_per_ms'] = df_prep['sphere_count'] / df_prep['total_preprocess_ms']
    df_prep['vram_used_MB'] = df_prep['vram_sum_estimated_bytes'] / (1024**2)
    df_prep['block_file_MB'] = df_prep['block_file_bytes'] / (1024**2)
    df_prep['host_MB'] = df_prep['host_structures_bytes'] / (1024**2)
    df_prep['vram_pct_used'] = 100.0 * (1.0 - df_prep['cuda_mem_free_bytes'] / df_prep['cuda_mem_total_bytes'])

    prep_grp = df_prep.groupby(['scene_type','sphere_count']).agg(
        n_runs=('total_preprocess_ms','count'),
        avg_ms=('total_preprocess_ms','mean'), std_ms=('total_preprocess_ms','std'),
        min_ms=('total_preprocess_ms','min'), max_ms=('total_preprocess_ms','max'),
        avg_morton=('ms_morton_pipeline','mean'), avg_blocks=('ms_blocks_and_file','mean'),
        avg_octree=('ms_octree_build','mean'), avg_throughput=('atoms_per_ms','mean'),
        avg_vram_MB=('vram_used_MB','mean'), avg_vram_pct=('vram_pct_used','mean'),
        avg_block_file_MB=('block_file_MB','mean'), avg_host_MB=('host_MB','mean'),
        block_count=('block_count','first'), octree_nodes=('octree_node_count','first'),
    ).reset_index().sort_values('sphere_count')

    print('=== Preprocess: summary by scene/size ===')
    display(prep_grp)

    pkg = prep_grp[prep_grp['scene_type'].str.contains('PACKAGE')].copy()
    if len(pkg) >= 2:
        fig, axes = plt.subplots(2, 3, figsize=(20, 10))
        ax = axes[0,0]
        ax.errorbar(pkg['sphere_count']/1e6, pkg['avg_ms'], yerr=pkg['std_ms'].fillna(0), marker='o', capsize=4)
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel('Spheres (millions)'); ax.set_ylabel('Total time (ms)')
        ax.set_title('Preprocess time vs entity count')
        ax = axes[0,1]
        for col, lbl in [('avg_morton','Morton+sort'), ('avg_blocks','Blocks+file'), ('avg_octree','Octree')]:
            ax.plot(pkg['sphere_count']/1e6, pkg[col], marker='o', label=lbl)
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel('Spheres (millions)'); ax.set_ylabel('ms')
        ax.set_title('Preprocess phase breakdown'); ax.legend()
        ax = axes[0,2]
        ax.plot(pkg['sphere_count']/1e6, pkg['avg_throughput'], marker='s', color='green')
        ax.set_xscale('log')
        ax.set_xlabel('Spheres (millions)'); ax.set_ylabel('Atoms/ms')
        ax.set_title('Preprocess throughput')
        ax = axes[1,0]
        ax.bar(pkg['sphere_count'].astype(str), pkg['avg_vram_MB'], color='steelblue')
        ax.set_xlabel('Spheres'); ax.set_ylabel('Estimated VRAM (MB)')
        ax.set_title('Total estimated VRAM'); ax.tick_params(axis='x', rotation=45)
        ax = axes[1,1]
        ax.bar(pkg['sphere_count'].astype(str), pkg['avg_vram_pct'], color='coral')
        ax.axhline(100, ls='--', color='red', lw=1)
        ax.set_xlabel('Spheres'); ax.set_ylabel('% VRAM used')
        ax.set_title('% GPU VRAM used'); ax.tick_params(axis='x', rotation=45)
        ax = axes[1,2]
        ax.plot(pkg['sphere_count']/1e6, pkg['avg_block_file_MB'], marker='o', label='Block file')
        ax.plot(pkg['sphere_count']/1e6, pkg['avg_host_MB'], marker='^', label='Host structs')
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel('Spheres (millions)'); ax.set_ylabel('MB')
        ax.set_title('On-disk and host data'); ax.legend()
        plt.tight_layout()
        show_and_save(fig, "preprocess/preprocess_overview.png")
else:
    print(f'\u26a0 Preprocess CSV not found at {PREPROCESS_CSV}')
''',
36: r'''if df_batch_all is None:
    df_batch_all = load_batch_stats_table()

if df_batch_all is None or df_batch_all.empty:
    print('No batch stats in', _profiler_csv_dir())
else:
    print('\n=== Summary by scene ===')
    summary = df_batch_all.groupby('scene_id').agg(
        sphere_count=('sphere_count','first'), total_blocks=('total_blocks','first'),
        pool_slots=('pool_slots','first'), n_batches=('avg_fps','count'),
        fps_mean=('avg_fps','mean'), fps_std=('avg_fps','std'),
        frame_ms_mean=('avg_frame_ms','mean'), visible_mean=('avg_numVisible','mean'),
        filtered_mean=('avg_numFiltered','mean'), requests_mean=('avg_numRequests','mean'),
        active_mean=('avg_activeCount','mean'),
    ).reset_index()
    display(summary)

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    order = sorted(df_batch_all['scene_id'].unique(), key=lambda x: _scene_sort_key(x))
    sns.boxplot(data=df_batch_all, x='scene_id', y='avg_fps', order=order, ax=axes[0,0], palette='viridis')
    axes[0,0].set_title('FPS distribution by scene'); axes[0,0].set_xlabel('Scene'); axes[0,0].set_ylabel('FPS')
    axes[0,0].tick_params(axis='x', rotation=30)
    sns.boxplot(data=df_batch_all, x='scene_id', y='avg_frame_ms', order=order, ax=axes[0,1], palette='magma')
    axes[0,1].set_title('Frame time (ms) by scene'); axes[0,1].set_xlabel('Scene'); axes[0,1].set_ylabel('ms')
    axes[0,1].tick_params(axis='x', rotation=30)
    sns.boxplot(data=df_batch_all, x='scene_id', y='avg_activeCount', order=order, ax=axes[1,0], palette='crest')
    axes[1,0].set_title('Rasterized atoms by scene'); axes[1,0].set_xlabel('Scene'); axes[1,0].set_ylabel('Atoms')
    axes[1,0].tick_params(axis='x', rotation=30)
    sns.boxplot(data=df_batch_all, x='scene_id', y='avg_numRequests', order=order, ax=axes[1,1], palette='flare')
    axes[1,1].set_title('Streaming-requested blocks by scene'); axes[1,1].set_xlabel('Scene')
    axes[1,1].set_ylabel('Requests/batch'); axes[1,1].tick_params(axis='x', rotation=30)
    plt.tight_layout()
    show_and_save(fig, "batch/summary_boxplots.png")

    for bk in order:
        sub = df_batch_all[df_batch_all['scene_id'] == bk].copy()
        lbl = sub['label'].iloc[0]
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle(f'{lbl} ({bk})', fontsize=14)
        axes[0,0].plot(sub['batch_index'], sub['avg_fps'], marker='.', ms=3)
        axes[0,0].fill_between(sub['batch_index'], sub['min_fps'], sub['max_fps'], alpha=0.15)
        axes[0,0].set_title('FPS'); axes[0,0].set_xlabel('Batch'); axes[0,0].set_ylabel('FPS')
        axes[0,1].plot(sub['batch_index'], sub['avg_numVisible'], label='Visible (frustum)', ms=3, marker='.')
        axes[0,1].plot(sub['batch_index'], sub['avg_numFiltered'], label='Post-occlusion', ms=3, marker='.')
        axes[0,1].set_title('Blocks: visible vs post-occlusion'); axes[0,1].legend()
        axes[1,0].plot(sub['batch_index'], sub['avg_activeCount'], color='green', ms=3, marker='.')
        axes[1,0].fill_between(sub['batch_index'], sub['min_activeCount'], sub['max_activeCount'], alpha=0.15, color='green')
        axes[1,0].set_title('Rasterized atoms'); axes[1,0].set_ylabel('Atoms')
        axes[1,1].plot(sub['batch_index'], sub['avg_numRequests'], color='orange', ms=3, marker='.')
        axes[1,1].fill_between(sub['batch_index'], sub['min_numRequests'], sub['max_numRequests'], alpha=0.15, color='orange')
        axes[1,1].set_title('Blocks requested for streaming'); axes[1,1].set_ylabel('Requests')
        plt.tight_layout()
        show_and_save(fig, f"batch/timeseries/{bk}.png")

    fig, ax = plt.subplots(figsize=(12, 7))
    for bk in order:
        sub = df_batch_all[df_batch_all['scene_id'] == bk]
        ax.scatter(sub['avg_activeCount'], sub['avg_fps'], s=15, alpha=0.5, label=bk)
    ax.set_xlabel('Mean rasterized atoms'); ax.set_ylabel('Mean FPS')
    ax.set_title('FPS vs rasterized atoms (all scenes)')
    ax.legend(title='Scene')
    plt.tight_layout()
    show_and_save(fig, "batch/fps_vs_active_atoms.png")
''',
38: r'''if BATCH_CSVS and 'df_batch_all' in dir():
    pkg_only = df_batch_all[df_batch_all['scene_type'].str.contains('PACKAGE', na=False)].copy()
    if not pkg_only.empty:
        sc_grp = pkg_only.groupby('scene_id').agg(
            sphere_count=('sphere_count','first'),
            fps_mean=('avg_fps','mean'), fps_std=('avg_fps','std'),
            fps_median=('avg_fps', 'median'),
            fps_p25=('avg_fps', lambda x: x.quantile(0.25)),
            fps_p75=('avg_fps', lambda x: x.quantile(0.75)),
        ).reset_index().sort_values('sphere_count')

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        ax = axes[0]
        ax.errorbar(sc_grp['sphere_count']/1e6, sc_grp['fps_mean'], yerr=sc_grp['fps_std'].fillna(0),
                    marker='o', capsize=5, lw=2)
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel('Spheres (millions)'); ax.set_ylabel('Mean FPS')
        ax.set_title('FPS vs entity count'); ax.grid(True, which='both', alpha=0.3)
        if len(sc_grp) >= 2:
            coeffs = np.polyfit(np.log10(sc_grp['sphere_count']), np.log10(sc_grp['fps_mean']), 1)
            xs = np.logspace(np.log10(sc_grp['sphere_count'].min()), np.log10(sc_grp['sphere_count'].max()), 50)
            ax.plot(xs/1e6, 10**(coeffs[0]*np.log10(xs) + coeffs[1]), '--', color='red',
                    label=f'Trend: slope={coeffs[0]:.2f}')
            ax.legend()
        ax = axes[1]
        y_lo = (sc_grp['fps_median'] - sc_grp['fps_p25']).clip(lower=0)
        y_hi = (sc_grp['fps_p75'] - sc_grp['fps_median']).clip(lower=0)
        ax.errorbar(sc_grp['sphere_count']/1e6, sc_grp['fps_median'], yerr=[y_lo, y_hi],
                    marker='s', capsize=5, lw=2, color='darkorange')
        ax.set_xscale('log')
        ax.set_xlabel('Spheres (millions)'); ax.set_ylabel('Median FPS')
        ax.set_title('FPS with interquartile range (P25–P75)')
        ax.grid(True, which='both', alpha=0.3)
        plt.tight_layout()
        show_and_save(fig, "batch/scaling_fps_loglog.png")
''',
40: r'''if BATCH_CSVS and 'df_batch_all' in dir():
    order = sorted(df_batch_all['scene_id'].unique(), key=lambda x: _scene_sort_key(x))
    n = len(order)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 4*rows), squeeze=False)
    fig.suptitle('Streaming convergence: requests per batch', fontsize=14)
    for idx, bk in enumerate(order):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        sub = df_batch_all[df_batch_all['scene_id'] == bk].sort_values('batch_index')
        ax.semilogy(sub['batch_index'], sub['avg_numRequests'].clip(lower=0.1), marker='.', ms=3)
        ax.set_title(bk); ax.set_xlabel('Batch'); ax.set_ylabel('Requests (log)')
        ax.grid(True, alpha=0.3)
    for idx in range(n, rows*cols):
        r, c = divmod(idx, cols)
        axes[r][c].set_visible(False)
    plt.tight_layout()
    show_and_save(fig, "batch/streaming_requests_grid.png")

    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 4*rows), squeeze=False)
    fig.suptitle('Post-occlusion / visible block ratio', fontsize=14)
    for idx, bk in enumerate(order):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        sub = df_batch_all[df_batch_all['scene_id'] == bk].sort_values('batch_index')
        ratio = sub['avg_numFiltered'] / sub['avg_numVisible'].replace(0, np.nan)
        ax.plot(sub['batch_index'], ratio, marker='.', ms=3, color='teal')
        ax.set_title(bk); ax.set_ylim(0, 1); ax.grid(True, alpha=0.3)
    for idx in range(n, rows*cols):
        r, c = divmod(idx, cols)
        axes[r][c].set_visible(False)
    plt.tight_layout()
    show_and_save(fig, "batch/occlusion_ratio_grid.png")
''',
42: r'''if BATCH_CSVS and 'df_batch_all' in dir():
    order = sorted(df_batch_all['scene_id'].unique(), key=lambda x: _scene_sort_key(x))
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    df_batch_all['pool_util_pct'] = 100.0 * df_batch_all['avg_numVisible'] / df_batch_all['pool_slots'].replace(0, np.nan)
    sns.boxplot(data=df_batch_all, x='scene_id', y='pool_util_pct', order=order, ax=axes[0], palette='coolwarm')
    axes[0].set_title('Pool utilization (visible blocks / slots)')
    axes[0].set_xlabel('scene_id')
    axes[0].tick_params(axis='x', rotation=30)
    df_batch_all['atom_fraction_pct'] = 100.0 * df_batch_all['avg_activeCount'] / df_batch_all['sphere_count'].replace(0, np.nan)
    sns.boxplot(data=df_batch_all, x='scene_id', y='atom_fraction_pct', order=order, ax=axes[1], palette='RdYlGn')
    axes[1].set_title('% rasterized atoms vs total scene spheres')
    axes[1].set_xlabel('scene_id')
    axes[1].tick_params(axis='x', rotation=30)
    plt.tight_layout()
    show_and_save(fig, "batch/pool_and_atom_fraction.png")
''',
44: r'''if BATCH_CSVS and 'df_batch_all' in dir():
    order = sorted(df_batch_all['scene_id'].unique(), key=lambda x: _scene_sort_key(x))
    corr_cols = ['avg_frame_ms','avg_fps','avg_numVisible','avg_numFiltered','avg_numRequests','avg_activeCount']
    n = len(order)
    cols_per_row = min(3, n)
    rows = (n + cols_per_row - 1) // cols_per_row
    fig, axes = plt.subplots(rows, cols_per_row, figsize=(6*cols_per_row, 5*rows), squeeze=False)
    fig.suptitle('Pearson correlations between batch metrics', fontsize=14)
    for idx, bk in enumerate(order):
        r, c = divmod(idx, cols_per_row)
        ax = axes[r][c]
        sub = df_batch_all[df_batch_all['scene_id'] == bk][corr_cols].dropna()
        if len(sub) >= 3:
            corr = sub.corr()
            sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0, vmin=-1, vmax=1, ax=ax, cbar=False,
                        xticklabels=[c.replace('avg_','') for c in corr_cols],
                        yticklabels=[c.replace('avg_','') for c in corr_cols])
        ax.set_title(bk)
    for idx in range(n, rows * cols_per_row):
        r, c = divmod(idx, cols_per_row)
        axes[r][c].set_visible(False)
    plt.tight_layout()
    show_and_save(fig, "batch/correlation_grid.png")
''',
46: r'''if BATCH_CSVS and 'df_batch_all' in dir():
    order = sorted(df_batch_all['scene_id'].unique(), key=lambda x: _scene_sort_key(x))
    n = len(order)
    fig, axes = plt.subplots(2, n, figsize=(5*n, 8), squeeze=False)
    fig.suptitle('Performance distributions per scene', fontsize=14)
    for idx, bk in enumerate(order):
        sub = df_batch_all[df_batch_all['scene_id'] == bk]
        axes[0, idx].hist(sub['avg_fps'], bins=40, alpha=0.7, color='steelblue', edgecolor='white')
        axes[0, idx].set_title(f'{bk} - FPS')
        axes[1, idx].hist(sub['avg_frame_ms'], bins=40, alpha=0.7, color='coral', edgecolor='white')
        axes[1, idx].set_title(f'{bk} - Frame time')
    plt.tight_layout()
    show_and_save(fig, "batch/performance_distributions.png")
''',
48: r'''if PREPROCESS_CSV.exists() and 'prep_grp' in dir():
    pkg = prep_grp[prep_grp['scene_type'].str.contains('PACKAGE')].copy()
    if len(pkg) >= 2:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        ax = axes[0]
        ax.plot(pkg['sphere_count']/1e6, pkg['block_count'], marker='o', color='navy')
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel('Spheres (millions)'); ax.set_ylabel('Blocks')
        ax.set_title('Generated blocks vs entity count'); ax.grid(True, alpha=0.3)
        ax = axes[1]
        ax.plot(pkg['sphere_count']/1e6, pkg['octree_nodes'], marker='s', color='darkred')
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel('Spheres (millions)'); ax.set_ylabel('Octree nodes')
        ax.set_title('Octree nodes vs entity count'); ax.grid(True, alpha=0.3)
        ax = axes[2]
        ax.plot(pkg['sphere_count']/1e6, pkg['avg_block_file_MB'], marker='^', color='darkgreen')
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel('Spheres (millions)'); ax.set_ylabel('Block file (MB)')
        ax.set_title('Block file size'); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        show_and_save(fig, "preprocess/scaling_blocks_octree.png")
''',
50: r'''if len(dfs) >= 2:
    combined = []
    for scene_key, df in dfs.items():
        tmp = df.copy()
        tmp['scene'] = SCENES[scene_key]['label']
        combined.append(tmp)
    df_all = pd.concat(combined, ignore_index=True)
    for met in ['gr_active_pct', 'gr_idle_pct', 'l1tex_hit_pct', 'pcie_throughput']:
        if df_all[met].dropna().empty:
            continue
        fig, ax = plt.subplots(figsize=(14, 6))
        sub = df_all[df_all['pipeline_range'].isin(focus_ranges)]
        sns.boxplot(data=sub, x='pipeline_range', y=met, hue='scene', ax=ax)
        ax.set_title(f'Comparison — {met}')
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        show_and_save(fig, f"cross_scene/{_slug(met)}_comparison.png")
else:
    print('Only one scene loaded; skipping cross-scene comparison.')
''',
52: r'''for scene_key, df in dfs.items():
    cfg = SCENES[scene_key]
    sub = df[df['pipeline_range'].isin(focus_ranges)].copy()
    if sub.empty:
        continue
    fig, ax = plt.subplots(figsize=(12, 7))
    for rng in focus_ranges:
        rng_df = sub[sub['pipeline_range'] == rng]
        if rng_df[['pcie_throughput', 'sm_issue_active']].dropna().empty:
            continue
        ax.scatter(rng_df['pcie_throughput'], rng_df['sm_issue_active'], label=rng, s=60, alpha=0.7)
    ax.axvline(60, ls='--', color='gray', alpha=0.5)
    ax.axhline(60, ls='--', color='gray', alpha=0.5)
    ax.set_xlabel('PCIe Throughput [%]')
    ax.set_ylabel('SM Issue Active [%]')
    ax.set_title(f"{cfg['label']} — Memory-bound vs compute-bound?")
    ax.legend(fontsize=8)
    plt.tight_layout()
    show_and_save(fig, f"nsight/memory_bound/{scene_key}.png")
''',
}


def patch_config_cell(src: str) -> str:
    marker = "OOC_PIPELINE_RANGES = ["
    if "FIG_ROOT = Path" in src:
        return src
    idx = src.find("]\n\nKEY_METRIC_IDS")
    if idx == -1:
        raise SystemExit("Could not find insertion point in config cell")
    return src[: idx + 2] + CONFIG_INSERT + src[idx + 2 :]


def set_cell_source(cell, text: str):
    lines = text.splitlines(keepends=True)
    if text and not text.endswith("\n"):
        lines[-1] += "\n"
    cell["source"] = lines
    cell["outputs"] = []
    cell["execution_count"] = None


def sanitize_notebook_scene_labels(nb: dict) -> int:
    """Rename escena_id -> scene_id in code cells (axis labels, column names)."""
    n = 0
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if "escena_id" not in src:
            continue
        cell["source"] = [
            line.replace("escena_id", "scene_id") for line in cell["source"]
        ]
        n += 1
    return n


def main():
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    set_cell_source(nb["cells"][3], patch_config_cell("".join(nb["cells"][3]["source"])))
    for idx, src in CELL_SOURCES.items():
        set_cell_source(nb["cells"][idx], src)
    renamed = sanitize_notebook_scene_labels(nb)
    NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("Patched notebook cells:", sorted(CELL_SOURCES.keys()) + [3])
    if renamed:
        print(f"Renamed escena_id -> scene_id in {renamed} additional code cells")


if __name__ == "__main__":
    main()
