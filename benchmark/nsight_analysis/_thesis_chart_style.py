"""Shared matplotlib/seaborn styling for thesis benchmark figures."""
from __future__ import annotations

import math

import matplotlib.pyplot as plt

FONT_SUPTITLE = 18
FONT_FACET_TITLE = 15
FONT_AXIS_LABEL = 15
FONT_TICK = 13
FONT_LEGEND = 13
FONT_LEGEND_TITLE = 14


def panel_col_wrap(n_panels: int) -> tuple[int, bool]:
    """Narrow layout for 1–2 panels; full grid (col_wrap=3) otherwise."""
    if n_panels <= 2:
        return n_panels, False
    return min(n_panels, 3), True


def apply_thesis_chart_style(
    g,
    *,
    n_panels: int,
    col_wrap: int,
    legend_out: bool,
    suptitle: str,
    x_label: str = "Range",
    y_label: str = "Value",
) -> None:
    n_rows = max(1, math.ceil(n_panels / col_wrap))

    g.set_titles("{col_name}", size=FONT_FACET_TITLE, pad=10)
    g.set_axis_labels(x_label, y_label, fontsize=FONT_AXIS_LABEL)
    for ax in g.axes.flat:
        if ax is not None:
            ax.tick_params(labelsize=FONT_TICK)

    bottom = 0.10 if legend_out else 0.26
    top = 0.78 if n_rows == 1 else (0.84 if n_rows == 2 else 0.88)
    hspace = 0.42 if n_rows > 1 else 0.30
    g.fig.subplots_adjust(top=top, bottom=bottom, hspace=hspace, wspace=0.28)

    suptitle_y = top + (0.14 if n_rows == 1 else 0.10)
    g.fig.suptitle(suptitle, fontsize=FONT_SUPTITLE, y=suptitle_y)

    if legend_out:
        leg = g._legend
        if leg is not None:
            leg.set_title("Grid", prop={"size": FONT_LEGEND_TITLE})
            for text in leg.get_texts():
                text.set_fontsize(FONT_LEGEND)
    else:
        if g._legend is not None:
            g._legend.remove()
        handles, labels = g.axes.flat[0].get_legend_handles_labels()
        if handles:
            ncol = min(4, len(labels))
            g.fig.legend(
                handles,
                labels,
                title="Grid",
                loc="upper center",
                bbox_to_anchor=(0.5, 0.01),
                ncol=ncol,
                fontsize=FONT_LEGEND,
                title_fontsize=FONT_LEGEND_TITLE,
                frameon=True,
            )


def style_time_per_mark_figure(g, *, n_panels: int, col_wrap: int, legend_out: bool, suptitle: str) -> None:
    apply_thesis_chart_style(
        g,
        n_panels=n_panels,
        col_wrap=col_wrap,
        legend_out=legend_out,
        suptitle=suptitle,
        y_label="Duration (ms)",
    )
