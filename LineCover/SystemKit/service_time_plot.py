"""
Visualize the service time as sawtooth function.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple
import math

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from LineCover.SystemKit import figstyle as fs


def _bus_teeth(system, bus, u: Any, v: Any, T: float) -> List[Tuple[float, float, float]]:
    """
    Teeth of one bus over the period T_{u,v}.

    Args:
        system (BusSystem): System of the bus.
        bus (Bus): Bus covering (u, v).
        u (Any): Origin node.
        v (Any): Destination node.
        T (float): Period T_{u,v}.

    Returns:
        List[Tuple[float, float, float]]: (t_start, t_end, C) per tooth, equal teeth merged.
    """
    starts, consts = system._bus_period_dissection(bus, u, v)
    P_b = bus.P_b
    bounds = list(starts) + [float(P_b)]

    teeth: List[Tuple[float, float, float]] = []
    for period in range(int(round(T / P_b))):
        offset = period * P_b
        for m in range(len(starts)):
            t_start, t_end = bounds[m] + offset, bounds[m + 1] + offset
            if t_end <= t_start:
                continue
            intercept = consts[m] + offset
            if teeth and teeth[-1][2] == intercept and teeth[-1][1] == t_start:
                teeth[-1] = (teeth[-1][0], t_end, intercept)
            else:
                teeth.append((t_start, t_end, intercept))

    return teeth


def _draw_sawtooth(
        ax,
        teeth: Sequence[Tuple[float, float, float]],
        color: str,
        T: float,
        linewidth: float,
        dot_diameter: float,
        helpers: bool = True,
        zorder: float = 3.0,
) -> float:
    """
    Draws one sawtooth function with its endpoints.

    Args:
        ax: Matplotlib axis.
        teeth (Sequence): (t_start, t_end, C) per tooth.
        color (str): Colour.
        T (float): End of the period.
        linewidth (float): Line width in points.
        dot_diameter (float): Marker diameter in points.
        helpers (bool): If True, connect the ends of every jump by a dashed line.
        zorder (float): Drawing order.

    Returns:
        float: Largest value.
    """
    top = 0.0
    for index, (t_start, t_end, intercept) in enumerate(teeth):
        y_start, y_end = intercept - t_start, intercept - t_end
        top = max(top, y_start)

        ax.plot([t_start, t_end], [y_start, y_end],
                color=color, lw=linewidth, solid_capstyle="round", zorder=zorder)

        # Open on the left, closed on the right, t = 0 closed and t = T open.
        _dot(ax, t_start, y_start, color, dot_diameter,
             filled=(index == 0), zorder=zorder + 0.1)
        _dot(ax, t_end, y_end, color, dot_diameter,
             filled=(t_end < T), zorder=zorder + 0.1)

        if helpers and index + 1 < len(teeth):
            next_start = teeth[index + 1][2] - t_end
            ax.plot([t_end, t_end], [y_end, next_start],
                    color=color, lw=fs.LW_HELPER, ls=(0, (4, 2)),
                    alpha=0.45, zorder=zorder - 0.5)

    return top


def _dot(ax, x: float, y: float, color: str, diameter: float,
         filled: bool, zorder: float = 4.0) -> None:
    """
    Draws an endpoint marker.

    Args:
        ax: Matplotlib axis.
        x (float): Time.
        y (float): Service time.
        color (str): Marker colour.
        diameter (float): Marker diameter in points.
        filled (bool): True if the value is attained.
        zorder (float): Drawing order.
    """
    ax.plot(
        [x], [y], marker="o", markersize=diameter,
        markerfacecolor=color if filled else "white",
        markeredgecolor=color, markeredgewidth=0.7 if filled else 1.0,
        zorder=zorder, linestyle="none",
    )


def _axes(ax, T: float, y_ticks: Sequence[float], y_top: float) -> None:
    """
    Draws the grey axes of the panel.

    Args:
        ax: Matplotlib axis.
        T (float): End of the time axis.
        y_ticks (Sequence[float]): Ticks of the service time axis.
        y_top (float): Top of the service time axis.
    """
    ax.set_axis_off()
    arrow = dict(arrowstyle="-|>", color=fs.AXIS_GREY, lw=fs.LW_AXIS,
                 mutation_scale=fs.ARROWSIZE, shrinkA=0, shrinkB=0)
    ax.annotate("", xy=(T + 0.9, 0), xytext=(0, 0), arrowprops=arrow)
    ax.annotate("", xy=(0, y_top), xytext=(0, 0), arrowprops=arrow)
    ax.text(T + 0.9, -0.55, "$t$", color=fs.LABEL_GREY,
            fontsize=fs.FS_SCRIPT, ha="right", va="top")

    step = 1 if T <= 12 else max(1, int(round(T / 12)))
    for tick in range(0, int(T) + 1, step):
        ax.plot([tick, tick], [0, -0.35], color=fs.AXIS_GREY, lw=fs.LW_AXIS)
    label_step = step if T <= 4 else 2 * step
    for tick in range(0, int(T) + 1, label_step):
        ax.text(tick, -0.5, f"{tick}", color=fs.LABEL_GREY,
                fontsize=fs.FS_SCRIPT, ha="center", va="top")

    for tick in y_ticks:
        ax.plot([0, -0.18], [tick, tick], color=fs.AXIS_GREY, lw=fs.LW_AXIS)
        value = int(tick) if float(tick).is_integer() else tick
        ax.text(-0.28, tick, f"{value}", color=fs.LABEL_GREY,
                fontsize=fs.FS_SCRIPT, ha="right", va="center")


def _default_y_ticks(values: Sequence[float], maximum: int = 5) -> List[float]:
    """
    Ticks of the service time axis. The values at the endpoints, or evenly spaced integers if there are too many.

    Args:
        values (Sequence[float]): Service times at the endpoints.
        maximum (int): Maximal number of ticks.

    Returns:
        List[float]: Tick positions.
    """
    distinct = sorted({round(value, 6) for value in values if value > 0})
    if len(distinct) <= maximum:
        return distinct

    top = max(distinct)
    step = max(1, int(math.ceil(top / maximum)))

    return [float(tick) for tick in range(step, int(top) + 1, step)]


def visualize_service_time(
        system,
        u: Any,
        v: Any,
        filename: str = "service_time.pdf",
        colors: Optional[Sequence[str]] = None,
        labels: Optional[Sequence[str]] = None,
        show_system: bool = True,
        show_helpers: bool = True,
        show_breakpoints: bool = True,
        annotate_discarded: bool = True,
        y_ticks: Optional[Sequence[float]] = None,
        legend: bool = True,
        figsize: Optional[Tuple[float, float]] = None,
        show: bool = True,
):
    """
    Plots the service time of one pair as sawtooth function.

    Args:
        system (BusSystem): System.
        u (Any): Origin node.
        v (Any): Destination node.
        filename (str): Output path, .pdf or .svg for vector output.
        colors (Sequence[str], optional): One colour per covering bus.
        labels (Sequence[str], optional): Legend labels.
        show_system (bool): Whether to draw the system service time.
        show_helpers (bool): Whether to connect the ends of a jump by a dashed line.
        show_breakpoints (bool): Whether to label the true breakpoints.
        annotate_discarded (bool): Whether to mark candidate breakpoints that are no true ones.
        y_ticks (Sequence[float], optional): Ticks of the vertical axis.
        legend (bool): Whether to draw the legend.
        figsize (tuple, optional): Figure size.
        show (bool): Whether to show the figure.

    Returns:
        matplotlib.figure.Figure: The saved figure.

    Raises:
        ValueError: If (u, v) is covered by no bus.
    """
    fs.apply_style()

    covering = [bus for bus in system.buses.values() if bus.covers(u, v)]
    if not covering:
        raise ValueError(f"Pair {(u, v)} is covered by no bus.")

    T = float(math.lcm(*(bus.P_b for bus in covering)))
    if colors is None:
        colors = [fs.BUS_COLORS[i % len(fs.BUS_COLORS)] for i in range(len(covering))]
    if labels is None:
        labels = [f"$s_{{b_{{{bus.id}}}}}(\\cdot,u,v)$" for bus in covering]

    teeth: Dict[Any, List[Tuple[float, float, float]]] = {
        bus.id: _bus_teeth(system, bus, u, v, T) for bus in covering
    }
    system_teeth = [(t_start, t_end, intercept)
                    for t_start, t_end, intercept, _ in system.sweep_pair(u, v)]

    # Candidate breakpoints and the ones removed by Lemma 4.3 (3).
    candidates = {round(t_end, 9)
                  for bus_teeth in teeth.values()
                  for _, t_end, _ in bus_teeth[:-1]}
    true_points = sorted({round(t_start, 9) for t_start, _, _ in system_teeth
                          if t_start > 0})
    discarded = sorted(candidates - set(true_points))

    endpoint_values = [intercept - t for bus_teeth in teeth.values()
                       for t_start, t_end, intercept in bus_teeth
                       for t in (t_start, t_end)]
    y_max = max(endpoint_values)
    y_top = y_max + 1.2
    if y_ticks is None:
        y_ticks = _default_y_ticks(endpoint_values)

    xlim, ylim = (-0.8, T + 1.2), (-2.8, y_top)
    unit_x, unit_y = fs.ST_UNIT_X_CM, fs.ST_UNIT_Y_CM
    if figsize is not None:
        unit_x = figsize[0] / (xlim[1] - xlim[0]) * 2.54
        unit_y = figsize[1] / (ylim[1] - ylim[0]) * 2.54
    fig, ax = fs.exact_axes(xlim, ylim, unit_x_cm=unit_x, unit_y_cm=unit_y,
                            bottom_in=fs.ST_LEGEND_IN if legend else 0.0)

    # System service time below the buses.
    if show_system:
        _draw_sawtooth(ax, system_teeth, fs.SYS_GREY, T,
                       linewidth=fs.LW_SYS,
                       dot_diameter=fs.HALO_DIAMETER_PT,
                       helpers=False, zorder=1.0)

    if annotate_discarded:
        for index, t in enumerate(discarded):
            ax.plot([t, t], [0, y_top - 0.6], color=fs.AXIS_GREY,
                    lw=fs.LW_HELPER, ls=(0, (1, 2)), zorder=0.5)
            if index == 0:
                ax.text(t + 0.15, y_top - 0.9, "no true breakpoint",
                        color=fs.HINT_GREY, fontsize=fs.FS_SCRIPT,
                        ha="left", va="bottom")

    _axes(ax, T, y_ticks, y_top)

    for bus, color in zip(covering, colors):
        _draw_sawtooth(ax, teeth[bus.id], color, T,
                       linewidth=fs.LW_SAWTOOTH,
                       dot_diameter=fs.DOT_DIAMETER_PT,
                       helpers=show_helpers, zorder=3.0)

    if show_breakpoints:
        for index, t in enumerate(true_points, start=1):
            ax.text(t, -1.35, f"$t^{{u,v}}_{{{index}}}$", color=fs.HINT_GREY,
                    fontsize=fs.FS_SCRIPT, ha="center", va="top")

    if legend:
        handles = [Line2D([0], [0], color=color, lw=fs.LW_SAWTOOTH * 1.4,
                          label=label)
                   for color, label in zip(colors, labels)]
        if show_system:
            handles.append(Line2D([0], [0], color=fs.SYS_GREY, lw=fs.LW_SYS,
                                  label="$s(\\cdot,u,v)$"))
        ax.legend(
            handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.01),
            ncol=min(len(handles), 3), frameon=False, fontsize=fs.FS_SCRIPT,
            handlelength=1.6, columnspacing=1.8, borderpad=0.0,
            handletextpad=0.5,
        )

    fig.savefig(filename, bbox_inches="tight", transparent=True, dpi=300)
    if show:
        plt.show()
    plt.close(fig)

    return fig
