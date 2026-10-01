"""
Visualization of data of a bus system.
"""
from typing import Any, Dict, List, Optional, Tuple
import math
import random

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import FancyArrowPatch

from LineCover.SystemKit import figstyle as fs


def _plain_axes(ax) -> None:
    """
    Removes the top and right spine and greys the remaining ones.

    Args:
        ax: Axes to style.
    """
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(fs.AXIS_GREY)
        ax.spines[side].set_linewidth(fs.LW_AXIS)
    ax.tick_params(labelsize=fs.FS_SCRIPT, color=fs.AXIS_GREY, labelcolor=fs.LABEL_GREY)


def period_histogram(systems: Dict[str, Any], filename: str = "periods.pdf",
                     colors: Optional[list] = None, bins: int = 30,
                     panel_size: Tuple[float, float] = (2.9, 2.6), show: bool = True):
    """
    Histogram of the circuit lengths P_b, one panel per system.

    Args:
        systems (Dict[str, Any]): Label -> BusSystem, one panel each, in the given order.
        filename (str): Output path, .pdf or .svg for vector output.
        colors (Optional[list]): Colours of the panels, TUred, TUblue and PIKorange by default.
        bins (int): Number of bins, shared by all panels.
        panel_size (Tuple[float, float]): Size of a single panel in inches.
        show (bool): Whether to show the figure.

    Returns:
        Tuple: fig, axes.

    Notes:
        - All panels share the bins and both axes, so that their heights are comparable.
    """
    fs.apply_style()
    colors = colors or [fs.TURED, fs.TUBLUE, fs.PIKORANGE]
    periods = {label: [bus.P_b for bus in system.buses.values()] for label, system in systems.items()}
    low = min(min(values) for values in periods.values())
    high = max(max(values) for values in periods.values())

    width, height = panel_size
    fig, axes = plt.subplots(1, len(periods), sharex=True, sharey=True,
                             figsize=(width * len(periods), height))
    axes = axes if len(periods) > 1 else [axes]
    for ax, color, (label, values) in zip(axes, colors * len(periods), periods.items()):
        ax.hist(values, bins=bins, range=(low, high), color=color, alpha=0.65)
        ax.set_title(label, fontsize=fs.FS_SMALL, color=fs.LABEL_GREY)
        ax.set_xlabel("circuit length $P_b$", fontsize=fs.FS_SMALL)
        _plain_axes(ax)
    axes[0].set_ylabel("number of buses", fontsize=fs.FS_SMALL)
    fig.tight_layout()
    fig.savefig(filename)
    if show:
        plt.show()

    return fig, axes


def pair_periods(system, sample: int = 2000, seed: int = 0) -> List[int]:
    """
    Pair periods T_uv of randomly drawn node pairs.

    Args:
        system: BusSystem to read.
        sample (int): Number of pairs to draw.
        seed (int): Seed of the draw.

    Returns:
        List[int]: T_uv, the least common multiple of the periods of the buses serving the pair.
    """
    rng, nodes, values = random.Random(seed), sorted(system.nodes), []
    for _ in range(sample):
        u, v = rng.sample(nodes, 2)
        serving = [bus.P_b for bus in system.buses.values() if bus.covers(u, v)]
        if serving:
            values.append(math.lcm(*serving))

    return values


def pair_period_histogram(system, filename: str = "pair_periods.pdf", sample: int = 2000,
                          seed: int = 0, color: Optional[str] = None,
                          figsize: Tuple[float, float] = (5.2, 3.0), show: bool = True):
    """
    Histogram of the number of decimal digits of the pair periods T_uv.

    Args:
        system: BusSystem to read.
        filename (str): Output path, .pdf or .svg for vector output.
        sample (int): Number of pairs to draw.
        seed (int): Seed of the draw.
        color (Optional[str]): Colour of the bars, TUblue by default.
        figsize (Tuple[float, float]): Figure size in inches.
        show (bool): Whether to show the figure.

    Returns:
        Tuple: fig, ax.

    Notes:
        - T_uv spans many orders of magnitude, so the digits are counted instead of the value.
    """
    fs.apply_style()
    digits = [len(str(value)) for value in pair_periods(system, sample=sample, seed=seed)]
    low, high = min(digits), max(digits)

    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(digits, bins=range(low, high + 2), align="left", color=color or fs.TUBLUE, alpha=0.65)
    ax.set_xlabel("decimal digits of $T_{u,v}$", fontsize=fs.FS_SMALL)
    ax.set_ylabel("number of pairs", fontsize=fs.FS_SMALL)
    ax.set_xticks(range(low, high + 1))
    _plain_axes(ax)
    fig.tight_layout()
    fig.savefig(filename)
    if show:
        plt.show()

    return fig, ax


def arc_load(system) -> Dict[Tuple[Any, Any], int]:
    """
    Number of bus traversals of every arc.

    Args:
        system: BusSystem to read.

    Returns:
        Dict[Tuple[Any, Any], int]: Arc -> how often a bus travels it per period.

    Notes:
        - A bus that travels an arc twice in one period counts twice.
    """
    load: Dict[Tuple[Any, Any], int] = {}
    for bus in system.buses.values():
        for arc in bus.ArcList:
            key = (arc[0], arc[1])
            load[key] = load.get(key, 0) + 1

    return load


def route_figure(network, buses: list, filename: str = "routes.pdf", labels: Optional[list] = None,
                 colors: Optional[list] = None, unit_cm: float = 0.35,
                 bend_deg: float = fs.ARC_BEND_DEG, stagger: float = 0.06,
                 legend: bool = True, show: bool = True):
    """
    Draws a few routes on the base graph, without node markers.

    Args:
        network (Network): Network of the system, for the node positions.
        buses (list): Bus objects to draw.
        filename (str): Output path, .pdf or .svg for vector output.
        labels (Optional[list]): Legend labels, the circuit lengths by default.
        colors (Optional[list]): Route colours, figstyle.BUS_COLORS by default.
        unit_cm (float): Length of one graph unit in centimetres.
        bend_deg (float): Bend angle of arcs travelled in both directions.
        stagger (float): Additional curvature per route, so that shared arcs stay visible.
        legend (bool): Whether to draw the legend.
        show (bool): Whether to show the figure.

    Returns:
        Tuple: fig, ax.

    Notes:
        - Meant for grids too large for visualize_buses, whose node markers do not scale.
    """
    fs.apply_style()
    pos = fs.normalized_layout(network.pos, network.undi_graph.edges())
    colors = colors or fs.BUS_COLORS
    labels = labels or [f"$P_b = {bus.P_b}$" for bus in buses]

    xs = [point[0] for point in pos.values()]
    ys = [point[1] for point in pos.values()]
    width = (max(xs) - min(xs) + 1) * unit_cm / 2.54
    height = (max(ys) - min(ys) + 1) * unit_cm / 2.54
    bottom = fs.ST_LEGEND_IN if legend else 0.0
    if legend:
        width = max(width, 0.9 * len(buses) + 0.4)   # room for the legend entries

    fig, ax = plt.subplots(figsize=(width, height + bottom))
    fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=bottom / (height + bottom))
    ax.add_collection(LineCollection([(pos[u], pos[v]) for u, v in network.undi_graph.edges()],
                                     colors=fs.GRID_EDGE, linewidths=fs.LW_GRID, zorder=1))

    for index, bus in enumerate(buses):
        color = colors[index % len(colors)]
        shift = (index - (len(buses) - 1) / 2.0) * stagger
        for arc, rad in zip(bus.ArcList, network._arc_curvature(bus.ArcList, bend_deg, shift)):
            ax.add_patch(FancyArrowPatch(pos[arc[0]], pos[arc[1]], arrowstyle="-",
                                         connectionstyle=f"arc3,rad={rad}", color=color,
                                         linewidth=fs.LW_ARC, shrinkA=0, shrinkB=0, zorder=2))
        ax.plot([], [], color=color, linewidth=fs.LW_ARC, label=labels[index])

    ax.set_xlim(min(xs) - 0.5, max(xs) + 0.5)
    ax.set_ylim(min(ys) - 0.5, max(ys) + 0.5)
    ax.set_aspect("equal")
    ax.axis("off")
    if legend:
        fig.legend(fontsize=fs.FS_SCRIPT, frameon=False, ncol=len(buses),
                   loc="lower center", handlelength=1.4, columnspacing=1.2)
    fig.savefig(filename)
    if show:
        plt.show()

    return fig, ax


def load_heatmap(network, system, filename: str = "arc_load.pdf",
                 unit_cm: float = 0.35, colorbar: bool = True, show: bool = True):
    """
    Draws the graph with every edge coloured by the number of buses travelling it.

    Args:
        network (Network): Network of the system, for the node positions.
        system: BusSystem to draw.
        filename (str): Output path, .pdf or .svg for vector output.
        unit_cm (float): Length of one graph unit in centimetres.
        colorbar (bool): Whether to draw the colour bar.
        show (bool): Whether to show the figure.

    Returns:
        Tuple: fig, ax.

    Notes:
        - Both directions of an edge are summed, so the figure shows the load of the street.
    """
    fs.apply_style()
    pos = fs.normalized_layout(network.pos, network.undi_graph.edges())
    load = arc_load(system)

    segments, values = [], []
    for u, v in network.undi_graph.edges():
        if u not in pos or v not in pos:
            continue
        segments.append((pos[u], pos[v]))
        values.append(load.get((u, v), 0) + load.get((v, u), 0))

    colormap = LinearSegmentedColormap.from_list("load", [fs.STOP_FILL, fs.TUBLUE])
    norm = Normalize(vmin=0, vmax=max(values) if values else 1)
    widths = [fs.LW_GRID + 2.2 * norm(value) for value in values]

    xs = [point[0] for point in pos.values()]
    ys = [point[1] for point in pos.values()]
    width = (max(xs) - min(xs) + 1) * unit_cm / 2.54
    height = (max(ys) - min(ys) + 1) * unit_cm / 2.54

    fig, ax = plt.subplots(figsize=(width * (1.25 if colorbar else 1.0), height))
    ax.add_collection(LineCollection(segments, colors=colormap(norm(values)), linewidths=widths,
                                     capstyle="round"))
    ax.set_xlim(min(xs) - 0.5, max(xs) + 0.5)
    ax.set_ylim(min(ys) - 0.5, max(ys) + 0.5)
    ax.set_aspect("equal")
    ax.axis("off")
    if colorbar:
        bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=colormap), ax=ax,
                           fraction=0.04, pad=0.02)
        bar.set_label("Bus traversals per edge", fontsize=fs.FS_SCRIPT)
        bar.ax.tick_params(labelsize=fs.FS_SCRIPT, colors=fs.LABEL_GREY)
        bar.outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(filename)
    if show:
        plt.show()

    return fig, ax
