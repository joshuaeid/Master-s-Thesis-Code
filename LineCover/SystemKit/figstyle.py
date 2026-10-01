"""
Figure style of the LineCover plots, same as the TikZ styles for Latex.
"""

from typing import Any, Dict, Iterable, Optional, Tuple

import matplotlib.pyplot as plt

# Colours.

TUBLUE = "#005B7F"        # \definecolor{TUblue}
TURED = "#C40D1E"         # \definecolor{TUred}
PIKORANGE = "#E37222"     # \definecolor{PIKorange}

# LaTeX greys on a white page.
GRID_EDGE = "#DFDFDF"     # gray!25, base graph edges (stgrid)
STOP_FILL = "#F0F0F0"     # gray!12, node fill (stop)
STOP_EDGE = "#A6A6A6"     # gray!70, node outline (stop)
OD_FILL = "#FCEEE4"       # PIKorange!12, fill of the u/v nodes (odstop)
AXIS_GREY = "#B9B9B9"     # gray!55, axes and ticks
LABEL_GREY = "#A6A6A6"    # gray!70, tick labels, breakpoint labels
HINT_GREY = "#9E9E9E"     # gray!75, annotations
SYS_GREY = "#8C8C8C"      # black!45, system service time band (sysline)

# Colours of the buses.
BUS_COLORS = [
    TUBLUE,
    TURED,
    PIKORANGE,
    "#009E73", "#CC79A7", "#56B4E9", "#999933", "#0072B2",
]

INK = "#1F2430"           # node label ink

# Line widths and sizes.

LW_GRID = 0.6             # stgrid
LW_STOP = 0.4             # stop (TikZ default draw width)
LW_ODSTOP = 0.9           # odstop
LW_ARC = 1.0              # arc
LW_SAWTOOTH = 1.1         # s_b in the service-time panel
LW_SYS = 4.0              # sysline
LW_HELPER = 0.6           # dashed jump helpers, dotted candidate breakpoints
LW_AXIS = 0.6

STOP_DIAMETER_MM = 6.0    # stop/minimum size
ARROW_LENGTH_PT = 5.0     # Stealth[length=5pt]
ARROWSIZE = ARROW_LENGTH_PT / 0.4   # head length is 0.4 * mutation scale
ARC_BEND_DEG = 20.0       # to[bend left=20]

UNIT_CM = 1.45            # x=1.45cm

# Unit of the service time panel.
ST_UNIT_X_CM = 0.75
ST_UNIT_Y_CM = 0.42

# Space below the service time panel for the legend in inches.
ST_LEGEND_IN = 0.30

# Marker diameters in points.
DOT_DIAMETER_PT = 2 * 1.1 + 0.4      # fdot
HALO_DIAMETER_PT = 2 * 2.1           # fhalo

# Font sizes of \small and \scriptsize.
FS_SMALL = 9.0
FS_SCRIPT = 7.0


def mm_to_pt(mm: float) -> float:
    """
    Converts millimetres to points (1 pt = 1/72 in).

    Args:
        mm (float): Length in millimetres.

    Returns:
        float: The same length in points.
    """
    return mm / 25.4 * 72.0


def node_size(diameter_mm: float = STOP_DIAMETER_MM) -> float:
    """
    Matplotlib node_size (area in pt^2) of a circle.

    Args:
        diameter_mm (float): Node diameter in millimetres.

    Returns:
        float: Area in points squared.
    """
    return mm_to_pt(diameter_mm) ** 2


def apply_style() -> None:
    """
    Sets the matplotlib rcParams to the font of the thesis (Latin Modern).
    """
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Latin Modern Roman", "CMU Serif", "DejaVu Serif"],
        "font.size": FS_SMALL,
        "mathtext.fontset": "cm",
        "svg.fonttype": "none",
        "axes.unicode_minus": False,
    })


def normalized_layout(
        pos: Dict[Any, Tuple[float, float]],
        edges: Iterable[Tuple[Any, Any]],
) -> Dict[Any, Tuple[float, float]]:
    """
    Rescales a layout by the median edge length, so that adjacent nodes are one unit apart.

    Args:
        pos (Dict): Node positions {node: (x, y)}.
        edges (Iterable): Edges of the graph.

    Returns:
        Dict: Rescaled positions.
    """
    lengths = []
    for u, v in edges:
        if u in pos and v in pos:
            (x1, y1), (x2, y2) = pos[u], pos[v]
            lengths.append(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)
    lengths = [length for length in lengths if length > 0]
    if not lengths:
        return dict(pos)

    lengths.sort()
    median = lengths[len(lengths) // 2]

    return {node: (x / median, y / median) for node, (x, y) in pos.items()}


def exact_axes(
        xlim: Tuple[float, float],
        ylim: Tuple[float, float],
        unit_x_cm: float = UNIT_CM,
        unit_y_cm: float = UNIT_CM,
        bottom_in: float = 0.0,
        top_in: float = 0.0,
):
    """
    Figure in which one data unit has a fixed length on paper.

    Args:
        xlim (tuple): Limits of the horizontal axis.
        ylim (tuple): Limits of the vertical axis.
        unit_x_cm (float): Length of one horizontal unit in centimetres.
        unit_y_cm (float): Length of one vertical unit in centimetres.
        bottom_in (float): Space below the axes in inches.
        top_in (float): Space above the axes in inches.

    Returns:
        Tuple: fig, ax with transparent background.
    """
    width = (xlim[1] - xlim[0]) * unit_x_cm / 2.54
    height = (ylim[1] - ylim[0]) * unit_y_cm / 2.54
    total = height + bottom_in + top_in

    fig = plt.figure(figsize=(width, total))
    ax = fig.add_axes([0.0, bottom_in / total, 1.0, height / total])
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    return fig, ax


def fit_unit(
        pos: Dict[Any, Tuple[float, float]],
        margin: float,
        figsize: Optional[Tuple[float, float]],
        unit_cm: float = UNIT_CM,
) -> float:
    """
    Length of one graph unit for a given figure size.

    Args:
        pos (Dict): Normalized node positions.
        margin (float): Margin in graph units.
        figsize (tuple, optional): Figure size in inches. If None, unit_cm is used.
        unit_cm (float): Default length of one graph unit in centimetres.

    Returns:
        float: Length of one graph unit in centimetres.
    """
    if figsize is None:
        return unit_cm

    xs = [x for x, _ in pos.values()]
    ys = [y for _, y in pos.values()]
    span_x = (max(xs) - min(xs)) + 2 * margin
    span_y = (max(ys) - min(ys)) + 2 * margin

    return min(figsize[0] / span_x, figsize[1] / span_y) * 2.54


def layout_limits(
        pos: Dict[Any, Tuple[float, float]],
        margin: float = 0.6,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Limits of the layout plus a margin.

    Args:
        pos (Dict): Normalized node positions.
        margin (float): Margin in graph units.

    Returns:
        Tuple: xlim, ylim.
    """
    xs = [x for x, _ in pos.values()]
    ys = [y for _, y in pos.values()]

    return ((min(xs) - margin, max(xs) + margin),
            (min(ys) - margin, max(ys) + margin))
