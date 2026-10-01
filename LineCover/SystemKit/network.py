from typing import Any, List, Tuple, Optional, Dict
import math

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D

from LineCover.SystemKit import figstyle as fs

class Network:
    """
    Represents a graph network.

    Attributes:
        undi_graph (nx.Graph): Underlying undirected graph.
        graph (nx.DiGraph): Simple symmetric directed graph based on the undirected graph.
        pos (dict): Optional dictionary of node positions as {node: (x, y)}.
    """
    def __init__(
            self,
            edge_list: List[Tuple],
            pos: Optional[Dict] = None,
            seed: int = 42
    ):
        """

        """
        # Initialize graphs.
        self.undi_graph = nx.Graph(edge_list)
        self.graph = nx.DiGraph(self.undi_graph)

        # Initialize positions to visualize.
        if pos is None:
            self.pos = nx.spring_layout(self.graph, seed=seed)
        else:
            self.pos = pos

    def visualize_graph(
            self,
            bus_arc_list: list = [],
            filename: str = "graph.png",
            offset: float = 0.1,
            width: int = 8,
            height: int = 8
    ):
        """
        Visualize a NetworkX graph using PyVis, save as HTML, and display in Jupyter.
        If `is_busline` is True, edges are labeled and nodes retain their positions from `G`.

        Args:
            bus_arc_list (List[tuple]): Ordered list of bus edges for labeling if it is a bus.
            filename (str): Path to save HTML.
            width (int): Width in inches.
            height (int): Height in inches.
        """
        plt.style.use('default')
        plt.figure(figsize=(width, height), facecolor='white')

        # Draw nodes.
        nx.draw_networkx_nodes(self.graph, self.pos, node_size=700, node_color='skyblue')
        nx.draw_networkx_labels(self.graph, self.pos, font_weight='bold')

        # Draw arcs.
        if bus_arc_list:
            arcs_to_show = bus_arc_list

            for i, arc in enumerate(arcs_to_show, start=1):
                u, v = arc
                x1, y1 = self.pos[u]
                x2, y2 = self.pos[v]

                # Compute the vector of the arc.
                dx = x2 - x1
                dy = y2 - y1
                # Compute the center of the arc.
                x_mid = (x1 + x2) / 2
                y_mid = (y1 + y2) / 2

                # Compute the normal vector.
                px = -dy
                py = dx
                # Normalize the normal vector.
                length = (px ** 2 + py ** 2) ** 0.5
                if length > 0:
                    px /= length
                    py /= length

                # Compute the position near to the peak of the curved arc.
                x = x_mid - offset * px
                y = y_mid - offset * py

                plt.text(
                    x, y, f"{i}.",
                    color='red',
                    fontsize=12,
                    ha='center',
                    va='center',
                    bbox=dict(facecolor='none', edgecolor='none')
                )
        else:
            arcs_to_show = self.graph.edges()

        nx.draw_networkx_edges(
            self.graph, self.pos,
            edgelist=arcs_to_show,
            arrowsize=20,
            node_size=700,
            connectionstyle='arc3,rad=0.1',
            arrows=True,
            arrowstyle='-|>',
            min_source_margin=10,
            min_target_margin=10,
            width=2.0,
        )

        # Some more options.
        plt.title(filename)
        plt.axis('off')

        # Save the plot.
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor=plt.gcf().get_facecolor())

        # Show the plot.
        plt.show()

    # Space below a panel for caption and legend in inches.
    _CAPTION_IN = 0.22
    _LEGEND_IN = 0.30

    def _panel(self, pos, unit_cm, figsize, margin=0.6, bottom_in=0.0):
        """
        Empty network panel in which one graph unit is unit_cm on paper.

        Args:
            pos (dict): Normalized node positions.
            unit_cm (float): Length of one graph unit in centimetres.
            figsize (tuple, optional): Panel size in inches.
            margin (float): Margin in graph units.
            bottom_in (float): Space below the axes in inches.

        Returns:
            Tuple: fig, ax.
        """
        xlim, ylim = fs.layout_limits(pos, margin=margin)
        unit_cm = fs.fit_unit(pos, margin, figsize, unit_cm)
        fig, ax = fs.exact_axes(xlim, ylim, unit_x_cm=unit_cm, unit_y_cm=unit_cm,
                                bottom_in=bottom_in)
        ax.set_aspect("equal")

        return fig, ax

    def _draw_base(self, ax, pos):
        """
        Draws the grey base graph with all nodes.

        Args:
            ax: Matplotlib axis.
            pos (dict): Normalized node positions.
        """
        nx.draw_networkx_edges(
            self.graph, pos, ax=ax,
            edgelist=list(self.undi_graph.edges()),
            edge_color=fs.GRID_EDGE, width=fs.LW_GRID, arrows=False,
        )
        nx.draw_networkx_nodes(
            self.graph, pos, ax=ax,
            node_size=fs.node_size(), node_color=fs.STOP_FILL,
            edgecolors=fs.STOP_EDGE, linewidths=fs.LW_STOP,
        )

    def _draw_od_pair(self, ax, pos, od_pair):
        """
        Draws the pair (u, v) in orange with labels u and v.

        Args:
            ax: Matplotlib axis.
            pos (dict): Normalized node positions.
            od_pair (tuple): (u, v).
        """
        u, v = od_pair
        nx.draw_networkx_nodes(
            self.graph, pos, ax=ax, nodelist=[u, v],
            node_size=fs.node_size(), node_color=fs.OD_FILL,
            edgecolors=fs.PIKORANGE, linewidths=fs.LW_ODSTOP,
        )
        nx.draw_networkx_labels(
            self.graph, pos, ax=ax, labels={u: "$u$", v: "$v$"},
            font_size=fs.FS_SCRIPT, font_color=fs.PIKORANGE,
        )

    @staticmethod
    def _arc_curvature(arc_list, bend_deg, stagger=0.0):
        """
        Curvature of every arc. Only arcs whose reverse is also in the route are bent.

        Args:
            arc_list (list): Arcs of the route.
            bend_deg (float): Bend angle in degrees.
            stagger (float): Additional curvature of every arc.

        Returns:
            list: rad value of arc3 per arc.
        """
        # TikZ bend theta corresponds to rad = tan(theta)/2.
        bend = math.tan(math.radians(bend_deg)) / 2.0
        arcs = set(arc_list)
        rads = []
        for (a, b) in arc_list:
            rads.append(stagger + (bend if (b, a) in arcs else 0.0))

        return rads

    def _draw_route(self, ax, pos, arc_list, color, rads):
        """
        Draws a directed route.

        Args:
            ax: Matplotlib axis.
            pos (dict): Normalized node positions.
            arc_list (list): Arcs of the route.
            color (str): Route colour.
            rads (list): Curvature per arc.
        """
        for arc, rad in zip(arc_list, rads):
            nx.draw_networkx_edges(
                self.graph, pos, ax=ax, edgelist=[arc],
                edge_color=[to_rgba(color)], width=fs.LW_ARC,
                arrows=True, arrowstyle="-|>", arrowsize=fs.ARROWSIZE,
                connectionstyle=f"arc3,rad={rad}",
                node_size=fs.node_size(), min_source_margin=2, min_target_margin=2,
            )

    def _draw_order_labels(self, fig, ax, pos, arc_list, color, rads, offset):
        """
        Numbers the arcs of a route in order. Labels of straight arcs face away from the centroid.

        Args:
            fig: Matplotlib figure.
            ax: Matplotlib axis.
            pos (dict): Normalized node positions.
            arc_list (list): Arcs of the route.
            color (str): Label colour.
            rads (list): Curvature per arc.
            offset (float): Distance from the arc in points.
        """
        fig.canvas.draw()
        trans = ax.transData
        inv = trans.inverted()
        (CX, CY) = trans.transform(np.mean([pos[n] for n in self.graph.nodes()], axis=0))
        for i, ((a, b), rad) in enumerate(zip(arc_list, rads), start=1):
            (X1, Y1) = trans.transform(pos[a])
            (X2, Y2) = trans.transform(pos[b])
            ddx, ddy = X2 - X1, Y2 - Y1
            length = (ddx ** 2 + ddy ** 2) ** 0.5 or 1.0
            # Outward normal.
            bx, by = ddy / length, -ddx / length
            if rad == 0:
                # Straight arc faces away from the centroid.
                side = ((X1 + X2) / 2 - CX) * bx + ((Y1 + Y2) / 2 - CY) * by
                if side < -1e-6 * length:
                    bx, by = -bx, -by
            apex = ((X1 + X2) / 2 + 0.5 * rad * ddy,
                    (Y1 + Y2) / 2 - 0.5 * rad * ddx)
            ax.annotate(
                f"{i}.", xy=inv.transform(apex),
                xytext=(offset * bx, offset * by), textcoords="offset points",
                ha="center", va="center", zorder=5,
                color=color, fontsize=fs.FS_SCRIPT,
            )

    @staticmethod
    def _bus_caption(bus, name=None):
        """
        Caption of a bus panel, e.g. $b_1$: $P_{b_1}=8$.

        Args:
            bus: Bus object.
            name (str, optional): Name of the bus, default b_{id}.

        Returns:
            str: Caption, or empty string if bus has no period.
        """
        period = getattr(bus, "P_b", None)
        if period is None:
            return ""
        if name is None:
            name = f"b_{{{getattr(bus, 'id', '')}}}".replace("b_{}", "b")
        t_b = getattr(bus, "t_b", None)
        caption = f"${name}$:  $P_{{{name}}}={period}$"
        if t_b is not None:
            t_b = int(t_b) if float(t_b).is_integer() else t_b
            caption += f",  $t^{{{name}}}={t_b}$"

        return caption

    @staticmethod
    def _as_arc_list(bus):
        """
        Arc list of a Bus object or an arc list.

        Args:
            bus: Bus object or list of arcs.

        Returns:
            list: The arc list.
        """
        return bus.ArcList if hasattr(bus, "ArcList") else bus

    def visualize_bus(
            self,
            bus,
            filename: str = "bus.pdf",
            color: str = fs.TUBLUE,
            od_pair: Optional[Tuple[Any, Any]] = None,
            caption: Optional[str] = None,
            show_order: bool = True,
            show_labels: bool = True,
            order_offset: float = 9.0,
            bend_deg: float = fs.ARC_BEND_DEG,
            unit_cm: float = fs.UNIT_CM,
            figsize: Optional[tuple] = None,
            show: bool = True,
    ):
        """
        Visualizes the network with one bus route, as in Figure 2 of the thesis.

        Args:
            bus: Bus object or cyclic list of arcs.
            filename (str): Output path, .pdf or .svg for vector output.
            color (str): Route colour.
            od_pair (tuple, optional): Pair (u, v) to highlight.
            caption (str, optional): Caption below the panel. Built from the bus if None, "" for no caption.
            show_order (bool): If True, number the arcs.
            show_labels (bool): If True, print the node ids.
            order_offset (float): Distance of the order labels from the arc in points.
            bend_deg (float): Bend angle of arcs travelled in both directions.
            unit_cm (float): Length of one graph unit in centimetres.
            figsize (tuple, optional): Figure size.
            show (bool): Whether to show the figure.
        """
        fs.apply_style()
        arc_list = self._as_arc_list(bus)
        pos = fs.normalized_layout(self.pos, self.undi_graph.edges())
        if caption is None:
            caption = self._bus_caption(bus, name="b")

        fig, ax = self._panel(pos, unit_cm, figsize,
                              bottom_in=self._CAPTION_IN if caption else 0.0)
        self._draw_base(ax, pos)
        rads = self._arc_curvature(arc_list, bend_deg)
        self._draw_route(ax, pos, arc_list, color, rads)
        if od_pair is not None:
            self._draw_od_pair(ax, pos, od_pair)
        if show_labels:
            labels = {n: str(n) for n in self.graph.nodes()
                      if od_pair is None or n not in od_pair}
            nx.draw_networkx_labels(
                self.graph, pos, ax=ax, labels=labels,
                font_size=fs.FS_SCRIPT, font_color=fs.INK,
            )

        ax.set_axis_off()
        if caption:
            fig.text(0.5, 0.0, caption, color=color, fontsize=fs.FS_SCRIPT,
                     ha="center", va="bottom")

        if show_order:
            self._draw_order_labels(fig, ax, pos, arc_list, color, rads,
                                    order_offset)

        fig.savefig(filename, bbox_inches="tight", transparent=True, dpi=300)
        if show:
            plt.show()
        plt.close(fig)

        return fig

    def visualize_buses(
            self,
            buses: list,
            filename: str = "buses.pdf",
            colors: Optional[list] = None,
            labels: Optional[list] = None,
            od_pair: Optional[Tuple[Any, Any]] = None,
            show_order: bool = False,
            show_labels: bool = True,
            order_offset: float = 9.0,
            bend_deg: float = fs.ARC_BEND_DEG,
            stagger: float = 0.10,
            legend: bool = True,
            unit_cm: float = fs.UNIT_CM,
            figsize: Optional[tuple] = None,
            show: bool = True,
    ):
        """
        Visualizes the network with several bus routes in different colours.

        Args:
            buses (list): Bus objects or arc lists.
            filename (str): Output path, .pdf or .svg for vector output.
            colors (list, optional): Route colours, default BUS_COLORS.
            labels (list, optional): Legend labels, built from the buses if None.
            od_pair (tuple, optional): Pair (u, v) to highlight.
            show_order (bool): If True, number the arcs of every route.
            show_labels (bool): If True, print the node ids.
            order_offset (float): Distance of the order labels from the arc.
            bend_deg (float): Bend angle of arcs travelled in both directions.
            stagger (float): Extra curvature per bus, applied as 0, +s, -s, +2s, -2s, ...
            legend (bool): Whether to draw the legend.
            unit_cm (float): Length of one graph unit in centimetres.
            figsize (tuple, optional): Figure size.
            show (bool): Whether to show the figure.
        """
        fs.apply_style()
        pos = fs.normalized_layout(self.pos, self.undi_graph.edges())

        arc_lists, auto_labels = [], []
        for k, b in enumerate(buses):
            arc_lists.append(self._as_arc_list(b))
            name = f"b_{{{getattr(b, 'id', k + 1)}}}"
            auto_labels.append(self._bus_caption(b, name=name) or f"${name}$")
        if labels is None:
            labels = auto_labels
        if colors is None:
            colors = [fs.BUS_COLORS[i % len(fs.BUS_COLORS)]
                      for i in range(len(arc_lists))]

        fig, ax = self._panel(pos, unit_cm, figsize,
                              bottom_in=self._LEGEND_IN if legend else 0.0)
        self._draw_base(ax, pos)

        # Buses sharing an arc are drawn on opposite sides.
        staggers = [stagger * ((i + 1) // 2) * (1 if i % 2 else -1)
                    for i in range(len(arc_lists))]
        all_rads = []
        for arc_list, color, offset in zip(arc_lists, colors, staggers):
            rads = self._arc_curvature(arc_list, bend_deg, stagger=offset)
            all_rads.append(rads)
            self._draw_route(ax, pos, arc_list, color, rads)

        if od_pair is not None:
            self._draw_od_pair(ax, pos, od_pair)
        if show_labels:
            node_labels = {n: str(n) for n in self.graph.nodes()
                           if od_pair is None or n not in od_pair}
            nx.draw_networkx_labels(
                self.graph, pos, ax=ax, labels=node_labels,
                font_size=fs.FS_SCRIPT, font_color=fs.INK,
            )

        if legend:
            handles = [Line2D([0], [0], color=c, lw=fs.LW_ARC * 1.6, label=lab)
                       for c, lab in zip(colors, labels)]
            ax.legend(
                handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0),
                ncol=min(len(handles), 3), frameon=False,
                fontsize=fs.FS_SCRIPT, handlelength=1.6, columnspacing=1.6,
                borderpad=0.0, handletextpad=0.5,
            )

        ax.set_axis_off()

        if show_order:
            for arc_list, color, rads in zip(arc_lists, colors, all_rads):
                self._draw_order_labels(fig, ax, pos, arc_list, color, rads,
                                        order_offset)

        fig.savefig(filename, bbox_inches="tight", transparent=True, dpi=300)
        if show:
            plt.show()
        plt.close(fig)

        return fig


def get_grid_graph_data(m: int, n: int) -> Tuple[List[Tuple[Any, Any]], Dict[Any, Any]]:
    """
    Computes the edge list and positions of a m x n Grid graph.

    Parameters:
        - m, n: Dimensions of the grid (rows, columns).

    Returns:
        - List of edges as tuples: [(u, v), ...]
        - Positions dictionary: {node: (x, y), ...}
    """
    graph = nx.grid_2d_graph(m, n)

    # Relabel the nodes of (x,y) to 0, 1, 2, ...
    mapping = {node: i for i, node in enumerate(graph.nodes())}
    graph_relabeled = nx.relabel_nodes(graph, mapping)
    edge_list = list(graph_relabeled.edges())
    pos = {
        i: (i % n, -(i // n))
        for i in range(m * n)
    }

    return edge_list, pos

def get_cycle_graph_data(n: int) -> Tuple[List[Tuple[Any, Any]], Dict[Any, Tuple[float, float]]]:
    """
    Computes the edge list and positions of a circular graph (cycle graph) with `n` nodes.

    Parameters:
        - n: Number of nodes in the cycle.

    Returns:
        - List of edges as tuples: [(u, v), ...]
        - Positions dictionary: {node: (x, y), ...}
    """
    graph = nx.cycle_graph(n)  # Create a cycle graph with n nodes

    # Relabel nodes to 0, 1, 2, ... (if needed, though cycle_graph already uses 0..n-1)
    # No relabeling needed since nx.cycle_graph(n) already uses 0..n-1

    edge_list = list(graph.edges())

    # Positions for circular layout
    pos = {
        i: (np.cos(2 * np.pi * i / n), np.sin(2 * np.pi * i / n))
        for i in range(n)
    }

    return edge_list, pos

def get_star_graph_data(n: int) -> Tuple[List[Tuple[Any, Any]], Dict[Any, Tuple[float, float]]]:
    """
    Computes the edge list and positions of a star graph with `n` nodes.
    The central node is labeled `0`, and outer nodes are labeled `1` to `n-1`.

    Parameters:
        - n: Total number of nodes in the star (central node + outer nodes).

    Returns:
        - List of edges as tuples: [(0, 1), (0, 2), ..., (0, n-1)]
        - Positions dictionary: {node: (x, y), ...}
    """
    if n < 2:
        raise ValueError("A star graph requires at least 2 nodes (1 center + 1 outer).")

    graph = nx.star_graph(n - 1)  # n-1 outer nodes + 1 center = n total nodes
    edge_list = list(graph.edges())

    # Positions: center at (0, 0), outer nodes on a circle
    pos = {0: (0.0, 0.0)}  # Center node
    pos.update({
        i: (np.cos(2 * np.pi * i / (n - 1)), np.sin(2 * np.pi * i / (n - 1)))
        for i in range(1, n)
    })

    return edge_list, pos

def get_wheel_graph_data(n: int) -> Tuple[List[Tuple[Any, Any]], Dict[Any, Tuple[float, float]]]:
    """
    Computes the edge list and positions of a wheel graph with `n` nodes.
    The central node is labeled `0`, and outer nodes are labeled `1` to `n-1`.

    Parameters:
        - n: Total number of nodes in the wheel (central node + outer cycle).

    Returns:
        - List of edges as tuples: [(0, 1), (0, 2), ..., (0, n-1), (1, 2), (2, 3), ..., (n-1, 1)]
        - Positions dictionary: {node: (x, y), ...}
    """
    graph = nx.wheel_graph(n)  # Creates a wheel with n nodes (1 center + n-1 outer)

    edge_list = list(graph.edges())

    # Positions: center at (0, 0), outer nodes on a circle
    pos = {0: (0.0, 0.0)}  # Center node
    pos.update({
        i: (np.cos(2 * np.pi * i / (n - 1)), np.sin(2 * np.pi * i / (n - 1)))
        for i in range(1, n)
    })

    return edge_list, pos

def get_line_graph_data(n: int) -> Tuple[List[Tuple[Any, Any]], Dict[Any, Any]]:
    """
    Computes the edge list and positions of a line graph with n nodes.

    Parameters:
        - n: Number of nodes in the line graph.

    Returns:
        - List of edges as tuples: [(u, v), ...]
        - Positions dictionary: {node: (x, y), ...}
    """
    graph = nx.path_graph(n)

    edge_list = list(graph.edges())
    pos = {
        i: (i, 0)
        for i in range(n)
    }

    return edge_list, pos