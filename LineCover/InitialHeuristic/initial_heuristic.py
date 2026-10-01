from typing import Any, Dict, Tuple
import networkx as nx
from LineCover.SystemKit import Bus
from LineCover.InitialHeuristic.build_buses import build_buses
from LineCover.InitialHeuristic.build_buses_storage import BuildBusesStorage
from LineCover.InitialHeuristic.extend_buses import extend_buses, normalize_uncovered_pairs

def build_initial_solution(G: nx.DiGraph, B: int,
                           lazy_service_time: bool = False) -> Tuple[Dict[Any, Bus], BuildBusesStorage]:
    """
    Start solution with build_buses and extend_buses. The buses are rebuilt with full_init=True.

    Args:
        G (nx.DiGraph): Directed graph.
        B (int): Number of buses.
        lazy_service_time (bool): If True, the service time data of a pair is computed on first access.

    Returns:
        Tuple[Dict[Any, Bus], BuildBusesStorage]: Buses and storage.

    Notes:
        - storage is also used by delete_node and add_node.
    """
    buses, uncovered_pairs, storage = build_buses(G, B)
    if uncovered_pairs:
        extend_buses(buses, normalize_uncovered_pairs(uncovered_pairs), storage)
    buses = {
        bid: Bus(bus_id=b.id, arc_list=list(b.ArcList), full_init=True, t_b=b.t_b,
                 lazy_service_time=lazy_service_time)
        for bid, b in buses.items()
    }
    return buses, storage