from typing import Optional, Set
from collections import deque
import networkx as nx
from LineCover.SystemKit import Bus
from LineCover.InitialHeuristic.shortest_path_sampling import (
    BFS_with_predecessors_and_weights,
    brw_linear_sample,
)

def _seg_positions(start_pos: int, end_pos: int, P_b: int) -> set:
    """
    Set of ArcList indices that belong to the segment [start_pos..end_pos] (mod P_b).

    Args:
        start_pos (int): Index of the segment start.
        end_pos (int): Index of the segment end.
        P_b: Length of the bus.

    Returns:
        (set): Set of arc indices of the segment.
    """
    if start_pos <= end_pos:
        return set(range(start_pos, end_pos + 1))
    return set(range(start_pos, P_b)) | set(range(0, end_pos + 1))

def _splice(arc_list: list, start_pos: int, end_pos: int, new_path_arcs: list) -> list:
    """
    Replace the segment [start_pos..end_pos] in arc_list with new_path_arcs.

    Args:
        arc_list (list): Bus encoded as a list of arcs.
        start_pos (int): Index of the segment start.
        end_pos (int): Index of the segment end.
        new_path_arcs (list): Segment of arcs that replaces the removed segment.

    Returns:
        (list): Bus encoded as a new list of arcs.

    """
    if start_pos <= end_pos:
        return list(arc_list[:start_pos]) + new_path_arcs + list(arc_list[end_pos + 1:])
    else:
        # Wrap-around: surviving arcs sit between end_pos+1 and start_pos-1.
        return new_path_arcs + list(arc_list[end_pos + 1:start_pos])

def _bus_service_proxy(bus: Bus) -> float:
    """
    Single-bus ensemble-average proxy (bus treated as sole coverage source).

    Args:
        bus (Bus): Bus.
    Notes:
        - Used to rank candidate reconnection paths when n_samples > 1.
        - Equivalent to BusSystem.ensemble_average() for a one-bus system.
    """
    n = len(bus.NodeSet)
    num_pairs = n * (n - 1)
    total = 0.0
    for u, v in bus.covered_pairs():
        bus.ensure_pair(u, v)
        ivs = bus.intervals[(u, v)]
        cs = bus.constants[(u, v)]
        for j in range(len(ivs) - 1):
            total += cs[j] * (ivs[j + 1] - ivs[j])
    total -= 0.5 * bus.P_b * bus.P_b * num_pairs
    return total / (bus.P_b * num_pairs)

def reroute_segment(bus: Bus, start_pos: int, end_pos: int,
                    G: nx.DiGraph, n_samples: int = 1, rng=None) -> Optional[Bus]:
    """
    Replaces the arc segment [start_pos..end_pos] (mod P_b) of the bus by a shortest path, so that the bus stays a circuit.

    Args:
        bus (Bus): Bus.
        start_pos (int): Index of the first arc to remove.
        end_pos (int): Index of the last arc to remove.
        G (nx.DiGraph): Directed graph.
        n_samples (int): Number of sampled paths. For n_samples > 1 the one with the lowest proxy is kept.
        rng: Not used, seed random externally.

    Returns:
        Optional[Bus]: The resulting bus, or None if no valid reroute exists.
    """
    P_b = bus.P_b
    a = bus.NodeList[(start_pos - 1 + P_b) % P_b]
    b = bus.NodeList[end_pos]

    seg_pos: Set[int] = _seg_positions(start_pos, end_pos, P_b)
    surviving = [bus.ArcList[i] for i in range(P_b) if i not in seg_pos]

    # Case 1: segment forms a closed sub-loop (a == b)
    if a == b:
        if not surviving:
            return None  # would empty the bus. reject
        return Bus(bus_id=bus.id, arc_list=surviving, full_init=True, t_b=bus.t_b,
                   lazy_service_time=bus.lazy_service_time)

    # Case 2: attempt an edge-disjoint reconnection from a to b
    G_prime = G.copy()
    for arc in surviving:
        if G_prime.has_edge(*arc):
            G_prime.remove_edge(*arc)

    dist, pred, sigma = BFS_with_predecessors_and_weights(G_prime, a, t=b)

    # Sample candidate paths and keep the best
    best_arc_list = None
    best_score = float('inf')

    for _ in range(n_samples):
        path = list(brw_linear_sample(a, b, pred, sigma, path=deque([])))
        new_path_arcs = list(nx.utils.pairwise(path))
        candidate = _splice(bus.ArcList, start_pos, end_pos, new_path_arcs)

        if n_samples == 1:
            best_arc_list = candidate
            break

        current_bus = Bus(bus_id=bus.id, arc_list=candidate, full_init=True, t_b=bus.t_b,
                           lazy_service_time=bus.lazy_service_time)
        score = _bus_service_proxy(current_bus)
        if score < best_score:
            best_score = score
            best_arc_list = candidate

    return Bus(bus_id=bus.id, arc_list=best_arc_list, full_init=True, t_b=bus.t_b,
               lazy_service_time=bus.lazy_service_time)