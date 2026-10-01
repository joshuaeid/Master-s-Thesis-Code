from typing import Any, Optional
import random
import networkx as nx
from LineCover.SystemKit import Bus, BusSystem
from LineCover.InitialHeuristic.build_buses_storage import BuildBusesStorage
from LineCover.InitialHeuristic.extend_buses import extend_buses
from LineCover.RouteHeuristic.seed import move_seed
from LineCover.RouteHeuristic.evaluate import commit_move
from LineCover.RouteHeuristic.local_search import (
    _L_DELETE, _L_ADD, _apply_reroute, _apply_delete, _apply_add, _fresh_bus,
)

# Values of L in the seed for selection and repair.
_L_SELECT = 999
_L_REPAIR = 777

_KINDS = ("reroute", "delete", "add")

def _selection_seed(iteration: int, j: int) -> int:
    return move_seed(iteration, 0, j, _L_SELECT)

def _random_candidate(system: BusSystem, G: nx.DiGraph, storage: BuildBusesStorage,
                      iteration: int, j: int, L_max: int = 3) -> Optional[Any]:
    """Random move (bus_id, new_bus), or None if the move is not applicable."""
    rng = random.Random(_selection_seed(iteration, j))
    bus_id = rng.choice(list(system.buses.keys()))
    bus = system.buses[bus_id]
    if bus.P_b < 2:
        return None

    kind = rng.choice(_KINDS)

    if kind == "reroute":
        position = rng.randrange(bus.P_b)
        L = rng.randint(2, L_max)
        seed = move_seed(iteration, bus_id, position, L)
        new_bus = _apply_reroute(bus, G, position, L, seed)
    elif kind == "delete":
        if bus.P_b < 3:
            return None
        position = rng.randrange(bus.P_b)
        seed = move_seed(iteration, bus_id, position, _L_DELETE)
        new_bus = _apply_delete(bus, storage, position, seed)
    else:
        candidates = [n for n in G.nodes() if n not in bus.NodeSet]
        if not candidates:
            return None
        u = rng.choice(candidates)
        seed = move_seed(iteration, bus_id, u, _L_ADD)
        new_bus = _apply_add(bus, storage, u, seed)

    if new_bus is None:
        return None
    return bus_id, new_bus

def _repair(system: BusSystem, G: nx.DiGraph, storage: BuildBusesStorage, iteration: int) -> BusSystem:
    """
    One ExtendBuses pass over the pairs left uncovered by the perturbation.

    Notes:
        - Bus.extend only updates the attributes of the InitialHeuristic, so changed buses are rebuilt with _fresh_bus.
    """
    covered = set()
    for bus in system.buses.values():
        covered.update(bus.covered_pairs())
    uncovered = [pair for pair in system.node_pairs() if pair not in covered]
    if not uncovered:
        return system
    normalized = {(u, v) if u <= v else (v, u) for (u, v) in uncovered}

    before_arcs = {bid: list(b.ArcList) for bid, b in system.buses.items()}
    # Copy, since extend_buses changes NodeSet in place.
    before_nodes = {bid: set(b.NodeSet) for bid, b in system.buses.items()}

    random.seed(move_seed(iteration, 0, 0, _L_REPAIR))
    extend_buses(system.buses, normalized, storage)

    affected_pairs = set()
    for bid, bus in list(system.buses.items()):
        if list(bus.ArcList) != before_arcs[bid]:
            fresh = _fresh_bus(bus)
            affected_pairs.update((u, v) for u in before_nodes[bid]
                                  for v in before_nodes[bid] if u != v)
            affected_pairs.update(fresh.covered_pairs())
            system.buses[bid] = fresh

    system.update_pair_means(affected_pairs)
    return system

def perturb(system: BusSystem, G: nx.DiGraph, storage: BuildBusesStorage,
           iteration: int, k: int, L_max: int = 3) -> BusSystem:
    """
    Applies k random moves from all three neighborhoods and repairs the coverage with one ExtendBuses pass.

    Args:
        system (BusSystem): Current system.
        G (nx.DiGraph): Directed graph.
        storage (BuildBusesStorage): Storage with predecessors, sigma and distances.
        iteration (int): ILS iteration, part of the seed of every move.
        k (int): Number of random moves.
        L_max (int): Largest segment length for reroute_segment, L is drawn uniformly from 2, ..., L_max.

    Returns:
        BusSystem: The perturbed and repaired system.
    """
    for j in range(1, k + 1):
        candidate = _random_candidate(system, G, storage, iteration, j, L_max=L_max)
        if candidate is None:
            continue
        bus_id, new_bus = candidate
        system = commit_move(system, bus_id, new_bus)

    return _repair(system, G, storage, iteration)