from typing import Any, List, Optional, Tuple
import random
import networkx as nx
from LineCover.SystemKit import Bus, BusSystem
from LineCover.InitialHeuristic.build_buses_storage import BuildBusesStorage
from LineCover.RouteHeuristic.reroute_segment import reroute_segment
from LineCover.RouteHeuristic.node_deletion import delete_node
from LineCover.RouteHeuristic.add_node import add_node
from LineCover.RouteHeuristic.seed import move_seed
from LineCover.RouteHeuristic.evaluate import try_move

# Values of L in the seed for delete_node and add_node.
_L_DELETE = 0
_L_ADD = 1

def _fresh_bus(b: Bus) -> Bus:
    return Bus(bus_id=b.id, arc_list=list(b.ArcList), full_init=True, t_b=b.t_b,
               lazy_service_time=b.lazy_service_time)

def _shuffled(items: List, seed: int) -> List:
    """Shuffled copy with a local Random, so the global random stays untouched."""
    items = list(items)
    random.Random(seed).shuffle(items)
    return items

def _apply_reroute(bus: Bus, G: nx.DiGraph, position: int, L: int, seed: int) -> Optional[Bus]:
    P_b = bus.P_b
    end_pos = (position + L - 1) % P_b
    random.seed(seed)
    trial = _fresh_bus(bus)
    result = reroute_segment(trial, position, end_pos, G)
    return None if result is None or list(result.ArcList) == list(bus.ArcList) else result

def _apply_delete(bus: Bus, storage: BuildBusesStorage, position: int, seed: int) -> Optional[Bus]:
    random.seed(seed)
    trial = _fresh_bus(bus)
    result = delete_node(trial, position, storage)
    return None if result is None or list(result.ArcList) == list(bus.ArcList) else result

def _apply_add(bus: Bus, storage: BuildBusesStorage, u, seed: int) -> Optional[Bus]:
    random.seed(seed)
    trial = _fresh_bus(bus)
    result = add_node(trial, u, storage)
    return None if result is None or list(result.ArcList) == list(bus.ArcList) else result

def _scan_neighborhood(system: BusSystem, iteration: int, L: int,
                       bus_slots, apply_fn, budget=None) -> Tuple[BusSystem, bool]:
    """
    First improvement scan over one neighborhood with fixed L.

    Args:
        system (BusSystem): Current system.
        iteration (int): ILS iteration, part of the seed.
        L (int): Segment length, part of the seed.
        bus_slots: Candidates (bus_id, slot). slot is a position for reroute_segment and delete_node and a node for add_node.
        apply_fn: (bus, slot, seed) -> Optional[Bus]. One of the operation functions.

    Returns:
        Tuple[BusSystem, bool]: System and whether an improving move was applied.
    """
    candidates = _shuffled(list(bus_slots), seed=move_seed(iteration, 0, 0, L))
    current_objective = system.ensemble_average()

    for bus_id, slot in candidates:
        if budget is not None and budget.exhausted:
            break
        bus = system.buses[bus_id]
        seed = move_seed(iteration, bus_id, slot, L)
        new_bus = apply_fn(bus, slot, seed)
        accepted, system, _ = try_move(system, bus_id, new_bus, current_objective, budget=budget)
        if accepted:
            return system, True

    return system, False

def _neighborhood_reroute(system: BusSystem, G: nx.DiGraph, iteration: int, budget=None,
                          L_max: int = 3) -> Tuple[BusSystem, bool]:
    """
    The reroute_segment neighborhood for L = 2, ..., L_max. L is only scanned if L-1 found no improving move.

    Args:
        L_max (int): Largest segment length.
    """
    for L in range(2, L_max + 1):
        bus_positions = [
            (bus_id, position)
            for bus_id, bus in system.buses.items() if bus.P_b >= 2
            for position in range(bus.P_b)
        ]
        system, found = _scan_neighborhood(
            system, iteration, L, bus_positions,
            apply_fn=lambda bus, position, seed: _apply_reroute(bus, G, position, L, seed),
            budget=budget,
        )
        if found:
            return system, True
        if budget is not None and budget.exhausted:
            return system, False
    return system, False

def _neighborhood_delete(system: BusSystem, storage: BuildBusesStorage, iteration: int,
                         budget=None) -> Tuple[BusSystem, bool]:
    bus_positions = [
        (bus_id, position)
        for bus_id, bus in system.buses.items() if bus.P_b >= 3
        for position in range(bus.P_b)
    ]
    return _scan_neighborhood(
        system, iteration, _L_DELETE, bus_positions,
        apply_fn=lambda bus, position, seed: _apply_delete(bus, storage, position, seed),
        budget=budget,
    )

def _neighborhood_add(system: BusSystem, G: nx.DiGraph, storage: BuildBusesStorage,
                      iteration: int, budget=None) -> Tuple[BusSystem, bool]:
    all_nodes = list(G.nodes())
    bus_targets = [
        (bus_id, u)
        for bus_id, bus in system.buses.items()
        for u in all_nodes if u not in bus.NodeSet
    ]
    return _scan_neighborhood(
        system, iteration, _L_ADD, bus_targets, budget=budget,
        apply_fn=lambda bus, u, seed: _apply_add(bus, storage, u, seed),
    )

def local_search(system: BusSystem, G: nx.DiGraph, storage: BuildBusesStorage, iteration: int,
                 budget=None, L_max: int = 3) -> BusSystem:
    """
    First improvement over the neighborhoods reroute_segment, delete_node and add_node in this order.

    Args:
        system (BusSystem): Current system with initialized pair_means.
        G (nx.DiGraph): Directed graph.
        storage (BuildBusesStorage): Storage with predecessors, sigma and distances.
        iteration (int): ILS iteration, part of the seed of every move.
        budget (Optional[Budget]): Budget in evaluations. The search stops when it is exhausted.
        L_max (int): Largest segment length for reroute_segment.

    Returns:
        BusSystem: system itself, changed in place, once no improving move is left.
    """
    i = 1
    while i <= 3:
        if budget is not None and budget.exhausted:
            break
        if i == 1:
            system, found = _neighborhood_reroute(system, G, iteration, budget=budget, L_max=L_max)
        elif i == 2:
            system, found = _neighborhood_delete(system, storage, iteration, budget=budget)
        else:
            system, found = _neighborhood_add(system, G, storage, iteration, budget=budget)

        i = 1 if found else i + 1

    return system