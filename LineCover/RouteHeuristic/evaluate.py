from typing import Any, Callable, Optional, Tuple
from LineCover.SystemKit import Bus, BusSystem

def affected_pairs(old_bus: Bus, new_bus: Bus) -> list:
    """
    Pairs whose set B_{u,v} can change by a move.

    Args:
        old_bus (Bus): Bus before the move.
        new_bus (Bus): Bus after the move.

    Returns:
        list: Pairs covered by the old or the new bus.
    """
    pairs = list(old_bus.covered_pairs())
    pairs.extend(pair for pair in new_bus.covered_pairs() if not old_bus.covers(*pair))

    return pairs

def is_feasible(buses_system: BusSystem, bus_id: Any, new_bus: Bus) -> bool:
    """
    Checks if all node pairs are still covered after replacing the bus with new_bus.

    Args:
        buses_system (BusSystem): Current feasible system.
        bus_id (Any): ID of the changed bus.
        new_bus (Bus): Candidate bus.

    Returns:
        bool: True iff all pairs are still covered.
    """
    old_bus = buses_system.buses[bus_id]
    lost_nodes = old_bus.NodeSet - new_bus.NodeSet
    if not lost_nodes:
        return True

    # It could be that other buses cover pairs that are not covered by new_bus.
    other_buses = [bus for other_id, bus in buses_system.buses.items() if other_id != bus_id]
    for u in lost_nodes:
        for v in old_bus.NodeSet:
            if u == v:
                continue
            for pair in ((u, v), (v, u)):
                if new_bus.covers(*pair):
                    continue
                if not any(other_bus.covers(*pair) for other_bus in other_buses):
                    return False

    return True

def evaluate_move(buses_system: BusSystem, bus_id: Any, new_bus: Bus) -> float:
    """
    Ensemble average after replacing the bus with new_bus, without applying the change.

    Args:
        buses_system (BusSystem): Current system.
        bus_id (Any): ID of the changed bus.
        new_bus (Bus): Candidate bus.

    Returns:
        float: Ensemble average of the candidate.

    Notes:
        - new_bus must be feasible.
    """
    if buses_system.mc_enabled:
        return buses_system.mc_swap_bus(bus_id, new_bus)

    old_bus = buses_system.buses[bus_id]
    pairs = affected_pairs(old_bus, new_bus)
    old_pair_means = {p: buses_system.pair_means[p] for p in pairs}
    old_sum = buses_system._pair_mean_sum

    buses_system.buses[bus_id] = new_bus
    buses_system.update_pair_means(pairs)
    objective = buses_system.ensemble_average()

    buses_system.buses[bus_id] = old_bus
    buses_system.pair_means.update(old_pair_means)
    buses_system._pair_mean_sum = old_sum

    return objective

def commit_move(buses_system: BusSystem, bus_id: Any, new_bus: Bus) -> BusSystem:
    """
    Replaces the bus with new_bus and updates the means of the affected pairs.

    Args:
        buses_system (BusSystem): Current system, changed in place.
        bus_id (Any): ID of the changed bus.
        new_bus (Bus): Candidate bus.

    Returns:
        BusSystem: buses_system itself.

    Notes:
        - In Monte Carlo mode the estimate is recomputed on the next read.
    """
    if buses_system.mc_enabled:
        buses_system.mc_swap_bus(bus_id, new_bus, apply=True)
        buses_system.buses[bus_id] = new_bus
        return buses_system

    old_bus = buses_system.buses[bus_id]
    pairs = affected_pairs(old_bus, new_bus)
    buses_system.buses[bus_id] = new_bus
    try:
        buses_system.update_pair_means(pairs)
    except ValueError:
        # A pair is not covered anymore. This happens in the perturbation before the repair.
        buses_system._pair_mean_sum = None
    return buses_system

def try_move(buses_system: BusSystem, bus_id: Any, new_bus: Optional[Bus], current_objective: float,
             budget=None,
             accept: Callable[[float, float], bool] = lambda old, new: new < old) -> Tuple[bool, BusSystem, float]:
    """
    Evaluates a candidate and applies it if accept returns True.

    Args:
        buses_system (BusSystem): Current system.
        bus_id (Any): ID of the changed bus.
        new_bus (Optional[Bus]): Candidate bus, or None if the move did nothing.
        current_objective (float): Current ensemble average.
        budget (Optional[Budget]): Spends one unit per evaluated candidate.
        accept (Callable[[float, float], bool]): Acceptance criterion, strict improvement by default.

    Returns:
        Tuple[bool, BusSystem, float]: Accepted, system and objective.

    Notes:
        - None or infeasible candidates are rejected without spending budget.
    """
    if new_bus is None or not is_feasible(buses_system, bus_id, new_bus):
        return False, buses_system, current_objective

    objective = evaluate_move(buses_system, bus_id, new_bus)
    if budget is not None:
        budget.spend()
    if accept(current_objective, objective):
        system = commit_move(buses_system, bus_id, new_bus)
        return True, system, objective
    return False, buses_system, current_objective