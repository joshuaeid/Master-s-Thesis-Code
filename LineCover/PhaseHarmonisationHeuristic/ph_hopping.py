from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import random
import math
import numpy as np
from LineCover.SystemKit import Bus, BusSystem
from LineCover.PhaseHarmonisationHeuristic.ph_reduce_problem import PhaseComponent
from LineCover.PhaseHarmonisationHeuristic.ph_quadratic_form import (QuadraticForm, ODPairState, q_form_from_pair_state)
from LineCover.PhaseHarmonisationHeuristic.ph_arrangement import (Region, BusPairArrangement, build_coincidence_arrangements, seed_region, advance_region)

"""
Shift application and one step of the region-hopping walk for the heuristic.
"""

def apply_shift(bus: Bus, delta: float) -> Bus:
    """
    Same route shifted by delta.

    Args:
        bus (Bus): Bus to shift.
        delta (float): Shift in units of arcs, only delta mod P_b matters.

    Returns:
        Bus: New shifted bus.

    Notes:
        - The integer part of delta rotates NodeList, the fractional part in (0,1] goes into t_b.
    """
    # Rotating NodeList by r to the left leads to delta = -r - (t_b_new - bus.t_b).
    rotation = math.ceil(bus.t_b - delta) - 1
    t_b_new = bus.t_b - delta - rotation

    nodes = bus.NodeList
    shift = rotation % bus.P_b
    if shift:
        nodes = nodes[shift:] + nodes[:shift]

    return Bus(bus_id=bus.id, node_list=list(nodes), full_init=True, t_b=t_b_new,
               lazy_service_time=bus.lazy_service_time)

def shift_system(buses_system: BusSystem, deltas: Dict[int, float]) -> BusSystem:
    """
    Copy of a system with the given shifts applied.

    Args:
        buses_system (BusSystem): System to shift.
        deltas (Dict[int, float]): Bus id -> shift. Missing buses are not shifted.

    Returns:
        BusSystem: New system.
    """
    buses = {}
    for bus_id, bus in buses_system.buses.items():
        delta = deltas.get(bus_id, 0.0)
        buses[bus_id] = bus if delta == 0.0 else apply_shift(bus, delta)

    return BusSystem(buses=buses, V=buses_system.nodes, lambda_=buses_system.lambda_, lazy_service_time=buses_system.lazy_service_time)

def random_start(component: PhaseComponent, buses_system: BusSystem, arrangements: Dict[Tuple[int, int], BusPairArrangement], seed: int = 0,
                 max_attempts: int = 20) -> Dict[int, float]:
    """
    Random shift drawn uniformly from the box of all shifts.

    Args:
        component (PhaseComponent): Component to shift.
        buses_system (BusSystem): System with the periods P_b.
        arrangements (Dict[Tuple[int, int], BusPairArrangement]): Arrangement of the component.
        seed (int): Seed of the draw.
        max_attempts (int): Maximal number of draws.

    Returns:
        Dict[int, float]: Bus id -> shift, reference bus at 0.

    Raises:
        RuntimeError: If no draw is in the interior of a region.

    Notes:
        - A draw lies on a hyperplane with probability zero, the retry only covers this case.
    """
    others = [bus_id for bus_id in component.bus_ids
              if bus_id != component.reference_bus_id]
    shifts: Dict[int, float] = {component.reference_bus_id: 0.0}
    if not others:
        return shifts

    rng = random.Random(seed)
    for _ in range(max_attempts):
        shifts.update((bus_id, rng.uniform(0.0, buses_system.buses[bus_id].P_b))
                      for bus_id in others)
        region, _ = seed_region(component, arrangements, shifts)
        if region.margin > 0.0:
            return shifts

    raise RuntimeError(f"random_start: no interior point found in {max_attempts} draws (seed={seed}).")

@dataclass(frozen=True)
class HopState:
    """
    State of the region hopping.

    Attributes:
        component (PhaseComponent): Component.
        arrangements (Dict[Tuple[int, int], BusPairArrangement]): Arrangement of the component.
        index (Dict[int, int]): Bus id -> coordinate.
        deltas (Dict[int, float]): Total shift of every bus relative to buses_system.
        region (Region): Region containing deltas.
        indices (Dict[Tuple[int, int], int]): Bisection indices of the region.
        H (np.ndarray), h (np.ndarray), alpha (float): Aggregate quadratic form at deltas.
        optimum (Optional[np.ndarray]), value (Optional[float]): Minimiser and value of the region if already known.
    """
    component: PhaseComponent
    arrangements: Dict[Tuple[int, int], BusPairArrangement]
    index: Dict[int, int]
    deltas: Dict[int, float]
    region: Region
    indices: Dict[Tuple[int, int], int]
    H: np.ndarray
    h: np.ndarray
    alpha: float
    optimum: Optional[np.ndarray] = None
    value: Optional[float] = None

def aggregate_q_form(buses_system: BusSystem, component: PhaseComponent, delta: Dict[int, float], index: Dict[int, int]) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Aggregate quadratic form of the component at delta.

    Args:
        buses_system (BusSystem): System with fixed routes at delta = 0.
        component (PhaseComponent): Component.
        delta (Dict[int, float]): Shift.
        index (Dict[int, int]): Bus id -> coordinate.

    Returns:
        Tuple[np.ndarray, np.ndarray, float]: H, h, alpha summed over all pairs with weight 1 / (n(n-1) T_uv).
    """
    shifted = shift_system(buses_system, delta)
    size = len(index)
    n = buses_system.n
    scale = 1.0 / (n * (n - 1))

    H = np.zeros((size, size), dtype=float)
    h = np.zeros(size, dtype=float)
    alpha = 0.0

    for (u, v) in component.pairs:
        starts: List[float] = []
        consts: List[float] = []
        minimizers: List[int] = []
        T_uv = None
        for t_start, t_end, c, bus_id in shifted.sweep_pair(u, v):
            starts.append(t_start)
            consts.append(c)
            minimizers.append(bus_id)
            T_uv = t_end

        state = ODPairState(starts=tuple(starts), consts=tuple(consts), minimizers=tuple(minimizers), T_uv=T_uv)
        Hp, hp, ap = q_form_from_pair_state(state, index)
        weight = scale / T_uv
        H += weight * Hp
        h += weight * hp
        alpha += weight * ap

    return H, h, alpha

def build_hop_state(buses_system: BusSystem, component: PhaseComponent, delta: Dict[int, float],
                    arrangements: Optional[Dict[Tuple[int, int], BusPairArrangement]] = None) -> HopState:
    """
    HopState at the start shift.

    Args:
        buses_system (BusSystem): System with fixed routes.
        component (PhaseComponent): Component.
        delta (Dict[int, float]): Bus id -> start shift in the interior of a region.
        arrangements (Optional[Dict[Tuple[int, int], BusPairArrangement]]): Arrangement.

    Returns:
        HopState: Start state for hop_once.

    Raises:
        ValueError: If delta does not lie in the interior of a region.
    """
    if arrangements is None:
        arrangements = build_coincidence_arrangements(buses_system, component)

    region, indices = seed_region(component, arrangements, delta)
    if region.margin <= 0.0:
        raise ValueError("build_hop_state: start does not lie in the interior of a region.")

    index = {bus_id: i for i, bus_id in enumerate(component.bus_ids)}
    H, h, alpha = aggregate_q_form(buses_system, component, delta, index)

    return HopState(component=component, arrangements=arrangements, index=index,
                    deltas=dict(delta), region=region, indices=indices,
                    H=H, h=h, alpha=alpha)

def solve_region_qp(form: QuadraticForm, region: Region, reference_bus_id: int, ridge: float = 1e-9) -> Tuple[np.ndarray, float, List[Tuple[int, int]]]:
    """
    Minimises the quadratic form over the region.

    Args:
        form (QuadraticForm): Form at the reference delta of the region.
        region (Region): Bounds delta_b - delta_b' <= u.
        reference_bus_id (int): Reference bus fixed to delta = 0.
        ridge (float): Small multiple of the identity added to H, since H is only positive semidefinite.

    Returns:
        Tuple[np.ndarray, float, List[Tuple[int, int]]]: Minimiser, value of the form and active facets.
    """
    import gurobipy as gp
    from gurobipy import GRB

    index = {bus_id: i for i, bus_id in enumerate(form.bus_ids)}
    size = len(form.bus_ids)
    keys = list(region.bounds)
    H = form.H + ridge * np.eye(size)

    model = gp.Model()
    model.Params.OutputFlag = 0
    model.Params.Threads = 1
    x = model.addMVar(size, lb=-GRB.INFINITY, ub=GRB.INFINITY)
    model.setObjective(0.5 * x @ H @ x + form.h @ x, GRB.MINIMIZE)
    model.addConstr(x[index[reference_bus_id]] == 0.0)
    rows = [model.addConstr(x[index[b]] - x[index[other]] <= region.bounds[(b, other)]) for (b, other) in keys]
    model.optimize()
    if model.Status != GRB.OPTIMAL:
        raise RuntimeError(f"QP not solved to optimality (status {model.Status}).")
    delta = np.array(x.X, dtype=float)
    active = [key for key, row in zip(keys, rows) if abs(row.Slack) < 1e-9]

    return delta, form.value(delta), active

def _crossing_point(region: Region, facet: Tuple[int, int], index: Dict[int, int],
                    size: int, step: float) -> np.ndarray:
    """
    Point just beyond a facet.

    Args:
        region (Region): Current region with delta = 0 inside.
        facet (Tuple[int, int]): Pair (b, b') whose bound is crossed.
        index (Dict[int, int]): Bus id -> coordinate.
        size (int): Number of coordinates.
        step (float): Distance beyond the bound.

    Returns:
        np.ndarray: The point.

    Notes:
        - Moves symmetrically along e_b - e_b'. Other crossed bounds are handled by advance_region.
    """
    b, other = facet
    delta = np.zeros(size)
    reach = 0.5 * (region.bounds[facet] + step)
    delta[index[b]] = reach
    delta[index[other]] = -reach

    return delta

@dataclass(frozen=True)
class HopResult:
    """
    Outcome of one hop_once call.

    Attributes:
        next_state (Optional[HopState]): State after the hop, or None at a local minimum.
        value (float): Optimum of the current region as ensemble average.
        optimum (np.ndarray): Minimiser of the current region.
        solves (int): Number of solved QPs.
    """
    next_state: Optional[HopState]
    value: float
    optimum: np.ndarray
    solves: int

def hop_once(buses_system: BusSystem, state: HopState, rng: random.Random, constant: float, step: float = 1e-6) -> HopResult:
    """
    One step of the region hopping.

    Args:
        buses_system (BusSystem): System with fixed routes at delta = 0.
        state (HopState): Current state.
        rng (random.Random): Random facet order.
        constant (float): Contribution of the rigid pairs.
        step (float): Distance beyond a facet.

    Returns:
        HopResult: Next state or None, optimum value, minimiser and number of QPs.

    Notes:
        - Facets in random order, first improvement.
        - The QP of the current region is only solved if state does not have it yet.
    """
    if state.optimum is not None and state.value is not None:
        optimum, value = state.optimum, state.value
        solves = 0
    else:
        form = QuadraticForm(H=state.H, h=state.h, alpha=state.alpha, bus_ids=state.component.bus_ids)
        optimum, value, _ = solve_region_qp(form, state.region, state.component.reference_bus_id)
        solves = 1
    here = value + constant

    index = state.index
    size = len(state.component.bus_ids)
    facets = list(state.region.bounds)
    rng.shuffle(facets)

    for facet in facets:
        delta = _crossing_point(state.region, facet, index, size, step)
        moved = {bus_id: float(delta[i]) for bus_id, i in index.items() if delta[i] != 0.0}

        candidate_region, candidate_indices = advance_region(state.arrangements, state.region, state.indices, facet, moved, step)

        candidate_deltas = {b: state.deltas.get(b, 0.0) + moved.get(b, 0.0) for b in state.component.bus_ids}
        candidate_H, candidate_h, candidate_alpha = aggregate_q_form(buses_system, state.component, candidate_deltas, index)
        candidate_form = QuadraticForm(H=candidate_H, h=candidate_h, alpha=candidate_alpha, bus_ids=state.component.bus_ids)
        candidate_optimum, candidate_value, _ = solve_region_qp(candidate_form, candidate_region, state.component.reference_bus_id)
        solves += 1
        if candidate_value + constant < here:
            candidate_state = HopState(component=state.component, arrangements=state.arrangements,
                                       index=index, deltas=candidate_deltas,
                                       region=candidate_region, indices=candidate_indices,
                                       H=candidate_H, h=candidate_h, alpha=candidate_alpha,
                                       optimum=candidate_optimum, value=candidate_value)
            return HopResult(candidate_state, here, optimum, solves)

    return HopResult(None, here, optimum, solves)