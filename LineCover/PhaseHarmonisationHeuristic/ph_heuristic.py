from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import random
from LineCover.SystemKit import BusSystem
from LineCover.PhaseHarmonisationHeuristic.ph_reduce_problem import PhaseComponent, reduce_phase_problem
from LineCover.PhaseHarmonisationHeuristic.ph_arrangement import BusPairArrangement, build_coincidence_arrangements
from LineCover.PhaseHarmonisationHeuristic.ph_hopping import (
    build_hop_state, hop_once, shift_system, random_start,
)

"""
Region hopping with single start and multistart.
"""

@dataclass(frozen=True)
class HoppingResult:
    """
    Outcome of a region-hopping run on one component.

    Attributes:
        system (BusSystem): Best system found.
        value (float): Its ensemble average.
        start_value (float): Ensemble average of the input system.
        hops (int): Number of accepted hops.
        qp_solves (int): Number of solved QPs.
        stopped (str): "local_minimum" or "hop_budget".
    """
    system: BusSystem
    value: float
    start_value: float
    hops: int
    qp_solves: int
    stopped: str

    @property
    def relative_improvement(self) -> float:
        """
        Relative improvement over the input configuration.

        Returns:
            float: (start_value - value) / start_value.
        """
        return (self.start_value - self.value) / self.start_value

def region_hopping(buses_system: BusSystem, component: PhaseComponent, constant: float, start: Dict[int, float], seed: int = 0, max_hops: int = 200,
                   arrangements: Optional[Dict[Tuple[int, int], BusPairArrangement]] = None) -> HoppingResult:
    """
    Walk from region to region until no neighbour improves.

    Args:
        buses_system (BusSystem): System with fixed routes.
        component (PhaseComponent): Component to optimise.
        constant (float): Contribution of the rigid pairs.
        start (Dict[int, float]): Bus id -> start shift in the interior of a region.
        seed (int): Seed for the facet order.
        max_hops (int): Maximal number of hops.
        arrangements (Optional[Dict[Tuple[int, int], BusPairArrangement]]): Already built arrangement.

    Returns:
        HoppingResult: Best system found and statistics.

    Raises:
        ValueError: If start is not in the interior of a region.
    """
    rng = random.Random(seed)
    start_value = buses_system.ensemble_average()

    if arrangements is None:
        arrangements = build_coincidence_arrangements(buses_system, component)

    state = build_hop_state(buses_system, component, start, arrangements=arrangements)
    index = state.index

    best_value = float("inf")
    best_delta = dict(start)
    hops = 0
    solves = 0
    stopped = "local_minimum"

    while True:
        result = hop_once(buses_system, state, rng, constant)
        solves += result.solves
        if result.value < best_value:
            best_value = result.value
            best_delta = {b: state.deltas.get(b, 0.0) + result.optimum[i] for b, i in index.items()}
        if result.next_state is None:
            break
        state = result.next_state
        hops += 1
        if hops >= max_hops:
            stopped = "hop_budget"
            break

    best_system = shift_system(buses_system, best_delta)
    return HoppingResult(system=best_system, value=best_value, start_value=start_value, hops=hops, qp_solves=solves, stopped=stopped)

def multistart_phase_harmonisation(buses_system: BusSystem, restarts: int = 4, max_hops: int = 200
                   ) -> Tuple[BusSystem, float]:
    """
    Runs the region hopping from several random starts per component.

    Args:
        buses_system (BusSystem): System with fixed routes.
        restarts (int): Number of starts per component.
        max_hops (int): Maximal number of hops per run.

    Returns:
        Tuple[BusSystem, float]: Best system and its ensemble average.

    Notes:
        - The arrangement of a component is built once and used for all its starts.
    """
    reduction = reduce_phase_problem(buses_system)
    system = buses_system

    for component in reduction.components:
        if component.is_rigid:
            continue
        arrangements = build_coincidence_arrangements(system, component)

        best_system, best_value = system, float("inf")
        for seed in range(restarts):
            start = random_start(component, system, arrangements, seed=seed)
            result = region_hopping(system, component, reduction.constant or 0.0, seed=seed, max_hops=max_hops, start=start, arrangements=arrangements)
            if result.value < best_value:
                best_system, best_value = result.system, result.value
        system = best_system

    return system, system.ensemble_average()