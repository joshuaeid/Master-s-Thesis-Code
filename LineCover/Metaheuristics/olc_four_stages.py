from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import random
import networkx as nx
from LineCover.SystemKit import BusSystem
from LineCover.InitialHeuristic.initial_heuristic import build_initial_solution
from LineCover.InitialHeuristic.build_buses_storage import BuildBusesStorage
from LineCover.RouteHeuristic.ils import ils
from LineCover.PhaseHarmonisationHeuristic.ph_heuristic import multistart_phase_harmonisation

"""
Four stage metaheuristic: initial solution, phase harmonisation, route ILS, phase harmonisation.
"""

STAGES = ("initial_solution", "phase_harmonisation_1", "route_ils", "phase_harmonisation_2")

@dataclass(frozen=True)
class OLCFourStagesResult:
    """
    Outcome of one run of the four-stage metaheuristic.

    Attributes:
        system (BusSystem): Final system.
        storage (BuildBusesStorage): Storage of the initial solution.
        values (Dict[str, float]): Ensemble average after each stage.
    """
    system: BusSystem
    storage: BuildBusesStorage
    values: Dict[str, float]

    @property
    def ensemble_average(self) -> float:
        """
        Returns:
            float: Ensemble average of the final system.
        """
        return self.values[STAGES[-1]]

    @property
    def relative_improvement(self) -> float:
        """
        Returns:
            float: (initial - final) / initial.
        """
        start = self.values[STAGES[0]]
        return (start - self.ensemble_average) / start

def _phase_harmonisation(system: BusSystem, value: float, restarts: int,
                         max_hops: int) -> Tuple[BusSystem, float]:
    """
    Phase harmonisation stage that keeps the input if it is better.

    Args:
        system (BusSystem): System with fixed routes.
        value (float): Its ensemble average.
        restarts (int): Starts per component.
        max_hops (int): Maximal number of hops per run.

    Returns:
        Tuple[BusSystem, float]: Better system and its ensemble average.

    Notes:
        - The starts are random, so the harmonised system can be worse than the input.
    """
    harmonised, harmonised_value = multistart_phase_harmonisation(
        system, restarts=restarts, max_hops=max_hops)

    if harmonised_value < value:
        return harmonised, float(harmonised_value)
    return system, value

def olc_four_stages(G: nx.DiGraph, N: int, max_evaluations: int, s: int = 20,
                    ph_restarts: int = 4, ph_max_hops: int = 200,
                    ls_L_max: int = 3, pert_L_max: int = 3,
                    lambda_: float = 1.0, seed: Optional[int] = None) -> OLCFourStagesResult:
    """
    Run the four stages IS -> PH -> RouteILS -> PH.

    Args:
        G (nx.DiGraph): Symmetric directed graph.
        N (int): Number of buses.
        max_evaluations (int): Budget of the route search in objective evaluations.
        s (int): Failed ILS iterations in a row before k is increased.
        ph_restarts (int): Starts per component in each phase harmonisation.
        ph_max_hops (int): Maximal number of hops per run.
        ls_L_max (int): Largest segment length in the local search.
        pert_L_max (int): Largest segment length in the perturbation.
        lambda_ (float): Request rate.
        seed (Optional[int]): Seed for the whole run.

    Returns:
        OLCFourStagesResult: Final system, storage and ensemble average after every stage.

    Raises:
        ValueError: If max_evaluations is None.

    Notes:
        - The ensemble average does not increase over the stages.
    """
    if max_evaluations is None:
        raise ValueError("max_evaluations must be given, the route search has no other stopping criterion.")

    if seed is not None:
        random.seed(seed)

    values: Dict[str, float] = {}

    # Stage 1: initial solution, N circuits covering every node pair.
    buses, storage = build_initial_solution(G, N)
    system = BusSystem(buses=buses, V=set(G.nodes()), lambda_=lambda_)
    system.init_pair_means()
    values[STAGES[0]] = float(system.ensemble_average())

    # Stage 2: phase harmonisation of the constructed routes.
    system, value = _phase_harmonisation(system, values[STAGES[0]], ph_restarts, ph_max_hops)
    values[STAGES[1]] = value

    # Stage 3: route ILS at fixed shifts.
    system = ils(G, N, s=s, max_evaluations=max_evaluations, system=system, storage=storage,
                 ls_L_max=ls_L_max, pert_L_max=pert_L_max)
    values[STAGES[2]] = float(system.ensemble_average())

    # Stage 4: phase harmonisation of the improved routes.
    system, value = _phase_harmonisation(system, values[STAGES[2]], ph_restarts, ph_max_hops)
    values[STAGES[3]] = value

    return OLCFourStagesResult(system=system, storage=storage, values=values)