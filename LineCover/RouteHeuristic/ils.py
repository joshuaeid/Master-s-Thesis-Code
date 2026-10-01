from typing import Optional
import random
import networkx as nx
from LineCover.SystemKit import BusSystem
from LineCover.InitialHeuristic.initial_heuristic import build_initial_solution
from LineCover.InitialHeuristic.build_buses_storage import BuildBusesStorage
from LineCover.RouteHeuristic.local_search import local_search
from LineCover.RouteHeuristic.perturbation import perturb
from LineCover.RouteHeuristic.budget import Budget

def _initial_system(G: nx.DiGraph, B: int):
    buses, storage = build_initial_solution(G, B)
    system = BusSystem(buses=buses, V=set(G.nodes()))
    system.init_pair_means()
    return system, storage

def ils(G: nx.DiGraph, B: int, s: int = 20, max_evaluations: Optional[int] = None,
       seed: Optional[int] = None, system: Optional[BusSystem] = None,
       storage: Optional[BuildBusesStorage] = None,
       ls_L_max: int = 3, pert_L_max: int = 3) -> BusSystem:
    """
    Iterated Local Search with perturbation, local search and better-only acceptance.

    Args:
        G (nx.DiGraph): Directed graph.
        B (int): Number of buses.
        s (int): Failed iterations in a row before k is increased.
        max_evaluations (Optional[int]): Budget in objective evaluations. None means unbounded.
        seed (Optional[int]): Seed for the whole run.
        system (Optional[BusSystem]): Start solution. If given, the InitialHeuristic is skipped.
        storage (Optional[BuildBusesStorage]): Storage of G, required if system is given.
        ls_L_max (int): Largest segment length in the local search.
        pert_L_max (int): Largest segment length in the perturbation.

    Returns:
        BusSystem: Best system found within the budget.

    Notes:
        - k starts at 2, increases by 1 after s failed iterations and resets to 2 on improvement.
    """
    if seed is not None:
        random.seed(seed)

    budget = Budget(max_evaluations)

    if system is None:
        system, storage = _initial_system(G, B)
    elif storage is None:
        raise ValueError("storage is required when a start system is given.")
    elif not system.pair_means:
        system.init_pair_means()
    incumbent = local_search(system, G, storage, iteration=0, budget=budget, L_max=ls_L_max)
    incumbent_objective = incumbent.ensemble_average()

    k = 2
    fails = 0
    iteration = 0
    while not budget.exhausted:
        iteration += 1
        # Work on a clone since perturb and local_search change the system in place.
        perturbed = perturb(incumbent.clone(), G, storage, iteration, k, L_max=pert_L_max)
        candidate = local_search(perturbed, G, storage, iteration, budget=budget, L_max=ls_L_max)
        candidate_objective = candidate.ensemble_average()
        budget.spend()

        if candidate_objective < incumbent_objective:
            incumbent = candidate
            incumbent_objective = candidate_objective
            k = 2
            fails = 0
        else:
            fails += 1
            if fails >= s:
                k += 1
                fails = 0

    return incumbent