from typing import Any, Callable, Dict, Optional, Tuple
import json
import time
from dataclasses import dataclass
import random
import networkx as nx
from LineCover.SystemKit import Bus, BusSystem
from LineCover.SystemKit.buses_system import system_period
from LineCover.InitialHeuristic.initial_heuristic import build_initial_solution
from LineCover.InitialHeuristic.build_buses_storage import BuildBusesStorage
from LineCover.RouteHeuristic.ils import ils

"""
Metaheuristic for bigger graphs: Route ILS with Monte Carlo estimation.
"""

STAGES_MC = ("initial_solution", "route_ils")

def randomize_phases(buses: Dict[Any, Bus], rng: random.Random) -> Dict[Any, Bus]:
    """
    Draw the shift of every bus uniformly at random.

    Args:
        buses (Dict[Any, Bus]): Buses of a solution, left untouched.
        rng (random.Random): Local generator, so that the phases are reproducible independently of the search.

    Returns:
        Dict[Any, Bus]: New buses on the same routes with random shifts.
    """
    shifted: Dict[Any, Bus] = {}
    for bus_id, bus in buses.items():
        rotation = rng.randrange(bus.P_b)
        nodes = bus.NodeList[rotation:] + bus.NodeList[:rotation]
        t_b = 1.0 - rng.random()
        shifted[bus_id] = Bus(bus_id=bus.id, node_list=list(nodes), full_init=True,
                              t_b=t_b, lazy_service_time=bus.lazy_service_time)

    return shifted

def routes_of(system: BusSystem) -> Dict[Any, Tuple[list, float]]:
    """
    Routes and shifts of a system, as plain data.

    Args:
        system (BusSystem): System to read.

    Returns:
        Dict[Any, Tuple[list, float]]: Bus id -> (arc list, t_b), enough to rebuild the system.
    """
    return {bus_id: ([list(arc) for arc in bus.ArcList], bus.t_b) for bus_id, bus in system.buses.items()}

def trace_writer(path: str) -> Callable[[int, BusSystem], None]:
    """
    Callback for on_block that writes the routes of every block to a file.

    Args:
        path (str): Target file, one JSON object per line.

    Returns:
        Callable[[int, BusSystem], None]: Callback for the on_block argument of olc_route_mc.

    Notes:
        - Writing per block keeps the trace out of memory and survives an interrupted run.
    """
    open(path, "w").close()

    def write(spent: int, system: BusSystem) -> None:
        record = {"evaluations": spent, "search_estimate": float(system.ensemble_average()),
                  "routes": {str(bus_id): route for bus_id, route in routes_of(system).items()}}
        with open(path, "a") as file:
            file.write(json.dumps(record) + "\n")

    return write

def system_of_routes(G: nx.DiGraph, routes: Dict[Any, Any], lambda_: float = 1.0) -> BusSystem:
    """
    Rebuild a system from the routes of a trace record.

    Args:
        G (nx.DiGraph): Graph of the run.
        routes (Dict[Any, Any]): Bus id -> (arc list, t_b), as written by trace_writer.
        lambda_ (float): Request rate.

    Returns:
        BusSystem: System on these routes, without a sample.
    """
    buses = {int(bus_id): Bus(bus_id=int(bus_id), arc_list=[tuple(arc) for arc in arcs], full_init=True,
                              t_b=t_b, lazy_service_time=True)
             for bus_id, (arcs, t_b) in routes.items()}

    return BusSystem(buses=buses, V=set(G.nodes()), lambda_=lambda_, lazy_service_time=True)

def evaluate_trace(G: nx.DiGraph, path: str, eval_M: int, sample_seed: int = 1,
                   lambda_: float = 1.0, delta: float = 0.05) -> list:
    """
    Re-evaluate every block of a trace on one fresh sample.

    Args:
        G (nx.DiGraph): Graph of the run.
        path (str): Trace file written by trace_writer.
        eval_M (int): Size of the fresh sample.
        sample_seed (int): Seed of the fresh sample, must differ from the seed of the search sample.
        lambda_ (float): Request rate.
        delta (float): Error probability of the confidence intervals.

    Returns:
        list: One dict per block with evaluations, search_estimate, eval_estimate and eval_interval.

    Notes:
        - All blocks share one sample, so their estimates are comparable without a second source of noise.
    """
    with open(path) as file:
        records = [json.loads(line) for line in file if line.strip()]

    samples, T, trace = None, system_period(G), []
    for record in records:
        system = system_of_routes(G, record["routes"], lambda_)
        if samples is None:
            system.init_mc(eval_M, T, seed=sample_seed)
            samples = system._mc_samples
        else:
            system._mc_samples, system._mc_T = samples, T
        trace.append({"evaluations": record["evaluations"], "search_estimate": record["search_estimate"],
                      "eval_estimate": float(system.ensemble_average()),
                      "eval_interval": system.mc_confidence_interval(delta)})

    return trace

@dataclass(frozen=True)
class OLCRouteMCResult:
    """
    Outcome of one run of the Monte Carlo metaheuristic.

    Attributes:
        system (BusSystem): Final system with the evaluation sample.
        storage (BuildBusesStorage): Storage of the initial solution.
        search_initial_estimate (float): Estimate of the start solution on the search sample.
        search_final_estimate (float): Estimate of the final system on the search sample.
        eval_initial_estimate (float): Estimate of the start solution on the evaluation sample.
        eval_final_estimate (float): Estimate of the final system on the evaluation sample.
        eval_difference (float): Paired difference of the two estimates on the evaluation sample.
        eval_difference_interval (Tuple[float, float]): Confidence interval of eval_difference.
        eval_initial_interval (Tuple[float, float]): Confidence interval of eval_initial_estimate.
        eval_final_interval (Tuple[float, float]): Confidence interval of eval_final_estimate.
        M (int): Size of the search sample.
        eval_M (int): Size of the evaluation sample.
    """
    system: BusSystem
    storage: BuildBusesStorage
    search_initial_estimate: float
    search_final_estimate: float
    eval_initial_estimate: float
    eval_final_estimate: float
    eval_initial_interval: Tuple[float, float]
    eval_final_interval: Tuple[float, float]
    eval_difference: float
    eval_difference_interval: Tuple[float, float]
    M: int
    eval_M: int

    @property
    def search_improvement(self) -> float:
        """
        Returns:
            float: (initial - final) / initial on the search sample.
        """
        return (self.search_initial_estimate - self.search_final_estimate) / self.search_initial_estimate

    @property
    def eval_improvement(self) -> float:
        """
        Returns:
            float: (initial - final) / initial on the evaluation sample.
        """
        return (self.eval_initial_estimate - self.eval_final_estimate) / self.eval_initial_estimate

def olc_route_mc(G: nx.DiGraph, N: int, max_evaluations: int, M: int = 10000,
                 s: int = 20, ls_L_max: int = 3, pert_L_max: int = 3,
                 lambda_: float = 1.0, delta: float = 0.05,
                 seed: Optional[int] = None,
                 sample_seed: int = 0, phase_seed: int = 0,
                 random_phases: bool = True, eval_M: Optional[int] = None,
                 block: Optional[int] = None,
                 on_block: Optional[Callable[[int, BusSystem], None]] = None) -> OLCRouteMCResult:
    """
    Run the two stages IS -> RouteILS with a Monte Carlo objective.

    Args:
        G (nx.DiGraph): Symmetric directed graph.
        N (int): Number of buses.
        max_evaluations (int): Budget of the route search in objective evaluations.
        M (int): Monte Carlo sample size.
        s (int): Failed ILS iterations in a row before k is increased.
        ls_L_max (int): Largest segment length in the local search.
        pert_L_max (int): Largest segment length in the perturbation.
        lambda_ (float): Request rate.
        delta (float): Error probability of the confidence interval.
        seed (Optional[int]): Seed for the whole search.
        sample_seed (int): Seed of the search sample, the evaluation sample uses sample_seed + 1.
        phase_seed (int): Seed of the random shifts of the start solution.
        random_phases (bool): If True, the buses of the start solution get random shifts instead of the aligned t_b = 1 of the initial heuristic.
        eval_M (Optional[int]): Size of the evaluation sample, M if not given.
        block (Optional[int]): If given, the search runs in blocks of this many evaluations and reports after each one.
        on_block (Optional[Callable[[int, BusSystem], None]]): Called with the evaluations spent and the current system, once with 0 before the search and then after every block.

    Returns:
        OLCRouteMCResult: Final system together with the estimates.

    Raises:
        ValueError: If max_evaluations is None.
    """
    if max_evaluations is None:
        raise ValueError("max_evaluations must be given, the route search has no other stopping criterion.")

    if seed is not None:
        random.seed(seed)

    # Stage 1: initial solution.
    buses, storage = build_initial_solution(G, N, lazy_service_time=True)
    if random_phases:
        buses = randomize_phases(buses, random.Random(phase_seed))
    system = BusSystem(buses=buses, V=set(G.nodes()), lambda_=lambda_, lazy_service_time=True)
    system.init_mc(M, system_period(G), seed=sample_seed)
    search_initial_estimate = float(system.ensemble_average())

    # Routes of the start solution, to evaluate it on the evaluation sample later on.
    initial_routes = {bus_id: (list(bus.ArcList), bus.t_b) for bus_id, bus in buses.items()}

    # Stage 2: route ILS on the estimated objective.
    spent, start = 0, time.perf_counter()
    if on_block is not None:
        on_block(spent, system)
    while spent < max_evaluations:
        step = max_evaluations - spent if block is None else min(block, max_evaluations - spent)
        system = ils(G, N, s=s, max_evaluations=step, system=system, storage=storage, ls_L_max=ls_L_max, pert_L_max=pert_L_max)
        spent += step
        if block is not None:
            print(f"{spent}/{max_evaluations} evaluations, estimate {system.ensemble_average():.3f}, "f"{time.perf_counter() - start:.0f} s", flush=True)
        if on_block is not None:
            on_block(spent, system)
    search_final_estimate = float(system.ensemble_average())

    # Evaluation of both systems on one fresh sample, independent of the search.
    eval_M = M if eval_M is None else eval_M
    system.init_mc(eval_M, system_period(G), seed=sample_seed + 1)
    eval_final_estimate = float(system.ensemble_average())
    eval_final_interval = system.mc_confidence_interval(delta)

    initial_system = BusSystem(
        buses={bus_id: Bus(bus_id=bus_id, arc_list=arcs, full_init=True, t_b=t_b, lazy_service_time=True) for bus_id, (arcs, t_b) in initial_routes.items()},
            V=set(G.nodes()), lambda_=lambda_, lazy_service_time=True)
    initial_system._mc_samples = system._mc_samples
    initial_system._mc_T = system._mc_T
    eval_initial_estimate = float(initial_system.ensemble_average())
    eval_initial_interval = initial_system.mc_confidence_interval(delta)
    eval_difference, eval_difference_interval = system.mc_paired_difference(initial_system, delta)

    return OLCRouteMCResult(system=system, storage=storage,
                            search_initial_estimate=search_initial_estimate,
                            search_final_estimate=search_final_estimate,
                            eval_initial_estimate=eval_initial_estimate,
                            eval_final_estimate=eval_final_estimate,
                            eval_initial_interval=eval_initial_interval,
                            eval_final_interval=eval_final_interval,
                            eval_difference=eval_difference,
                            eval_difference_interval=eval_difference_interval,
                            M=M, eval_M=eval_M)