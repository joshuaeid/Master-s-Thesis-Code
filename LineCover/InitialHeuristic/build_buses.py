from typing import Any, Dict, Tuple
from collections import deque
import networkx as nx
from LineCover.InitialHeuristic.shortest_path_sampling import brw_linear_sample
from LineCover.InitialHeuristic.build_buses_storage import BuildBusesStorage
from LineCover.SystemKit import Bus

def build_buses(G: nx.DiGraph, B: int) -> Tuple[Dict[Any, Bus], Dict[int, set], BuildBusesStorage]:
    """
    Construct a set of buses by sampling shortest paths in a graph.

    Args:
        G (nx.DiGraph): Directed graph.
        B (int): Number of buses to construct.

    Returns:
        Tuple containing:
            - buses (Dict[Any, Bus]): Mapping bus IDs to Bus objects.
            - uncovered_pairs (Dict[int, set]): Remaining uncovered node pairs grouped by distance.
            - storage (BuildBusesStorage): Storage object used during construction.
    """
    # Initialization.
    storage = BuildBusesStorage(G)
    predecessors = storage.predecessors
    sigma = storage.sigma
    buses: Dict[Any, Bus] = dict()
    all_covered = False

    # Construction of B buses.
    for b in range(1, B + 1):
        '''
            Bus looks like [(u)...v...u]
        '''
        # Choose randomly a non-covered node pair of maximal distance.
        u, v = storage.random_max_pair()

        if u is None and v is None:
            # Reset in order to be able to construct further buses.
            storage.reset()

            # Choose randomly a node pair of maximal distance after reset.
            u, v = storage.random_max_pair()

            # All node pairs have been covered.
            all_covered = True
            uncovered_pairs = {}

        # Sample uniformly a shortest path v -> u.
        path = brw_linear_sample(v, u, predecessors[v], sigma[v], path=deque([]))
        path.popleft()

        # Sample uniformly a shortest path u -> v and concatenate such that we have a path u -> v -> u.
        path = brw_linear_sample(u, v, predecessors[u], sigma[u], path=path)

        # Create a Bus class
        path.popleft()
        bus = Bus(bus_id=b, node_list=list(path))

        # Compute and remove covered pairs.
        covered_pairs = bus.covered_pairs()
        storage.remove_pairs(covered_pairs)

        # Store the bus
        buses[bus.id] = bus

    # Get uncovered node pairs in order to extend buses in a next step.
    if not all_covered:
        uncovered_pairs = {distance: set(pairs) for distance, pairs in storage.remaining_pairs.items()}

    # Reset in order to be able to construct a new set of buses.
    storage.reset()

    return buses, uncovered_pairs, storage