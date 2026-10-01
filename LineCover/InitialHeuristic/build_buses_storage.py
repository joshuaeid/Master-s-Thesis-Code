from typing import Any, Dict, Set, Tuple, List
from collections import defaultdict
import random
import networkx as nx
from LineCover.InitialHeuristic.shortest_path_sampling import BFS_with_predecessors_and_weights

class BuildBusesStorage:
    """
    Build and manage storage of node pairs grouped by shortest path distances.

    Attributes:
        nodes (Any): Nodes of the graph G.
        distance_to_pairs (Dict[int, Set[Tuple[Any, Any]]]): Maps distance to node pairs.
        pair_to_distance (Dict[Tuple[Any, Any], int]): Maps node pairs to their distance.
        predecessors (Dict[Any, Dict[Any, List[Any]]]): Stores predecessor mappings per source node.
        sigma (Dict[Any, Dict[Any, int]]): Stores number of shortest paths per source node.
        distances (List[int]): Sorted list of distances in decreasing order.
        remaining_pairs (Dict[int, Set[Tuple[Any, Any]]]): Remaining node pairs grouped by distance.
        i (int): Index tracking the current maximum distance bucket.
    """

    def __init__(self, G: nx.DiGraph) -> None:
        """
        Initialize the storage structure from a graph.

        Args:
            G (nx.DiGraph): Directed graph.

        Notes:
            - Node pairs (u, v) and (v, u) are treated separately.
        """
        # Initialization
        self.nodes = G.nodes()
        self.distance_to_pairs: Dict[int, Set[Tuple[Any, Any]]] = defaultdict(set)
        self.pair_to_distance: Dict[Tuple[Any, Any], int] = dict()
        self.predecessors: Dict[Any, Dict[Any, List[Any]]] = dict()
        self.sigma: Dict[Any, Dict[Any, int]] = dict()

        # Build Storage.
        for source in self.nodes:

            # Store information about the successor graph.
            distances, self.predecessors[source], self.sigma[source] = BFS_with_predecessors_and_weights(G, source)

            # Store node pairs in distance buckets and build a reverse mapping.
            for target, distance in distances.items():
                if target != source:
                    self.distance_to_pairs[distance].add((source, target))
                    self.pair_to_distance[(source, target)] = distance

        # Sort node pairs by their distance
        self.distances: List[int] = sorted(self.distance_to_pairs.keys(), reverse=True)

        # Copy the buckets.
        self.reset()

    def reset(self) -> None:
        """
        Reset the remaining pairs to the initial full set.

        Notes:
            - Copies distance_to_pairs into remaining_pairs.
            - Resets the index tracking the current maximum distance bucket.
        """
        self.remaining_pairs: Dict[int, Set[Tuple[Any, Any]]] = {
            distance: NodePairSet.copy()
            for distance, NodePairSet in self.distance_to_pairs.items()
        }
        self.i = 0

    def random_max_pair(self) -> Tuple[Any, Any] | None:
        """
        Select a random node pair among the currently farthest pairs.

        Returns:
            Tuple[Any, Any] | None: A randomly selected node pair, or None if no pairs remain.

        Notes:
            - Updates the counter i to skip empty distance buckets.
        """
        while self.i < len(self.distances):
            distance = self.distances[self.i]
            bucket = self.remaining_pairs[distance]

            if not bucket:
                self.i += 1
                continue

            source, target = random.choice(list(bucket))
            bucket.discard((source, target))

            return source, target

        return None, None

    def remove_pairs(self, pairs_to_remove: Set[Tuple[Any, Any]]) -> None:
        """
        Remove a set of node pairs from the remaining pairs.

        Args:
            pairs_to_remove (Set[Tuple[Any, Any]]): Node pairs to remove.

        Notes:
            - Removes each pair from its corresponding distance bucket.
        """
        for source, target in pairs_to_remove:
            distance = self.pair_to_distance.get((source, target))
            self.remaining_pairs[distance].discard((source, target))