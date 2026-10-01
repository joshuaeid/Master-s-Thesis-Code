from typing import Any, Dict, Set, Tuple, List
from collections import defaultdict
from sortedcontainers import SortedSet
import random

class ExtendBusesStorage:
    """
    Manage uncovered node pairs and efficiently select extensions for buses.

    Attributes:
        buses (Dict[Any, Any]): Mapping bus IDs to Bus objects.
        uncovered_pairs (Set[Tuple[Any, Any]]): Set of uncovered node pairs.
        pair_to_distance (Dict[Tuple[Any, Any], int]): Distance lookup for node pairs.
        cost_method (str): Method used to compute pair cost.
        node_to_uncovered_pairs (Dict[Any, Set[Tuple[Any, Any]]]): Maps nodes to related uncovered pairs.
        node_to_bus_dist (Dict[Any, Dict[Any, Tuple[int, Set[Any]]]]): Distance from node to each bus.
        buckets (Dict[int, Set[Tuple[Any, Any]]]): Maps cost to node pairs.
        pair_to_best (Dict[Tuple[Any, Any], Tuple[float, List[Any]]]): Best cost and the IDs of all buses attaining it for each pair.
        non_empty_buckets (Any): Sorted structure of non-empty buckets.
    """

    def __init__(
        self,
        uncovered_pairs: Set[Tuple[Any, Any]],
        pair_to_distance: Dict[Tuple[Any, Any], int],
        buses: Dict[Any, Any],
        cost_method: str = "max_metric",
    ) -> None:
        """
        Initialize storage for extending buses.

        Args:
            uncovered_pairs (Set[Tuple[Any, Any]]): Uncovered node pairs.
            pair_to_distance (Dict[Tuple[Any, Any], int]): Distance lookup.
            buses (Dict[Any, Any]): Mapping bus IDs to Bus objects.
            cost_method (str, optional): Cost function type. Defaults to "max_metric".
        """
        self.buses = buses
        self.uncovered_pairs = uncovered_pairs
        self.pair_to_distance = pair_to_distance
        self.cost_method = cost_method

        # Create a dictionary that maps a node (from some uncovered pair) to the respective uncovered node pairs.
        self.node_to_uncovered_pairs: Dict[Any, Set[Tuple[Any, Any]]] = defaultdict(set)
        for u, v in uncovered_pairs:
            self.node_to_uncovered_pairs[u].add((u, v))
            self.node_to_uncovered_pairs[v].add((u, v))

        # Create a dictionary that maps a node u (from some uncovered pair) to
        # a dictionary that maps every bus to a tuple of its (distance, corresponding set of bus nodes) to u.
        nodes = set(self.node_to_uncovered_pairs.keys())
        self.node_to_bus_dist: Dict[Any, Dict[Any, Tuple[int, Set[Any]]]] = {
            u: {bus_id: self._closest_nodes_to_bus(u, bus) for bus_id, bus in buses.items()}
            for u in nodes
        }

        # Initialize the cost buckets that map the cost to the set of node pairs
        self.buckets: Dict[int, Set[Tuple[Any, Any]]] = defaultdict(set)
        # Initialize a dictionary that maps a node pair to its current best cost and respective bus IDs.
        self.pair_to_best: Dict[Tuple[Any, Any], Tuple[float, List[Any]]] = {}
        # Initialize the SortedSet structure that maintains non-empty buckets.
        self.non_empty_buckets = SortedSet()

        # Fill the buckets.
        for u, v in self.uncovered_pairs:
            cost, best_bus_ids = self._pair_best_buses(u, v)
            bkt = int(cost)  # integer bucket
            self.buckets[bkt].add((u, v))
            self.pair_to_best[(u, v)] = (cost, best_bus_ids)
            self.non_empty_buckets.add(bkt)

    def _closest_nodes_to_bus(self, u: Any, bus: Any) -> Tuple[int, Set[Any]]:
        """
        Compute the closest nodes of a bus to a given node.

        Args:
            u (Any): Node.
            bus (Any): Bus object.

        Returns:
            Tuple[int, Set[Any]]: Minimum distance and set of bus nodes achieving it.

        Notes:
            - If u lies on the bus, d(u,u) = 0 since pair_to_distance has no entry for (u,u).
        """
        best_dist = float('inf')
        best_nodes = set()

        for w in bus.NodeList:
            d = 0 if w == u else self.pair_to_distance[(u, w)]
            if d < best_dist:
                best_dist = d
                best_nodes = {w}
            elif d == best_dist:
                best_nodes.add(w)

        return best_dist, best_nodes

    def _pair_best_buses(self, u: Any, v: Any) -> Tuple[float, List[Any]]:
        """
        Determine the best buses for a node pair based on cost.

        Args:
            u (Any): First node.
            v (Any): Second node.

        Returns:
            Tuple[float, List[Any]]: Best cost and the IDs of all buses attaining it.
        """
        # Initialize.
        best_cost = float('inf')
        best_bus_ids = []

        # Algorithm.
        for bus_id in self.buses:
            cost = self.pair_cost(u, v, bus_id)

            if cost < best_cost:
                best_cost = cost
                best_bus_ids = [bus_id]
            elif cost == best_cost:
                best_bus_ids.append(bus_id)

        return best_cost, best_bus_ids

    def pair_cost(self, u: Any, v: Any, bus_id: Any) -> float:
        """
        Compute the cost of assigning a node pair to a bus.

        Args:
            u (Any): First node.
            v (Any): Second node.
            bus_id (Any): Bus identifier.

        Returns:
            float: Computed cost.
        """
        cost_u, _ = self.node_to_bus_dist[u][bus_id]
        cost_v, _ = self.node_to_bus_dist[v][bus_id]

        if self.cost_method == "max_metric":
            cost = max(cost_u, cost_v)
        else:
            cost = cost_u + cost_v
        return cost

    def _sample_node_position(self, nodes: Set[Any], bus: Any) -> Tuple[Any, Any]:
        """
        Sample a node and position from a bus proportional to occurrences.

        Args:
            nodes (Set[Any]): Candidate nodes on the bus.
            bus (Any): Bus object.

        Returns:
            Tuple[Any, int]: Sampled node and its position.
        """
        # Total number of positions.
        total = sum(len(bus.node_to_positions[u]) for u in nodes)

        r = random.randrange(total)

        for u in nodes:
            positions = bus.node_to_positions[u]
            if r < len(positions):
                return u, random.choice(positions)
            r -= len(positions)

    def pop_closest_pair(self) -> Tuple[Any, Any, Any, Any, Tuple[Any, int]] | Tuple[None, None, None, None, None]:
        """
        Sample a node pair from the minimal cost bucket and return extension data.

        Returns:
            Tuple containing extension decision data, or all None if no pairs remain.
        """
        while self.non_empty_buckets:
            min_bucket = self.non_empty_buckets[0]
            pairs = self.buckets[min_bucket]

            # Sample node pair.
            u, v = random.choice(list(pairs))

            # Get best buses.
            cost, best_bus_ids = self.pair_to_best[(u, v)]

            # Sample one bus from best buses.
            bus_id = random.choice(best_bus_ids)
            bus = self.buses[bus_id]

            # Current distances and closest bus nodes.
            cost_u, nodes_u = self.node_to_bus_dist[u][bus_id]
            cost_v, nodes_v = self.node_to_bus_dist[v][bus_id]

            if cost_u < cost_v:
                if cost_u == 0:
                    single_node_extension = True
                    first_expand_node = v
                    second_expand_node = None
                    sampled_bus_node, sampled_bus_pos = self._sample_node_position(nodes_v, bus)
                else:
                    single_node_extension = False
                    first_expand_node = u
                    second_expand_node = v
                    sampled_bus_node, sampled_bus_pos = self._sample_node_position(nodes_u, bus)

            elif cost_v < cost_u:
                if cost_v == 0:
                    single_node_extension = True
                    first_expand_node = u
                    second_expand_node = None
                    sampled_bus_node, sampled_bus_pos = self._sample_node_position(nodes_u, bus)
                else:
                    single_node_extension = False
                    first_expand_node = v
                    second_expand_node = u
                    sampled_bus_node, sampled_bus_pos = self._sample_node_position(nodes_v, bus)

            else:
                pairs = [[u, nodes_u], [v, nodes_v]]
                random.shuffle(pairs)
                single_node_extension = False
                (first_expand_node, first_nodes), (second_expand_node, second_nodes) = pairs
                sampled_bus_node, sampled_bus_pos = self._sample_node_position(first_nodes, bus)

            return (
                single_node_extension,
                first_expand_node,
                second_expand_node,
                bus_id,
                (sampled_bus_node, sampled_bus_pos),
            )

        return None, None, None, None, None

    def pop_closest_data(self, bus: Any, v: Any) -> Tuple[Any, int]:
        """
        Sample a closest node on a bus to a given node.

        Args:
            bus (Any): Bus object.
            v (Any): Node.

        Returns:
            Tuple[Any, int]: Sampled node and its position on the bus.
        """
        cost_v, nodes_v = self.node_to_bus_dist[v][bus.id]
        sampled_v_node, sampled_v_pos = self._sample_node_position(nodes_v, bus)

        return sampled_v_node, sampled_v_pos

    def remove_covered_pair(self, u: Any, v: Any) -> None:
        """
        Remove a newly covered node pair from storage.

        Args:
            u (Any): First node.
            v (Any): Second node.
        """
        # Removal from sets and dictionaries.
        self.uncovered_pairs.remove((u, v))
        self.node_to_uncovered_pairs[u].discard((u, v))
        self.node_to_uncovered_pairs[v].discard((u, v))

        # Removal from buckets.
        bkt = int(self.pair_to_best[(u, v)][0])  # integer bucket
        self.buckets[bkt].discard((u, v))

    def update_after_extension(self, inserted_nodes: List[Any], bus: Any) -> None:
        """
        Update storage after extending a bus.

        Args:
            inserted_nodes (List[Any]): Nodes added to the bus.
            bus (Any): Updated Bus object.

        Notes:
            - Updates distances, costs, and bucket assignments.
            - Removes covered pairs and cleans empty buckets.
        """
        bus_set = bus.NodeSet
        bus_id = bus.id
        old_bkts = set()  # Non-empty buckets that get empty.
        covered_pairs = set()

        # Update distances of nodes in uncovered pairs.
        changed_nodes = set()
        for node, pairs in self.node_to_uncovered_pairs.items():
            if not pairs:
                continue
            dist, nodes = self.node_to_bus_dist[node][bus_id]
            original_dist = dist
            for w in inserted_nodes:
                d = 0 if node == w else self.pair_to_distance[(node, w)]
                if d < dist:
                    dist = d
                    nodes = {w}
                elif d == dist:
                    nodes.add(w)
            self.node_to_bus_dist[node][bus_id] = (dist, nodes)
            if dist < original_dist:
                changed_nodes.add(node)

        # Pairs with an inserted node or a node that got closer to the bus.
        affected_pairs = set()
        for w in inserted_nodes:
            affected_pairs.update(self.node_to_uncovered_pairs[w])
        for node in changed_nodes:
            affected_pairs.update(self.node_to_uncovered_pairs[node])

        for u, v in affected_pairs:
            # Remove node pair if covered.
            if u in bus_set and v in bus_set:
                covered_pairs.add((u, v))
                best_cost, _ = self.pair_to_best[(u, v)]
                old_bkt = int(best_cost)
                old_bkts.add(old_bkt)
            else:
                '''Buckets and pair_to_best'''
                new_cost = self.pair_cost(u, v, bus_id)
                best_cost, best_bus_ids = self.pair_to_best[(u, v)]

                if new_cost < best_cost:
                    self.pair_to_best[(u, v)] = (new_cost, [bus_id])

                    old_bkt = int(best_cost)
                    self.buckets[old_bkt].remove((u, v))
                    old_bkts.add(old_bkt)

                    new_bkt = int(new_cost)
                    self.buckets[new_bkt].add((u, v))
                    self.non_empty_buckets.add(new_bkt)

                elif new_cost == best_cost and bus_id not in best_bus_ids:
                    best_bus_ids.append(bus_id)

        # Remove covered pairs.
        for u, v in covered_pairs:
            self.remove_covered_pair(u, v)

        # Remove old bucket from non_empty_buckets if it gets empty.
        for old_bkt in old_bkts:
            if not self.buckets[old_bkt]:
                self.non_empty_buckets.discard(old_bkt)