from typing import Any, Iterator, List, Tuple, Optional, Set, Dict
from collections import defaultdict
import networkx as nx
import bisect

class Bus:
    """
    Represents a bus as a sequence of nodes and arcs.

    Attributes:
        id (Any): Unique identifier of the bus.
        NodeList (List[Any]): Ordered list of nodes representing the bus route.
        ArcList (List[Tuple[Any, Any]]): List of arcs corresponding to consecutive nodes.
        NodeSet (Set[Any]): Set of nodes covered by the bus.
        node_to_positions (default dict): Maps each node to a list of positions in NodeList.
        node_to_pred (default dict): Maps each node to its predecessor nodes.
        P_b (int): Number of nodes in NodeList.
        constants (Dict[Tuple[Any, Any], List[float]]): Maps each node pair to a list of constants (values of
         service time function) in the same order as the breakpoints.
        breakpoints (Dict[Tuple[Any, Any], List[float]]): Maps each node pair to a list of breakpoints (as positions
         on the bus) of the service time function.
        actual_breakpoints (Dict[Tuple[Any, Any], List[float]]): Maps each node pair to breakpoints, transformed
         by "+ 1 - t_b" into actual times (same order as breakpoints/constants).
        intervals (Dict[Tuple[Any, Any], List[float]]): Maps each node pair to a list of interval boundary
         points corresponding to the transformed breakpoints.
    """

    def __init__(self,
                 bus_id: Any, node_list: Optional[List[Any]] = None,
                 arc_list: Optional[List[Tuple[Any, Any]]] = None,
                 full_init: bool = False,
                 t_b: float = 1,
                 lazy_service_time: bool = False):
        """
        Initialize a Bus object.

        Args:
            bus_id (Any): Unique identifier for the bus.
            node_list (List[Any], optional): List of nodes representing the bus route.
            arc_list (List[Tuple[Any, Any]], optional): List of arcs representing the bus route.
            lazy_service_time (bool): If True, the service time data of a pair is computed on first access.

        Raises:
            ValueError: If neither node_list nor arc_list is provided.
        """

        ''' Initialize core data: NodeList and ArcList. '''
        if node_list is None and arc_list is None:
            raise ValueError("No bus data.")
        elif node_list is None:
            self.ArcList: List[Tuple[Any, Any]] = arc_list
            self.NodeList: List[Any] = [v for (u, v) in arc_list]  # Head of arc belongs to the arc
        elif arc_list is None:
            self.NodeList: List[Any] = node_list
            self.ArcList: List[Tuple[Any, Any]] = list(nx.utils.pairwise([node_list[-1]] + node_list))
        else:
            self.NodeList: List[Any] = node_list
            self.ArcList: List[Tuple[Any, Any]] = arc_list

        ''' Initialize attributes that are set in the future. '''
        # BusesSystem relevant.
        self.id = bus_id

        # Bus relevant.
        self.t_b: float = t_b
        self.lazy_service_time: bool = lazy_service_time

        # Service time attributes.
        self.constants: Dict[Tuple[Any, Any], List[float]] = {}
        self.breakpoints: Dict[Tuple[Any, Any], List[float]] = {}
        self.actual_breakpoints: Dict[Tuple[Any, Any], List[float]] = {}
        self.intervals: Dict[Tuple[Any, Any], List[float]] = {}

        # Intensify Heuristic.
        self.arc_to_position: Dict[Tuple[Any, Any], int] = {}

        ''' Initialize attributes derived from the bus for Initial Heuristic. '''
        self.P_b: int = len(self.NodeList)
        self.NodeSet: Set[Any] = set(self.NodeList)

        self.node_to_positions: Dict[Any, List[int]] = defaultdict(list)
        self.node_to_pred: Dict[Any, Set[Any]] = defaultdict(set)

        self.node_to_positions[self.NodeList[0]].append(0)
        self.node_to_pred[self.NodeList[0]].add(self.NodeList[-1])
        for i, node in enumerate(self.NodeList[1:], start=1):
            self.node_to_positions[node].append(i)
            self.node_to_pred[node].add(self.NodeList[i - 1])

        ''' Full initialization - all cases except in the Initial Heuristic. '''
        if full_init:
            # Service Time attributes.
            self.init_service_time(precompute=not lazy_service_time)

            # Intensify Heuristic attributes.
            self.init_intensify_heuristic()

    def covers(self, u: Any, v: Any) -> bool:
        """
        Whether the bus covers the pair (u,v).

        Args:
            u (Any): node of origin
            v (Any): node of destination

        Returns:
            bool: True iff u and v are distinct nodes of the bus.
        """
        return u != v and u in self.NodeSet and v in self.NodeSet

    def covered_pairs(self) -> Iterator[Tuple[Any, Any]]:
        """
        Iterate over the pairs the bus covers.

        Yields:
            Tuple[Any, Any]: Every ordered pair of distinct nodes of the bus.
        """
        for u in self.NodeSet:
            for v in self.NodeSet:
                if u != v:
                    yield (u, v)

    def number_of_covered_pairs(self) -> int:
        """
        Returns:
            int: |NodeSet| * (|NodeSet| - 1), the number of covered pairs.
        """
        n = len(self.NodeSet)
        return n * (n - 1)

    def _update_attributes_initial_heuristic(self, nodes: List[Any], start_pos: int):
        """
        Update derived attributes of the bus.

        Args:
            nodes (List[Any], optional): Nodes to update. If None, updates all nodes.
            start_pos (int, optional): Index in NodeList where nodes are inserted before. Defaults to 0.

        Notes:
            - Updates NodeSet, node_to_positions, and node_to_pred.
        """
        prev_node = self.NodeList[start_pos - 1] if start_pos > 0 else self.NodeList[-1]
        shift = len(nodes)
        # Shift positions of nodes after the inserted segment.
        if start_pos + shift < self.P_b:
            next_node = self.NodeList[start_pos + shift]
            self.node_to_pred[next_node].remove(self.NodeList[start_pos - 1])
            affected_nodes = set(self.NodeList[start_pos + shift:])
            for node in affected_nodes:
                positions = self.node_to_positions[node]
                for i in range(len(positions)):
                    if positions[i] >= start_pos:
                        positions[i] += shift
        else:
            self.node_to_pred[self.NodeList[0]].remove(self.NodeList[start_pos - 1])
        # Build positions and predecessors of inserted nodes
        for i, node in enumerate(nodes, start=start_pos):
            self.node_to_positions[node].append(i)
            self.node_to_pred[node].add(prev_node)
            prev_node = node
        if start_pos + shift < self.P_b:
            self.node_to_pred[next_node].add(prev_node)
        else:
            self.node_to_pred[self.NodeList[0]].add(self.NodeList[-1])
        self.NodeSet.update(nodes)

    def extend(self, insert_nodes: List[Any], position: int):
        """
        Insert nodes into the bus route at a specified position.

        Args:
            insert_nodes (List[Any]): Nodes to insert into NodeList.
            position (int): Index in NodeList to insert before.

        Notes:
            - Updates NodeList, ArcList, P_b, and all derived attributes.
            - Maintains cyclic consistency of arcs.
        """
        old_len = self.P_b
        # Insert nodes.
        self.NodeList[position:position] = insert_nodes
        # Update P_b.
        self.P_b = len(self.NodeList)
        # Compute neighbors (cyclic).
        left = self.NodeList[position - 1]
        right = self.NodeList[(position + len(insert_nodes)) % self.P_b]
        # Build new arcs (cyclically consistent).
        new_arcs = list(nx.utils.pairwise([left] + insert_nodes + [right]))
        if position == old_len:
            # Inserting at end: wrap-around arc goes to index 0.
            self.ArcList[position:position] = new_arcs[:-1]
            self.ArcList[0] = new_arcs[-1]
        else:
            # Replace old arc (left, right).
            self.ArcList[position:position + 1] = new_arcs
        self._update_attributes_initial_heuristic(nodes=insert_nodes, start_pos=position)

    def _sort_node_positions(self):
        """
        Sort all position lists in node_to_positions in-place.
        """
        for node in self.node_to_positions:
            self.node_to_positions[node].sort()

    def _compute_service_time_data(self, u: Any, v: Any):
        """
        Compute breakpoints and corresponding constants for the service time function s_b(t, u, v).

        Args:
            u (Any): node of origin
            v (Any): node of destination

        Notes:
            - node_to_positions has to be sorted already.
            - Note difference between 0-based (Python) and 1-based (Theory) positions.
        """
        _breakpoints = []
        _constants = []
        positions_u = self.node_to_positions[u]
        positions_v = self.node_to_positions[v]

        p_idx = -1
        idx = 0
        q_first = None
        for q in positions_v:
            while idx < len(positions_u) and positions_u[idx] < q:
                idx += 1
            last_valid_idx = idx - 1
            if last_valid_idx > p_idx and last_valid_idx >= 0:
                p_max = positions_u[last_valid_idx]
                c = q + 1 - self.t_b
                _breakpoints.append(p_max)
                _constants.append(c)
                if q_first is None:
                    q_first = q
                p_idx = last_valid_idx

        # Case 2: no p < q pair exists at all.
        if not _breakpoints:
            p = positions_u[-1]
            c1 = positions_v[0] + 1 - self.t_b + self.P_b
            c2 = positions_v[0] + 1 - self.t_b + 2 * self.P_b
            _breakpoints.append(p)
            _constants.append(c1)
            _constants.append(c2)
        # Case 1, subcase 1: p_{l(b,u,v)} = p_{l(b,u)}.
        elif _breakpoints[-1] == positions_u[-1]:
            c = _constants[0] + self.P_b
            _constants.append(c)
        # Case 1, subcase 2: q^{b,u,v}_1 = q^{b,v}_1.
        elif q_first == positions_v[0]:
            c = _constants[0] + self.P_b
            _constants.append(c)
        # Case 1, subcase 3: q^{b,u,v}_1 > q^{b,v}_1.
        else:
            p = positions_u[-1]
            c1 = positions_v[0] + 1 - self.t_b + self.P_b
            c2 = _constants[0] + self.P_b
            _breakpoints.append(p)
            _constants.append(c1)
            _constants.append(c2)

        self.breakpoints[(u,v)] = _breakpoints
        self.constants[(u,v)] = _constants
        self.actual_breakpoints[(u,v)] = [p + 1 - self.t_b for p in _breakpoints]

    def _compute_service_time_intervals(self, u: Any, v: Any):
        """
        Compute the exact intervals [t_j, t_{j+1}) for the service time function s_b(t, u, v).

        Args:
            u (Any): node of origin
            v (Any): node of destination

        Notes:
            - Needs a computation of self.actual_breakpoints (e.g. in self.compute_service_time_data) first.
            - Note difference between 0-based (Python) and 1-based (Theory) positions.
        """
        transformed = self.actual_breakpoints[(u,v)]
        _intervals = [0.0] + transformed + [float(self.P_b)] # If first interval [0,0], we get [00...].
        self.intervals[(u, v)] = _intervals

    def _compute_service_time(self):
        """
         Computes all necessary data for the service time of all nodes pairs of the bus.
        """
        for u in self.NodeSet:
            for v in self.NodeSet:
                if v != u:
                    self._compute_service_time_data(u, v)
                    self._compute_service_time_intervals(u, v)

    def ensure_pair(self, u: Any, v: Any):
        """
        Make sure the service time data of the pair (u,v) is available, computing it on first access.

        Args:
            u (Any): node of origin
            v (Any): node of destination

        Raises:
            KeyError: If the bus does not cover (u,v).
        """
        if (u, v) in self.intervals:
            return
        if not self.covers(u, v):
            raise KeyError((u, v))

        self._compute_service_time_data(u, v)
        self._compute_service_time_intervals(u, v)

    def init_service_time(self, precompute: bool = True):
        """
        Initialize necessary attributes and sort node_to_positions in-place.

        Args:
            precompute (bool): If True, the service time data of every covered pair is computed right away.
        """
        # Sort positions
        self._sort_node_positions()

        # Compute all necessary service time items.
        if precompute:
            self._compute_service_time()

    def time_to_constant(self, t: float, u: Any, v: Any) -> float:
        """
        Computes self.constant[u,v] entry relative to the time t.

        Args:
            t (float): time
            u (Any): node of origin
            v (Any): node of destination

        Returns:
            Constant c.
        """
        self.ensure_pair(u, v)
        t_period = t % self.P_b
        intervals = self.intervals[(u, v)]
        idx = bisect.bisect_left(intervals, t_period) - 1
        idx = max(0, idx) # If t_period == 0
        c = self.constants[(u,v)][idx]

        return c

    def service_time(self, t: float, u: Any, v: Any) -> float:
        """
        Computes exact service time s_b(t, u, v).

        Args:
            t (float): time
            u (Any): node of origin
            v (Any): node of destination

        Returns:
            Service time
        """
        t_period = t % self.P_b
        c = self.time_to_constant(t,u,v)

        return c - t_period

    def init_intensify_heuristic(self):
        """
        Initialize necessary attributes.
        """
        ''' Node Insertion. '''
        for i in range(self.P_b):
            self.arc_to_position[self.ArcList[i]] = i