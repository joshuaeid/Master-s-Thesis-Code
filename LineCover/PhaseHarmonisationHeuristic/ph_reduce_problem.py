from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from itertools import combinations
import networkx as nx
from LineCover.SystemKit import BusSystem

Pair = Tuple[int, int]

"""
The three reductions of the phase harmonisation problem.

Notes:
    - All tuples of bus IDs are sorted ascending.
"""

@dataclass(frozen=True)
class PhaseComponent:
    """
    One connected component of the coupling graph together with its reference bus.

    Attributes:
        bus_ids (Tuple[int, ...]): Buses of the component, ascending.
        pairs (Tuple[Pair, ...]): Pairs served by at least two buses of the component.
        reference_bus_id (int): Bus fixed to delta = 0.
        covering_buses (Dict[Pair, Tuple[int, ...]]): The sets B_{u,v}, ascending.
    """
    bus_ids: Tuple[int, ...]
    pairs: Tuple[Pair, ...]
    reference_bus_id: int
    covering_buses: Dict[Pair, Tuple[int, ...]]

    @property
    def dimension(self) -> int:
        """
        Number of free shift variables of this component after fixing the reference.

        Returns:
            int: |bus_ids| - 1.
        """
        return len(self.bus_ids) - 1

    @property
    def is_rigid(self) -> bool:
        """
        Whether the component has nothing to optimise.

        Returns:
            bool: True iff it has no pair depending on delta.
        """
        return not self.pairs

@dataclass(frozen=True)
class PhaseReduction:
    """
    Result of applying the reductions to the phase harmonisation problem.

    Attributes:
        components (Tuple[PhaseComponent, ...]): Independent subproblems, ordered by smallest bus id.
        rigid_pairs (Tuple[Pair, ...]): Pairs served by exactly one bus.
        rigid_mean_sum (Optional[float]): Sum of the means of rigid_pairs, or None.
        n (int): Number of nodes.
        ambient_dimension (int): Number of buses.
    """
    components: Tuple[PhaseComponent, ...]
    rigid_pairs: Tuple[Pair, ...]
    rigid_mean_sum: Optional[float]
    n: int
    ambient_dimension: int

    @property
    def dimension(self) -> int:
        """
        Total number of free shift variables.

        Returns:
            int: Number of buses minus number of components.
        """
        return self.ambient_dimension - len(self.components)

    @property
    def constant(self) -> Optional[float]:
        """
        Contribution of the rigid pairs to the ensemble average.

        Returns:
            Optional[float]: rigid_mean_sum / (n(n-1)), or None if the sum was not computed.
        """
        if self.rigid_mean_sum is None:
            return None
        return self.rigid_mean_sum / (self.n * (self.n - 1))

def split_pairs(buses_system: BusSystem,
                serving: Optional[Dict[Pair, List[int]]] = None
                ) -> Tuple[List[Pair], List[Pair]]:
    """
    Split the OD pairs into those that depend on delta and those that do not.

    Args:
        buses_system (BusSystem): Feasible system.
        serving (Optional[Dict[Pair, List[int]]]): Result of BusSystem.covering_buses.

    Returns:
        Tuple[List[Pair], List[Pair]]: Pairs with |B_{u,v}| >= 2 and pairs with |B_{u,v}| = 1.
    """
    if serving is None:
        serving = buses_system.covering_buses()

    harmonisable: List[Pair] = []
    rigid: List[Pair] = []
    for pair, bus_ids in serving.items():
        (harmonisable if len(bus_ids) >= 2 else rigid).append(pair)

    return harmonisable, rigid

def coupling_graph(buses_system: BusSystem) -> nx.Graph:
    """
    The coupling graph G_K on the buses.

    Args:
        buses_system (BusSystem): System whose buses are the vertices.

    Returns:
        nx.Graph: Buses as nodes and edges between coupled buses.

    Notes:
        - Two buses are coupled iff they share at least two nodes.
    """
    graph = nx.Graph()
    graph.add_nodes_from(buses_system.buses)
    graph.add_edges_from(
        (bus_id, other_id)
        for bus_id, other_id in combinations(sorted(buses_system.buses), 2)
        if len(buses_system.buses[bus_id].NodeSet & buses_system.buses[other_id].NodeSet) >= 2
    )

    return graph

def reference_bus(buses_system: BusSystem, bus_ids: List[int]) -> int:
    """
    Reference bus of a component.

    Args:
        buses_system (BusSystem): System the component belongs to.
        bus_ids (List[int]): Bus ids of one component.

    Returns:
        int: The bus id of largest period. Ties are broken by the smallest id.
    """
    return max(bus_ids, key=lambda bus_id: (buses_system.buses[bus_id].P_b, -bus_id))

def reduce_phase_problem(buses_system: BusSystem,
                         compute_constant: bool = True) -> PhaseReduction:
    """
    Apply reductions and return the reduced problem data.

    Args:
        buses_system (BusSystem): Feasible system with fixed routes.
        compute_constant (bool): Whether to compute the constant of the rigid pairs.

    Returns:
        PhaseReduction: Components with pairs and reference buses, rigid pairs and their constant.
    """
    serving = buses_system.covering_buses()
    harmonisable, rigid = split_pairs(buses_system, serving)

    components_ids = sorted(
        (sorted(component) for component in nx.connected_components(coupling_graph(buses_system))),
        key=lambda component: component[0],
    )
    component_of_bus = {
        bus_id: index for index, ids in enumerate(components_ids) for bus_id in ids
    }

    pairs_per_component: List[List[Pair]] = [[] for _ in components_ids]
    for pair in harmonisable:
        pairs_per_component[component_of_bus[serving[pair][0]]].append(pair)

    components = [
        PhaseComponent(
            bus_ids=tuple(bus_ids),
            pairs=tuple(pairs_per_component[index]),
            reference_bus_id=reference_bus(buses_system, bus_ids),
            covering_buses={pair: tuple(serving[pair]) for pair in pairs_per_component[index]},
        )
        for index, bus_ids in enumerate(components_ids)
    ]

    rigid_mean_sum: Optional[float] = None
    if compute_constant:
        rigid_mean_sum = sum(
            buses_system.pair_means[pair] if pair in buses_system.pair_means
            else buses_system.pair_mean(*pair)
            for pair in rigid
        )

    return PhaseReduction(
        components=tuple(components),
        rigid_pairs=tuple(rigid),
        rigid_mean_sum=rigid_mean_sum,
        n=buses_system.n,
        ambient_dimension=len(buses_system.buses),
    )