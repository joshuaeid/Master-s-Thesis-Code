from typing import Any, Dict, Set, List, Tuple
from collections import deque
from LineCover.InitialHeuristic.shortest_path_sampling import brw_linear_sample
from LineCover.InitialHeuristic.build_buses_storage import BuildBusesStorage
from LineCover.InitialHeuristic.extend_buses_storage import ExtendBusesStorage
from LineCover.SystemKit import Bus

def normalize_uncovered_pairs(uncovered_pairs: Dict[int, Set[Tuple[Any, Any]]]) -> Set[Tuple[Any, Any]]:
    """
    Normalize uncovered node pairs.

    Args:
        uncovered_pairs (Dict[int, Set[Tuple[Any, Any]]]): Dictionary mapping distance to node pairs.

    Returns:
        Set[Tuple[Any, Any]]: Set of normalized node pairs with u != v.

    Notes:
        - Normalization: u,v = min(u,v), max(u,v) so each pair is unique and consistent.
    """
    normalized_uncovered_pairs = {
        (u, v) if u <= v else (v, u)
        for pair_set in uncovered_pairs.values()
        for (u, v) in pair_set
        if u != v
    }
    return normalized_uncovered_pairs

def extend_bus(bus: Bus, u: Any, bus_u_node: Any, bus_u_pos: int, build_buses_storage: BuildBusesStorage) -> List[Any]:
    """
    Extend a bus by inserting a shortest path.

    Args:
        bus (Bus): Bus to be extended.
        u (Any): Target node to connect.
        bus_u_node (Any): Node on bus nearest to u.
        bus_u_pos (int): Position of bus_u_node in bus.
        build_buses_storage (BuildBusesStorage): Storage with graph data.

    Returns:
        List[Any]: Nodes inserted into the bus.

    Notes:
        Updates a bus by:
        - inserting a shortest path bus_u_node -> u -> bus_v_node
        - removing the arc (bus_u_node, bus_v_node)
    """
    # Get the following node.
    bus_v_pos = bus_u_pos + 1
    if bus_u_pos == bus.P_b - 1:
        bus_v_node = bus.NodeList[0]
    else:
        bus_v_node = bus.NodeList[bus_v_pos]

    # Prepare brw_linear_sample.
    predecessors = build_buses_storage.predecessors
    sigma = build_buses_storage.sigma

    # Ban predecessors to ensure a circuit.
    banned_predecessors = bus.node_to_pred[bus_v_node] - {bus_u_node}

    # Sample uniformly a shortest path u -> bus_v_node.
    path = brw_linear_sample(u, bus_v_node, predecessors[u], sigma[u], path=deque([]),
                             banned_predecessors=banned_predecessors)
    path.popleft()

    # Sample uniformly a shortest path bus_u_node -> u and concatenate.
    path = brw_linear_sample(bus_u_node, u, predecessors[bus_u_node], sigma[bus_u_node], path=path)

    # Extend the bus.
    path.popleft()
    path.pop()
    insert_nodes = list(path)
    bus.extend(insert_nodes=insert_nodes, position=bus_u_pos + 1)  # position where the path is inserted before

    return insert_nodes

def extend_buses(buses: Dict[Any, Bus], uncovered_pairs: Set[Tuple[Any, Any]],
                 build_buses_storage: BuildBusesStorage) -> None:
    """
    Extend all buses iteratively to cover remaining uncovered pairs.

    Args:
        buses (Dict[Any, Bus]): Dictionary of Bus objects.
        uncovered_pairs (Set[Tuple[Any, Any]]): Remaining uncovered node pairs.
        build_buses_storage (BuildBusesStorage): Storage with distances and predecessors.

    Notes:
        Iteratively selects closest uncovered node pairs and extends buses.
    """
    # Initialization.
    extend_buses_storage = ExtendBusesStorage(
        uncovered_pairs,
        build_buses_storage.pair_to_distance,
        buses,
        cost_method="max_metric"
    )

    # Algorithm.
    while extend_buses_storage.uncovered_pairs:
        (
            single_node_extension,
            first_expand_node,
            second_expand_node,
            bus_id,
            (bus_node, bus_pos),
        ) = extend_buses_storage.pop_closest_pair()

        bus = buses[bus_id]

        # First Extension.
        inserted_nodes = extend_bus(bus, first_expand_node, bus_node, bus_pos, build_buses_storage)
        extend_buses_storage.update_after_extension(inserted_nodes, bus)

        # Second Extension. Skip if second_expand_node is already on the bus.
        if not single_node_extension and second_expand_node not in bus.NodeSet:
            bus_node, bus_pos = extend_buses_storage.pop_closest_data(bus, second_expand_node)
            inserted_nodes = extend_bus(bus, second_expand_node, bus_node, bus_pos, build_buses_storage)
            extend_buses_storage.update_after_extension(inserted_nodes, bus)