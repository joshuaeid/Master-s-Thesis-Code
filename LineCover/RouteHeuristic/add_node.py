from collections import deque
from typing import Optional
import random
import networkx as nx
from LineCover.SystemKit import Bus
from LineCover.InitialHeuristic.build_buses_storage import BuildBusesStorage
from LineCover.InitialHeuristic.shortest_path_sampling import brw_linear_sample

def _closest_nodes_to_bus(u, bus: Bus, build_buses_storage: BuildBusesStorage):
    """
    Bus nodes closest to u by graph distance.

    Args:
        u (Any): Node not on the bus.
        bus (Bus): Bus.
        build_buses_storage (BuildBusesStorage): Storage with pair_to_distance.

    Returns:
        Set[Any]: Bus nodes with minimal distance to u.

    Notes:
        - Same as ExtendBusesStorage._closest_nodes_to_bus for a single bus and a single node.
    """
    pair_to_distance = build_buses_storage.pair_to_distance
    best_dist = float('inf')
    best_nodes = set()
    for w in bus.NodeSet:
        d = pair_to_distance[(u, w)]
        if d < best_dist:
            best_dist = d
            best_nodes = {w}
        elif d == best_dist:
            best_nodes.add(w)
    return best_nodes

def _sample_bus_position(nodes, bus: Bus) -> int:
    """
    Sample a position on the bus among the given nodes' occurrences.

    Args:
        nodes (Set[Any]): Candidate nodes on the bus.
        bus (Bus): Bus.

    Returns:
        int: Sampled position of one of the nodes.

    Notes:
        - Same as ExtendBusesStorage._sample_node_position.
    """
    total = sum(len(bus.node_to_positions[w]) for w in nodes)
    r = random.randrange(total)
    for w in nodes:
        positions = bus.node_to_positions[w]
        if r < len(positions):
            return random.choice(positions)
        r -= len(positions)

def add_node(bus: Bus,
             u,
             build_buses_storage: BuildBusesStorage) -> Optional[Bus]:
    """
    Inserts node u into the bus as a detour after its closest position. Counterpart to delete_node.

    Arguments:
        bus (Bus): Bus.
        u (Any): Node to insert.
        build_buses_storage (BuildBusesStorage): Storage with predecessors, sigma and distances.

    Returns:
        Optional[Bus]: The resulting bus, or None if u is already on the bus.

    Notes:
        - Samples a -> u -> b as in extend_bus and bans predecessors of b on the bus to keep the circuit.
    """
    if u in bus.NodeSet:
        return None

    closest_nodes = _closest_nodes_to_bus(u, bus, build_buses_storage)
    position = _sample_bus_position(closest_nodes, bus) + 1

    P_b = bus.P_b
    a = bus.NodeList[position - 1]
    b = bus.NodeList[position % P_b]

    predecessors = build_buses_storage.predecessors
    sigma = build_buses_storage.sigma

    # Ban predecessors of b already on the bus (except a) to preserve the circuit.
    banned_predecessors = bus.node_to_pred[b] - {a}

    # Sample uniformly a shortest path u -> b, excluding banned predecessors.
    path = brw_linear_sample(u, b, predecessors[u], sigma[u], path=deque([]),
                             banned_predecessors=banned_predecessors)
    path.popleft()

    # Sample uniformly a shortest path a -> u and concatenate.
    path = brw_linear_sample(a, u, predecessors[a], sigma[a], path=path)

    # Insert the detour without a and b.
    path.popleft()
    path.pop()
    insert_nodes = list(path)

    new_node_list = list(bus.NodeList)
    new_node_list[position:position] = insert_nodes
    new_arc_list = list(nx.utils.pairwise([new_node_list[-1]] + new_node_list))

    return Bus(bus_id=bus.id, node_list=new_node_list, arc_list=new_arc_list,
              full_init=True, t_b=bus.t_b, lazy_service_time=bus.lazy_service_time)