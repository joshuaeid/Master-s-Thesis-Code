from typing import Any, Dict, List, Deque, Set, Tuple
from collections import defaultdict, deque
import random
import networkx as nx

def BFS_with_predecessors_and_weights(
    G: nx.DiGraph, s: Any, t: Any = None
) -> Tuple[Dict[Any, int], Dict[Any, List[Any]], Dict[Any, int]]:
    """
    Perform BFS on a directed graph G to compute the shortest path distances,
    predecessors, and number of shortest paths from a source node s.

    Args:
        G (nx.DiGraph): Directed graph.
        s (Any): Source node.
        t (Any, optional): Target node. BFS stops after the distance layer of t. Defaults to None (full BFS).

    Returns:
        node_to_distance (Dict[Any, int]): Mapping reachable nodes from s to their distance.
        node_to_pred (Dict[Any, List[Any]]): Mapping each node to a list of predecessors in the successor graph.
        node_to_sigma (Dict[Any, int]): Mapping each node to the number of shortest paths from s.
    """
    # Initialization.
    node_to_distance = {s: 0}
    node_to_pred = defaultdict(list)
    node_to_sigma = defaultdict(int)
    node_to_sigma[s] = 1
    queue = deque([s])
    _t_dist = None

    # Algorithm.
    while queue:
        v = queue.popleft()
        v_dist = node_to_distance[v]
        # Stop after the distance layer of t.
        if _t_dist is not None and v_dist > _t_dist:
            break
        for w in G.neighbors(v):
            # Is w found for the first time?
            if w not in node_to_distance:
                node_to_distance[w] = v_dist + 1
                queue.append(w)
                if t is not None and w == t:
                    _t_dist = v_dist + 1
            # Shortest path to w via v?
            if node_to_distance[w] == v_dist + 1:
                node_to_pred[w].append(v)
                node_to_sigma[w] += node_to_sigma[v]

    return node_to_distance, node_to_pred, node_to_sigma

def rand_pred(pred_v: List[Any], node_to_sigma: Dict[Any, int], v: Any) -> Any:
    """
    Select a predecessor node randomly according to weights induced by sigma.

    Args:
        pred_v (List[Any]): List of predecessors of node v.
        node_to_sigma (Dict[Any, int]): Mapping each node to the number of shortest paths from s.
        v (Any): Node of the successor graph.

    Returns:
        w (Any): Node selected from pred_v according to weighted probability.
    """
    # Initialization
    r = random.randrange(node_to_sigma[v])  # pick a random "position" from 0 to sigma[v]-1
    i = 0
    w = pred_v[i]
    r -= node_to_sigma[w]

    # Algorithm
    while r >= 0:
        i += 1
        w = pred_v[i]
        r -= node_to_sigma[w]

    return w

def brw_linear_sample(
    s: Any,
    t: Any,
    node_to_pred: Dict[Any, List[Any]],
    node_to_sigma: Dict[Any, int],
    path: Deque[Any] = deque([]),
    banned_predecessors: Set[Any] = set()
) -> Deque[Any]:
    """
    Uniformly sample a shortest path from s to t using linear weighted backtracking.

    Args:
        s (Any): Source of shortest path.
        t (Any): Target of shortest path.
        node_to_pred (Dict[Any, List[Any]]): Mapping each node to its predecessors.
        node_to_sigma (Dict[Any, int]): Mapping each node to the number of shortest paths from s.
        path (Deque[Any], optional): Existing path to extend. Defaults to empty deque.
        banned_predecessors (Set[Any], optional): Nodes to avoid in path selection. Defaults to empty set.

    Returns:
        path (Deque[Any]): Uniformly sampled shortest path from s to t.
    """
    # Initialization.
    v = t

    # Remove banned predecessors and change node_to_sigma in the case of extension of a bus. See extend_bus function.
    if banned_predecessors:
        path.appendleft(v)

        node_to_sigma_v = node_to_sigma.copy()
        node_to_pred_v = []

        for node in node_to_pred[v]:
            if node in banned_predecessors:
                node_to_sigma_v[v] -= node_to_sigma[node]
            else:
                node_to_pred_v.append(node)

        v = rand_pred(node_to_pred_v, node_to_sigma_v, v)

    # Algorithm.
    while v != s:
        path.appendleft(v)
        v = rand_pred(node_to_pred[v], node_to_sigma, v)
    path.appendleft(s)

    return path