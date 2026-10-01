from collections import deque
from typing import Optional
from LineCover.SystemKit import Bus
from LineCover.InitialHeuristic.build_buses_storage import BuildBusesStorage
from LineCover.InitialHeuristic.shortest_path_sampling import brw_linear_sample

def delete_node(bus: Bus,
                node_pos: int,
                build_buses_storage: BuildBusesStorage) -> Optional[Bus]:
    """
    Deletes a node x at a specific position from the bus and changes the bus such that the circuit property is fulfilled.

    Arguments:
        bus (Bus): Bus
        node_pos (int): position of node in bus.NodeList
        build_buses_storage (BuildBusesStorage): BuildBusesStorage

    Returns:
        Optional[Bus]: The resulting bus (full_init=True), or None if no
        valid deletion exists (bus left unchanged).

    Notes:
        - If the path is u -> x -> u, deletes it.
        - If u and v are adjacent (d(u,v) = 1), replaces u -> x -> v by u -> v.
        - Otherwise tries to find another path u -> y -> v, such that not both arcs are already in the bus.
        - If such path does not exist, no change of the bus.
        - If such a path exists, adapts the bus and reinitialize all attributes.
        - A path exchange is only applied if the result is a circuit (no repeated arc).
    """
    u = bus.NodeList[node_pos - 1]
    x = bus.NodeList[node_pos]
    v = bus.NodeList[(node_pos + 1) % bus.P_b]

    if u == v:
        ''' Delete the path u -> x -> v. '''
        new_arc_list = list(bus.ArcList)
        start_idx = node_pos
        end_idx = (node_pos + 1) % bus.P_b
        if start_idx < bus.P_b - 2:
            new_arc_list = new_arc_list[:start_idx] + new_arc_list[end_idx + 1:]
        elif start_idx == bus.P_b - 2:
            new_arc_list = new_arc_list[:start_idx]
        else:
            new_arc_list = new_arc_list[end_idx + 1:start_idx]

        return Bus(bus_id=bus.id, arc_list=new_arc_list, full_init=True, t_b=bus.t_b,
                   lazy_service_time=bus.lazy_service_time)

    else:
        d_uv = build_buses_storage.pair_to_distance[(u, v)]

        if d_uv == 1:
            ''' Direct arc u -> v exists: replace u -> x -> v by u -> v. '''
            if (u, v) in bus.arc_to_position:
                ''' Arc (u,v) already in bus. No node deletion (would duplicate it). '''
                return None

            new_arc_list = list(bus.ArcList)
            if node_pos < bus.P_b - 1:
                ''' (u,x) and (x,v) are consecutive at node_pos, node_pos + 1. '''
                new_arc_list[node_pos:node_pos + 2] = [(u, v)]
            else:
                ''' Wrap-around: (u,x) at P_b - 1, (x,v) at 0. '''
                new_arc_list = [(u, v)] + new_arc_list[1:bus.P_b - 1]

            return Bus(bus_id=bus.id, arc_list=new_arc_list, full_init=True, t_b=bus.t_b,
                   lazy_service_time=bus.lazy_service_time)

        ''' d(u,v) = 2: compute an alternative path u -> y -> v. '''
        predecessors_u = build_buses_storage.predecessors[u]
        sigma_u = build_buses_storage.sigma[u]
        banned_predecessors = {x}
        path_exchange = False
        found = False

        while len(banned_predecessors) < len(predecessors_u[v]):

            # Sample uniformly a shortest path u -> v, excluding u -> x -> v.
            path = brw_linear_sample(u, v, predecessors_u, sigma_u, path=deque([]), banned_predecessors=banned_predecessors)
            y = path[1]

            if (u, y) in bus.arc_to_position and (y, v) in bus.arc_to_position:
                ''' Both arcs of the path are in the bus. No node deletion. '''
                banned_predecessors.add(y)

            elif not (u, y) in bus.arc_to_position and not (y, v) in bus.arc_to_position:
                ''' No arcs of the path are in the bus. Insert path u -> y -> v at the same position. '''
                found = True
                break

            else:
                ''' Exactly one arc is already in bus. '''
                path_exchange = True
                found = True
                path_to_reverse = list()

                if (u, y) in bus.arc_to_position:
                    ''' Only the first arc (u,y) is in bus. Walk from y back to u and reverse. '''
                    block_start = bus.arc_to_position[u, y]
                    block_end = (node_pos + 1) % bus.P_b
                    pos = block_start
                    for _ in range(bus.P_b):
                        pos = (pos + 1) % bus.P_b
                        arc = bus.ArcList[pos]
                        path_to_reverse.append(arc)
                        if arc[1] == u:
                            break
                    path_to_insert = [(m, n) for (n, m) in path_to_reverse[::-1]] + [(y, v)]
                else:
                    ''' Only the second arc (y,v) is in bus. Walk from v to y and reverse. '''
                    block_start = node_pos
                    block_end = bus.arc_to_position[y, v]
                    pos = (node_pos + 1) % bus.P_b
                    for _ in range(bus.P_b):
                        pos = (pos + 1) % bus.P_b
                        arc = bus.ArcList[pos]
                        path_to_reverse.append(arc)
                        if arc[1] == y:
                            break
                    path_to_insert = [(u, y)] + [(m, n) for (n, m) in path_to_reverse[::-1]]

                # If the reversed loop reuses an arc, the result is no circuit. Ban y and try again.
                if block_start <= block_end:
                    kept_arcs = bus.ArcList[:block_start] + bus.ArcList[block_end + 1:]
                else:
                    kept_arcs = bus.ArcList[block_end + 1:block_start]
                candidate_arcs = list(kept_arcs) + path_to_insert
                if len(set(candidate_arcs)) != len(candidate_arcs):
                    banned_predecessors.add(y)
                    path_exchange = False
                    found = False
                    continue
                break

        if not found:
            ''' No valid alternative path u -> y -> v exists. No change of the bus. '''
            return None

        elif path_exchange:
            '''
            Change the bus by replacing the contiguous block [block_start, block_end]
            (detour, reversed segment and the already present arc) by path_to_insert.
            '''
            new_arc_list = list(bus.ArcList)
            if block_start <= block_end:
                new_arc_list = new_arc_list[:block_start] + path_to_insert + new_arc_list[block_end + 1:]
            else:
                new_arc_list = path_to_insert + new_arc_list[block_end + 1:block_start]

            return Bus(bus_id=bus.id, arc_list=new_arc_list, full_init=True, t_b=bus.t_b,
                   lazy_service_time=bus.lazy_service_time)

        else:
            ''' Replace the path u -> x -> v by u -> y -> v. '''
            new_arc_list = list(bus.ArcList)
            new_arc_list[node_pos] = (u, y)
            new_arc_list[(node_pos + 1) % bus.P_b] = (y, v)

            return Bus(bus_id=bus.id, arc_list=new_arc_list, full_init=True, t_b=bus.t_b,
                   lazy_service_time=bus.lazy_service_time)