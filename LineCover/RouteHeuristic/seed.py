def move_seed(iteration: int, bus_id: int, position: int, L: int) -> int:
    """
    Deterministic seed derived from (iteration, bus_id, position, L).

    Notes:
        - Arithmetic combination instead of hash() to be independent of CPython.
        - The prime base 1_000_003 is larger than bus_id and L, so there are no collisions in practice.
    """
    return ((int(iteration) * 1_000_003 + int(bus_id)) * 1_000_003 + int(position)) * 1_000_003 + int(L)