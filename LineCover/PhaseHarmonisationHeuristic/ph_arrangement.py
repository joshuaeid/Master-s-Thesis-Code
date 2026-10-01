from typing import Dict, List, Tuple
from dataclasses import dataclass
from math import gcd
import bisect
from LineCover.SystemKit import BusSystem
from LineCover.PhaseHarmonisationHeuristic.ph_reduce_problem import PhaseComponent, Pair

"""
Coincidence arrangement between all bus pairs, regions and the next region after a hop.
"""

def _bisect_at(arrangement: "BusPairArrangement", d: float) -> Tuple[float, float, "ResidueContributor", "ResidueContributor", int]:
    """
    Bounds to the next residue in both directions and the bisection index at d.

    Args:
        arrangement (BusPairArrangement): Arrangement of the bus pair.
        d (float): Difference delta_b1 - delta_b2 in [0, g).

    Returns:
        Tuple[float, float, ResidueContributor, ResidueContributor, int]: Bounds, contributors and the index.
    """
    g = arrangement.g
    residues = arrangement.residues
    n = len(residues)
    idx = bisect.bisect(residues, d)

    if idx < n:
        bound_fwd = residues[idx] - d
    else:
        bound_fwd = residues[0] + g - d
    contrib_fwd = arrangement.contributors[idx % n]

    if idx > 0:
        bound_bwd = d - residues[idx - 1]
    else:
        bound_bwd = d - (residues[-1] - g)
    contrib_bwd = arrangement.contributors[idx - 1]

    return bound_fwd, bound_bwd, contrib_fwd, contrib_bwd, idx % n

@dataclass(frozen=True)
class ResidueContributor:
    """
    Contribution of a pair to a residue class.

    Attributes:
        pair (Pair): The pair (u, v).
        family (str): "breakpoint" or "footpoint".
        tooth_b1 (int): Index into the breakpoints or constants of b1.
        tooth_b2 (int): Index into the breakpoints or constants of b2.
    """
    pair: Pair
    family: str
    tooth_b1: int
    tooth_b2: int

@dataclass(frozen=True)
class BusPairArrangement:
    """
    Coincidence arrangement of a bus pair (b1, b2), b1 < b2, given by its residues mod g = gcd(P_b1, P_b2).

    Attributes:
        b1 (int): Smaller bus id.
        b2 (int): Larger bus id.
        g (int): gcd(P_b1, P_b2).
        residues (Tuple[float, ...]): Distinct residues of delta_b1 - delta_b2 in [0, g), ascending.
        contributors (Tuple[Tuple[ResidueContributor, ...], ...]): Contributors of residues[k].
    """
    b1: int
    b2: int
    g: int
    residues: Tuple[float, ...]
    contributors: Tuple[Tuple[ResidueContributor, ...], ...]

def build_bus_pair_arrangement(buses_system: BusSystem, pairs: Tuple[Pair, ...], b1: int, b2: int) -> BusPairArrangement:
    """
    Coincidence arrangement between two buses.

    Args:
        buses_system (BusSystem): System with fixed routes.
        pairs (Tuple[Pair, ...]): Pairs of the component.
        b1 (int): Smaller bus id.
        b2 (int): Larger bus id.

    Returns:
        BusPairArrangement: Arrangement over all pairs served by both buses.
    """
    bus1, bus2 = buses_system.buses[b1], buses_system.buses[b2]
    g = gcd(bus1.P_b, bus2.P_b)
    offset = bus1.t_b - bus2.t_b

    grouped: Dict[int, List[ResidueContributor]] = {}
    for (u, v) in pairs:
        if not bus1.covers(u, v) or not bus2.covers(u, v):
            continue

        bus1.ensure_pair(u, v)
        bus2.ensure_pair(u, v)
        p1, p2 = bus1.breakpoints[(u, v)], bus2.breakpoints[(u, v)]
        for i_1, pi_1 in enumerate(p1):
            for i_2, pi_2 in enumerate(p2):
                key = (pi_2 - pi_1) % g
                grouped.setdefault(key, []).append(ResidueContributor((u, v), "breakpoint", i_1, i_2))

        c1, c2 = bus1.constants[(u, v)], bus2.constants[(u, v)]
        for i_1, ci_1 in enumerate(c1):
            q_1 = round(ci_1 - 1 + bus1.t_b)
            for i_2, ci_2 in enumerate(c2):
                q_2 = round(ci_2 - 1 + bus2.t_b)
                key = (q_2 - q_1) % g
                grouped.setdefault(key, []).append(ResidueContributor((u, v), "footpoint", i_1, i_2))

    ordered_keys = sorted(grouped, key=lambda k: (k + offset) % g)
    residues = tuple((k + offset) % g for k in ordered_keys)
    contributors = tuple(tuple(grouped[k]) for k in ordered_keys)

    return BusPairArrangement(b1=b1, b2=b2, g=g, residues=residues, contributors=contributors)

def build_coincidence_arrangements(buses_system: BusSystem, component: PhaseComponent) -> Dict[Tuple[int, int], BusPairArrangement]:
    """
    Coincidence arrangements of all bus pairs of a component that share a pair.

    Args:
        buses_system (BusSystem): Reference system with fixed routes.
        component (PhaseComponent): Component.

    Returns:
        Dict[Tuple[int, int], BusPairArrangement]: (b1, b2), b1 < b2 -> arrangement. Independent of delta.
    """
    candidates = set()
    for (u, v) in component.pairs:
        serving = component.covering_buses[(u, v)]
        for i in range(len(serving)):
            for j in range(i + 1, len(serving)):
                candidates.add((serving[i], serving[j]))

    return {(b1, b2): build_bus_pair_arrangement(buses_system, component.pairs, b1, b2)
            for (b1, b2) in candidates}

def _gap(arrangement: BusPairArrangement, i: int) -> float:
    """
    Width residues[i] - residues[i-1] of the i-th residue interval, cyclic mod g.

    Args:
        arrangement (BusPairArrangement): Arrangement.
        i (int): Index mod len(residues).

    Returns:
        float: The gap in (0, g].
    """
    n = len(arrangement.residues)
    diff = arrangement.residues[i % n] - arrangement.residues[(i - 1) % n]
    if diff <= 0:
        diff += arrangement.g
    return diff

@dataclass(frozen=True)
class Region:
    """
    A region of the coincidence arrangement. Bounds are distances to the hyperplanes from the reference delta.

    Attributes:
        bus_ids (Tuple[int, ...]): Buses of the region, ascending.
        bounds (Dict[Tuple[int, int], float]): (b, b') -> tightest u with delta_b - delta_b' <= u. Missing if unconstrained.
        contributors (Dict[Tuple[int, int], Tuple[Pair, str]]): (b, b') -> pair and family of the tightest bound.
    """
    bus_ids: Tuple[int, ...]
    bounds: Dict[Tuple[int, int], float]
    contributors: Dict[Tuple[int, int], Tuple[Pair, str]]

    @property
    def margin(self) -> float:
        """
        Distance to the nearest hyperplane.

        Returns:
            float: Smallest bound, or inf. Non-positive if delta lies on a facet.
        """
        return min(self.bounds.values(), default=float("inf"))

def seed_region(component: PhaseComponent, arrangements: Dict[Tuple[int, int], BusPairArrangement], delta: Dict[int, float]) -> Tuple[Region, Dict[Tuple[int, int], int]]:
    """
    Region containing delta and the bisection index of every bus pair. Used only for the start.

    Args:
        component (PhaseComponent): Component.
        arrangements (Dict[Tuple[int, int], BusPairArrangement]): From build_coincidence_arrangements.
        delta (Dict[int, float]): Bus id -> shift.

    Returns:
        Tuple[Region, Dict[Tuple[int, int], int]]: Region and (b1, b2) -> bisection index.
    """
    bounds: Dict[Tuple[int, int], float] = {}
    contributors = {}
    indices: Dict[Tuple[int, int], int] = {}
    for (b1, b2), arrangement in arrangements.items():
        diff = delta.get(b1, 0.0) - delta.get(b2, 0.0)
        d = diff % arrangement.g
        bound_fwd, bound_bwd, contrib_fwd, contrib_bwd, idx = _bisect_at(arrangement, d)
        bounds[(b1, b2)] = bound_fwd
        bounds[(b2, b1)] = bound_bwd
        contributors[(b1, b2)] = contrib_fwd
        contributors[(b2, b1)] = contrib_bwd
        indices[(b1, b2)] = idx

    return Region(bus_ids=component.bus_ids, bounds=bounds, contributors=contributors), indices

def advance_region(arrangements: Dict[Tuple[int, int], BusPairArrangement], region: Region, indices: Dict[Tuple[int, int], int],
                   facet: Tuple[int, int], moved: Dict[int, float], step: float) -> Tuple[Region, Dict[Tuple[int, int], int]]:
    """
    Neighbouring region after crossing a facet.

    Args:
        arrangements (Dict[Tuple[int, int], BusPairArrangement]): From build_coincidence_arrangements.
        region (Region): Region before the hop.
        indices (Dict[Tuple[int, int], int]): Its bisection indices.
        facet (Tuple[int, int]): Crossed facet, a key of region.bounds.
        moved (Dict[int, float]): Bus id -> shift of this hop.
        step (float): Step past the hyperplane.

    Returns:
        Tuple[Region, Dict[Tuple[int, int], int]]: New region and indices, same as seed_region at the new delta.

    Notes:
        - Bounds are updated by subtraction. Only if a pair crosses its own bound a new bisection is done.
    """
    b, other = facet
    b1, b2 = (b, other) if b < other else (other, b)
    arrangement = arrangements[(b1, b2)]
    n = len(arrangement.residues)
    old_idx = indices[(b1, b2)]

    new_bounds = dict(region.bounds)
    new_contributors = dict(region.contributors)
    new_indices = dict(indices)

    if facet == (b1, b2):
        idx_new = (old_idx + 1) % n
        new_bounds[(b1, b2)] = _gap(arrangement, idx_new) - step
        new_bounds[(b2, b1)] = step
    else:
        idx_new = (old_idx - 1) % n
        new_bounds[(b1, b2)] = step
        new_bounds[(b2, b1)] = _gap(arrangement, idx_new) - step

    new_contributors[(b1, b2)] = arrangement.contributors[idx_new % n]
    new_contributors[(b2, b1)] = arrangement.contributors[(idx_new - 1) % n]
    new_indices[(b1, b2)] = idx_new % n

    seen = {(b1, b2)}
    for key in region.bounds:
        p1, p2 = key
        canon = (p1, p2) if p1 < p2 else (p2, p1)
        if canon in seen:
            continue
        seen.add(canon)

        shift = moved.get(canon[0], 0.0) - moved.get(canon[1], 0.0)
        if shift == 0.0:
            continue

        fwd_new = region.bounds[canon] - shift
        bwd_new = region.bounds[(canon[1], canon[0])] + shift
        if fwd_new > 0.0 and bwd_new > 0.0:
            # Same bisection interval.
            new_bounds[canon] = fwd_new
            new_bounds[(canon[1], canon[0])] = bwd_new
            continue

        # Hyperplane of this pair crossed, new bisection.
        other_arrangement = arrangements[canon]
        d = other_arrangement.residues[indices[canon]] - region.bounds[canon]
        d_new = (d + shift) % other_arrangement.g
        (new_bounds[canon], new_bounds[(canon[1], canon[0])],
         new_contributors[canon], new_contributors[(canon[1], canon[0])],
         new_indices[canon]) = _bisect_at(other_arrangement, d_new)

    return Region(bus_ids=region.bus_ids, bounds=new_bounds, contributors=new_contributors), new_indices