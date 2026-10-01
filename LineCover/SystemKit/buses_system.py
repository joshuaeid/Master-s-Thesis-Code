from typing import Any, List, Tuple, Set, Dict, Iterator, Iterable, Optional
from itertools import groupby
from statistics import NormalDist
import heapq, math, random
from LineCover.SystemKit import Bus

def system_period(G: Any) -> int:
    """
    System period T = lcm(1, ..., m) of a graph with m arcs.

    Args:
        G (Any): nx.DiGraph.

    Returns:
        int: T = lcm(1, ..., m)
    """
    return math.lcm(*range(1, G.number_of_edges() + 1))

class BusSystem:
    """
    Represents a system of buses and evaluates the ensemble-average objective.

    Attributes:
        buses (Dict[Any, Bus]): Mapping bus IDs to Bus objects.
        nodes (Set[Any]): Set of all nodes in the system.
        lambda_ (float): Request rate for the system.
        pair_means (Dict[Tuple[Any, Any], float]): Time average for every OD pair.
        _pair_mean_sum (float): Ensemble Average without 1/n(n-1)
    """

    def __init__(self, buses: Dict[Any, Bus], V: Set[Any], lambda_: float = 1.0,
                 lazy_service_time: bool = False):
        """
        Initialize a BusSystem object.

        Args:
            buses (Dict[Any, Bus]): Mapping bus IDs to Bus objects.
            V (Set[Any]): Set of all nodes in the system.
            lambda_ (float): Request rate for the system.
            lazy_service_time (bool): If True, the service time data of a pair is computed on first access.
        """
        self.buses: Dict[Any, Bus] = buses
        self.nodes: Set[Any] = set(V)
        self.n = len(V)
        self.lazy_service_time: bool = lazy_service_time

        for bus in self.buses.values():
            bus.init_service_time(precompute=not lazy_service_time)

        self.lambda_: float = lambda_

        # Time average m_{u,v} of every pair.
        self.pair_means: Dict[Tuple[Any, Any], float] = {}
        self._pair_mean_sum: Optional[float] = None

        # Monte Carlo objective, None means exact objective.
        self._mc_samples: Optional[List[Tuple[int, float, Any, Any]]] = None
        self._mc_values: Optional[List[float]] = None
        self._mc_contributions: Optional[List[Dict[Any, float]]] = None
        self._mc_value: Optional[float] = None
        self._mc_T: Optional[int] = None

    def is_node_pair(self, u: Any, v: Any) -> bool:
        """
        Whether (u,v) is an OD pair of the system.

        Args:
            u (Any): Origin node.
            v (Any): Destination node.

        Returns:
            bool: True iff u and v are distinct nodes of the system.
        """
        return u != v and u in self.nodes and v in self.nodes

    def node_pairs(self) -> Iterator[Tuple[Any, Any]]:
        """
        Iterate over the OD pairs of the system.

        Yields:
            Tuple[Any, Any]: Every ordered pair of distinct nodes.
        """
        for u in self.nodes:
            for v in self.nodes:
                if u != v:
                    yield (u, v)

    def number_of_node_pairs(self) -> int:
        """
        Returns:
            int: n(n-1), the number of OD pairs.
        """
        return self.n * (self.n - 1)

    def clone(self) -> "BusSystem":
        """
        Deep copy with new Bus objects and copied pair means.

        Returns:
            other (BusSystem): Cloned BusSystem.
        """
        buses = {
            bid: Bus(bus_id=b.id, arc_list=list(b.ArcList), full_init=True, t_b=b.t_b,
                     lazy_service_time=self.lazy_service_time)
            for bid, b in self.buses.items()
        }
        other = BusSystem(buses=buses, V=self.nodes, lambda_=self.lambda_,
                          lazy_service_time=self.lazy_service_time)
        other.pair_means = dict(self.pair_means)
        other._pair_mean_sum = self._pair_mean_sum
        other._mc_samples = self._mc_samples
        other._mc_values = None if self._mc_values is None else list(self._mc_values)
        other._mc_contributions = (None if self._mc_contributions is None else [dict(contribution) for contribution in self._mc_contributions])
        other._mc_value = self._mc_value
        other._mc_T = self._mc_T
        return other

    def _get_buses_for_pair(self, u: Any, v: Any) -> List[Bus]:
        """
        Returns all buses that cover the pair (u,v), i.e. B_{u,v}.

        Args:
            u (Any): Origin of request.
            v (Any): Destination of request.

        Returns:
            List[Bus]: List of buses that cover the pair (u,v).
        """
        return [bus for bus in self.buses.values() if bus.covers(u, v)]

    def covering_buses(self) -> Dict[Tuple[Any, Any], List[Any]]:
        """
        The sets B_{u,v} of all covered pairs.

        Returns:
            Dict[Tuple[Any, Any], List[Any]]: Pair -> ids of the covering buses, ascending.
        """
        covering: Dict[Tuple[Any, Any], List[Any]] = {}
        for bus_id in sorted(self.buses):
            for pair in self.buses[bus_id].covered_pairs():
                covering.setdefault(pair, []).append(bus_id)

        return covering

    def service_time(self, t: float, u: Any, v: Any) -> float:
        """
        Exact system service time s(t,u,v): minimum over all covering buses.

        Args:
            t (float): Time at which to evaluate the service time.
            u (Any): Origin node.
            v (Any): Destination node.

        Returns:
            float: Exact service time of the system at time t for a request (u,v).

        Raises:
            ValueError: If (u,v) is covered by no bus.
        """
        buses_covering = self._get_buses_for_pair(u, v)
        if not buses_covering:
            raise ValueError(f"Pair {(u, v)} is covered by no bus.")

        return min(bus.service_time(t, u, v) for bus in buses_covering)

    def _bus_period_dissection(self, bus: Bus, u: Any, v: Any) -> Tuple[List[float], List[float]]:
        """
        Dissection of one bus period [0, P_b) for the pair (u,v).

        Args:
            bus (Bus): Bus covering (u,v).
            u (Any): Origin node.
            v (Any): Destination node.

        Returns:
            Tuple[List[float], List[float]]: starts and consts, consts[m] valid on [starts[m], starts[m+1]).

        Notes:
            - An interval of length zero at t = 0 is dropped.
        """
        bus.ensure_pair(u, v)
        ivs = bus.intervals[(u, v)]
        cs = bus.constants[(u, v)]
        idx = 1 if ivs[0] == ivs[1] else 0

        return ivs[idx:len(ivs) - 1], cs[idx:]

    def _iter_bus_jumps(self, bus: Bus, u: Any, v: Any, k: int) -> Iterator[Tuple[float, Any, float]]:
        """
        Unrolled upward jumps of c_b over (0, k * P_b) for the pair (u,v).

        Args:
            bus (Bus): Bus covering (u,v).
            u (Any): Origin node.
            v (Any): Destination node.
            k (int): Number of periods, T_{u,v} / P_b.

        Yields:
            Tuple[float, Any, float]: (t, bus.id, c_new) for every t at which c_b increases.

        Notes:
            - Only the transition between two periods can be no jump, so only this case is checked.
        """
        starts, consts = self._bus_period_dissection(bus, u, v)
        P_b = bus.P_b
        m_count = len(starts)
        previous = consts[0]

        for m_period in range(k):
            offset = m_period * P_b
            if m_period > 0:
                # Transition to the next period.
                current = consts[0] + offset
                if current - previous > 0.0:
                    yield starts[0] + offset, bus.id, current
                previous = current
            # Within one period every entry is a jump.
            for m in range(1, m_count):
                current = consts[m] + offset
                yield starts[m] + offset, bus.id, current
                previous = current

    def _representative(self, order: List[Any], minimizing: Set[Any]) -> int:
        """
        Deterministic representative of the minimizing set.

        Args:
            order (List[Any]): Fixed bus-id order for this pair.
            minimizing (Set[Any]): Current minimizing set M.

        Returns:
            int: First bus id of order contained in minimizing.
        """
        for bus_id in order:
            if bus_id in minimizing:
                return bus_id
        raise ValueError("Minimizing set is empty.")

    def sweep_pair(self, u: Any, v: Any) -> Iterator[Tuple[float, float, float, Any]]:
        """
        Sweep of the system service time of one pair over [0, T_{u,v}) with true breakpoints only.

        Args:
            u (Any): Origin node.
            v (Any): Destination node.

        Yields:
            Tuple[float, float, float, Any]: (t_start, t_end, C, bus_id) per interval with service time C - t.

        Raises:
            ValueError: If (u,v) is covered by no bus.

        Notes:
            - C is the absolute intercept c + floor(t/P_b) * P_b.
        """
        buses_covering = self._get_buses_for_pair(u, v)
        if not buses_covering:
            raise ValueError(f"Pair {(u, v)} is covered by no bus.")

        T_uv = math.lcm(*(bus.P_b for bus in buses_covering))

        order = [bus.id for bus in buses_covering]
        intercept: Dict[Any, float] = {}
        for bus in buses_covering:
            _, consts = self._bus_period_dissection(bus, u, v)
            intercept[bus.id] = consts[0]

        c = min(intercept.values())
        minimizing = {bus_id for bus_id, value in intercept.items() if value == c}
        t_prev = 0.0

        streams = [
            self._iter_bus_jumps(bus, u, v, T_uv // bus.P_b)
            for bus in buses_covering
        ]
        merged = heapq.merge(*streams, key=lambda entry: entry[0])

        for t, group in groupby(merged, key=lambda entry: entry[0]):
            jumped = set()
            for _, bus_id, value in group:
                intercept[bus_id] = value
                jumped.add(bus_id)

            if minimizing <= jumped:
                # Every minimizer jumped, so this is a true breakpoint.
                yield t_prev, t, c, self._representative(order, minimizing)
                c = min(intercept.values())
                minimizing = {bus_id for bus_id, value in intercept.items() if value == c}
                t_prev = t
            else:
                # Some minimizer did not jump, so the service time is continuous here.
                minimizing -= jumped

        yield t_prev, float(T_uv), c, self._representative(order, minimizing)

    def sweep_pair_fine(self, u: Any, v: Any
                        ) -> Iterator[Tuple[float, float, Dict[Any, float], Any, Any]]:
        """
        Sweep of one pair over [0, T_{u,v}) with all breakpoints of all buses.

        Args:
            u (Any): Origin node.
            v (Any): Destination node.

        Yields:
            Tuple[float, float, Dict[Any, float], Any, Any]: (t_start, t_end, intercepts, jumping, minimizer) per interval.

        Raises:
            ValueError: If (u,v) is covered by no bus.

        Notes:
            - Needed for the regions, since sweep_pair drops the breakpoints that are not true ones.
            - If several buses jump at the same t, delta lies on a hyperplane.
        """
        buses_covering = self._get_buses_for_pair(u, v)
        if not buses_covering:
            raise ValueError(f"Pair {(u, v)} is covered by no bus.")

        T_uv = math.lcm(*(bus.P_b for bus in buses_covering))

        order = [bus.id for bus in buses_covering]
        intercept: Dict[Any, float] = {}
        for bus in buses_covering:
            _, consts = self._bus_period_dissection(bus, u, v)
            intercept[bus.id] = consts[0]

        streams = [
            self._iter_bus_jumps(bus, u, v, T_uv // bus.P_b)
            for bus in buses_covering
        ]
        merged = heapq.merge(*streams, key=lambda entry: entry[0])

        t_prev = 0.0
        for t, group in groupby(merged, key=lambda entry: entry[0]):
            pending = {}
            for _, bus_id, value in group:
                pending[bus_id] = value

            current = min(intercept.values())
            minimizing = {bus_id for bus_id, value in intercept.items() if value == current}
            yield (t_prev, t, dict(intercept),
                   self._representative(order, set(pending)),
                   self._representative(order, minimizing))

            intercept.update(pending)
            t_prev = t

        current = min(intercept.values())
        minimizing = {bus_id for bus_id, value in intercept.items() if value == current}
        yield t_prev, float(T_uv), dict(intercept), None, self._representative(order, minimizing)

    def pair_mean(self, u: Any, v: Any) -> float:
        """
        Time average m_{u,v} of the system service time over [0, T_{u,v}).

        Args:
            u (Any): Origin node.
            v (Any): Destination node.

        Returns:
            float: (1 / T_{u,v}) * int_0^{T_{u,v}} s(t,u,v) dt.
        """
        total = 0.0
        t_end = 0.0
        for t_start, t_end, c, _ in self.sweep_pair(u, v):
            total += c * (t_end - t_start)
        total -= 0.5 * t_end * t_end

        return total / t_end

    def init_pair_means(self) -> None:
        """
        Compute and cache the mean m_{u,v} of every pair.

        Notes:
            - Does nothing in Monte Carlo mode.
        """
        if self.mc_enabled:
            return
        self.pair_means = {(u, v): self.pair_mean(u, v) for (u, v) in self.node_pairs()}
        self._pair_mean_sum = sum(self.pair_means.values())

    def update_pair_means(self, pairs: Iterable[Tuple[Any, Any]]) -> None:
        """
        Recompute the cached means of the given pairs and update the running sum.

        Args:
            pairs (Iterable[Tuple[Any, Any]]): Pairs whose set B_{u,v} may have changed.

        Notes:
            - In Monte Carlo mode the estimate is recomputed on the next read.
        """
        if self.mc_enabled:
            self.invalidate_mc()
            return

        if self._pair_mean_sum is None:
            self.init_pair_means()
            return

        for pair in pairs:
            if not self.is_node_pair(*pair):
                continue
            self._pair_mean_sum -= self.pair_means.get(pair, 0.0)
            value = self.pair_mean(*pair)
            self.pair_means[pair] = value
            self._pair_mean_sum += value

    def ensemble_average(self) -> float:
        """
        Ensemble average of the BusSystem, or its estimate in Monte Carlo mode.

        Returns:
            float: EA = (1 / (n(n-1))) * sum_{u != v} m_{u,v}.
        """
        if self.mc_enabled:
            if self._mc_value is None:
                self._compute_mc()
            return self._mc_value

        if self._pair_mean_sum is None:
            self.init_pair_means()

        return self._pair_mean_sum / (self.n * (self.n - 1))

    @property
    def mc_enabled(self) -> bool:
        """
        Returns:
            bool: True iff a Monte Carlo sample has been drawn.
        """
        return self._mc_samples is not None

    def init_mc(self, M: int, T: int, seed: int = 0) -> None:
        """
        Draw a sample of size M and switch the system to Monte Carlo mode.

        Args:
            M (int): Sample size.
            T (int): System period.
            seed (int): Seed of the sample.

        Notes:
            - A sample is a request (tau, u, v) with tau = K + F uniform on [0, T) and (u,v) uniform on the node pairs.
            - The sample stays fixed, so all systems are evaluated on the same requests.
        """
        rng = random.Random(seed)
        nodes = sorted(self.nodes)
        n = len(nodes)
        samples = []
        for _ in range(M):
            i = rng.randrange(n)
            j = rng.randrange(n - 1)
            if j >= i:
                j += 1
            samples.append((rng.randrange(T), rng.random(), nodes[i], nodes[j]))
        self._mc_samples = samples
        self._mc_T = T
        self.invalidate_mc()

    def resample_mc(self, seed: int) -> None:
        """
        Replaces the sample by a new one of the same size.

        Args:
            seed (int): Seed of the new sample.

        Raises:
            ValueError: If the system is not in Monte Carlo mode.
        """
        if not self.mc_enabled:
            raise ValueError("resample_mc requires a system in Monte Carlo mode.")
        self.init_mc(len(self._mc_samples), self._mc_T, seed=seed)

    def invalidate_mc(self) -> None:
        """
        Drops the cached estimate.
        """
        self._mc_values = None
        self._mc_value = None
        self._mc_contributions = None

    def mc_sample_service_time(self, sample: Tuple[int, float, Any, Any]) -> float:
        """
        System service time of one sample.

        Args:
            sample (Tuple[int, float, Any, Any]): A sample (K, F, u, v).

        Returns:
            float: s(tau, u, v) with tau = K + F.

        Raises:
            ValueError: If (u,v) is covered by no bus.
        """
        K, F, u, v = sample
        best: Optional[float] = None
        for bus in self.buses.values():
            if bus.covers(u, v):
                value = bus.service_time((K % bus.P_b) + F, u, v)
                if best is None or value < best:
                    best = value

        if best is None:
            raise ValueError(f"Pair {(u, v)} is covered by no bus.")
        return best

    def mc_sample_contributions(self, sample: Tuple[int, float, Any, Any]) -> Dict[Any, float]:
        """
        Service time of one sample per serving bus.

        Args:
            sample (Tuple[int, float, Any, Any]): A sample (K, F, u, v).

        Returns:
            Dict[Any, float]: Bus id -> s_b(tau, u, v) for every bus serving (u,v).
        """
        K, F, u, v = sample
        return {bus_id: bus.service_time((K % bus.P_b) + F, u, v) for bus_id, bus in self.buses.items() if bus.covers(u, v)}

    def _compute_mc(self) -> None:
        """
        Evaluate the whole sample and cache the empirical average.

        Notes:
            - Also caches the contribution of every serving bus, which lets a candidate move be evaluated without scanning all buses again.
        """
        self._mc_contributions = [self.mc_sample_contributions(sample) for sample in self._mc_samples]
        self._mc_values = [min(contributions.values()) for contributions in self._mc_contributions]
        self._mc_value = sum(self._mc_values) / len(self._mc_values)

    def mc_swap_bus(self, bus_id: Any, new_bus: Bus, apply: bool = False) -> Optional[float]:
        """
        Estimate after replacing a bus, recomputing only the affected samples.

        Args:
            bus_id (Any): Bus to replace.
            new_bus (Bus): Candidate bus.
            apply (bool): If True, the caches are updated and the swap becomes permanent.

        Returns:
            Optional[float]: The estimate after the swap, or None if a pair of the sample would be covered by no bus, in which case the caches are dropped instead.

        Notes:
            - A sample is affected only if the old or the new bus serves its pair, and a sample keeps its value unless the bus it takes the minimum over changes.
        """
        if self._mc_value is None:
            self._compute_mc()

        old_bus = self.buses[bus_id]
        total = self._mc_value * len(self._mc_values)
        updates: List[Tuple[int, float, Dict[Any, float]]] = []

        for index, sample in enumerate(self._mc_samples):
            K, F, u, v = sample
            serves_new = new_bus.covers(u, v)
            if not serves_new and not old_bus.covers(u, v):
                continue

            contributions = dict(self._mc_contributions[index])
            contributions.pop(bus_id, None)
            if serves_new:
                contributions[bus_id] = new_bus.service_time((K % new_bus.P_b) + F, u, v)
            if not contributions:
                self.invalidate_mc()
                return None

            value = min(contributions.values())
            total += value - self._mc_values[index]
            updates.append((index, value, contributions))

        estimate = total / len(self._mc_values)
        if apply:
            for index, value, contributions in updates:
                self._mc_values[index] = value
                self._mc_contributions[index] = contributions
            self._mc_value = estimate

        return estimate

    def mc_variance(self) -> float:
        """
        Estimated variance of the empirical average.

        Returns:
            float: v_M = (1 / M^2) * sum_k (s_k - mean)^2

        Raises:
            ValueError: If the system is not in Monte Carlo mode.
        """
        if not self.mc_enabled:
            raise ValueError("mc_variance requires a system in Monte Carlo mode.")
        if self._mc_value is None:
            self._compute_mc()

        M = len(self._mc_values)
        return sum((value - self._mc_value) ** 2 for value in self._mc_values) / (M * M)

    def mc_paired_difference(self, other: "BusSystem", delta: float = 0.05) -> Tuple[float, Tuple[float, float]]:
        """
        Estimate of the difference of two systems on their shared sample.

        Args:
            other (BusSystem): System to compare against, sharing this system's sample.
            delta (float): Error probability of the interval.

        Returns:
            Tuple[float, Tuple[float, float]]: Mean of s_self - s_other over the sample and its approximate (1 - delta)-confidence interval.

        Raises:
            ValueError: If the two systems do not share one sample.
        """
        if not self.mc_enabled or self._mc_samples is not other._mc_samples:
            raise ValueError("mc_paired_difference requires both systems to share one sample.")
        if self._mc_value is None:
            self._compute_mc()
        if other._mc_value is None:
            other._compute_mc()

        M = len(self._mc_values)
        differences = [own - foreign for own, foreign in zip(self._mc_values, other._mc_values)]
        mean = sum(differences) / M
        variance = sum((difference - mean) ** 2 for difference in differences) / (M * M)
        half = NormalDist().inv_cdf(1.0 - delta / 2.0) * math.sqrt(variance)

        return mean, (mean - half, mean + half)

    def mc_confidence_interval(self, delta: float = 0.05) -> Tuple[float, float]:
        """
        Approximate (1 - delta)-confidence interval for the ensemble average.

        Args:
            delta (float): Error probability. Defaults to 0.05.

        Returns:
            Tuple[float, float]: Estimate -/+ z_{1-delta/2} * sqrt(v_M).
        """
        estimate = self.ensemble_average()
        half = NormalDist().inv_cdf(1.0 - delta / 2.0) * math.sqrt(self.mc_variance())
        return estimate - half, estimate + half