from typing import Dict, Tuple
from dataclasses import dataclass
import numpy as np

"""
Quadratic form of the region hopping and the contribution of a single pair (u, v).
"""

@dataclass(frozen=True)
class QuadraticForm:
    """
    A quadratic polynomial 1/2 x^T H x + h^T x + alpha in the shift variables.

    Attributes:
        H (np.ndarray): Symmetric positive semidefinite matrix, shape (N, N).
        h (np.ndarray): Linear part, shape (N,).
        alpha (float): Constant part, i.e. the value at delta = 0.
        bus_ids (Tuple[int, ...]): Buses indexing the coordinates, in order.
    """
    H: np.ndarray
    h: np.ndarray
    alpha: float
    bus_ids: Tuple[int, ...]

    def value(self, delta: np.ndarray) -> float:
        """
        Evaluate the polynomial.

        Args:
            delta (np.ndarray): Shift vector in the coordinates of bus_ids.

        Returns:
            float: 1/2 delta^T H delta + h^T delta + alpha.
        """
        return float(0.5 * delta @ self.H @ delta + self.h @ delta + self.alpha)

@dataclass(frozen=True)
class ODPairState:
    """
    System state of one (u, v) origin-destination pair.

    Attributes:
        starts (Tuple[float, ...]): True breakpoints, strictly increasing and closed by starts[0] + T_uv.
        consts (Tuple[float, ...]): Intercept on [starts[k], starts[k+1]).
        minimizers (Tuple[int, ...]): Minimizing bus on interval k.
        T_uv (float): Period of the pair.
    """
    starts: Tuple[float, ...]
    consts: Tuple[float, ...]
    minimizers: Tuple[int, ...]
    T_uv: float

    def __post_init__(self):
        if not (len(self.starts) == len(self.consts) == len(self.minimizers)):
            raise ValueError("ODPairState: starts, consts, minimizers must have equal length.")

def q_form_from_pair_state(state: ODPairState, index: Dict[int, int]
                           ) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    H_uv, h_uv, alpha_uv from a ODPairState.

    Args:
        state (ODPairState): State of the pair.
        index (Dict[int, int]): Bus id -> coordinate for every bus in state.minimizers.

    Returns:
        Tuple[np.ndarray, np.ndarray, float]: H, h, alpha without the weight 1 / (n(n-1) T_uv).
    """
    n = len(index)
    starts = np.asarray(state.starts, dtype=float)
    consts = np.asarray(state.consts, dtype=float)
    minimizers = list(state.minimizers)
    T_uv = state.T_uv

    length = len(starts)
    widths = np.empty(length, dtype=float)
    widths[:-1] = starts[1:] - starts[:-1]
    widths[-1] = starts[0] + T_uv - starts[-1]

    J = np.zeros((length, n), dtype=float)
    for i, bus_id in enumerate(minimizers):
        previous = minimizers[i - 1]
        if bus_id != previous:
            J[i, index[bus_id]] += 1.0
            J[i, index[previous]] -= 1.0

    H = J.T @ J
    h = J.T @ (consts - starts)
    alpha = float(consts @ widths - starts[0] * T_uv - 0.5 * T_uv * T_uv)

    return H, h, alpha