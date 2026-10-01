from typing import Optional

class Budget:
    """
    Counts objective evaluations (evaluate_move calls).

    Args:
        max_evaluations (Optional[int]): Cap on spend(). None means unlimited.

    Notes:
        - ILS progress is measured in evaluations.
    """

    def __init__(self, max_evaluations: Optional[int] = None):
        self.max_evaluations = max_evaluations
        self.spent = 0

    def spend(self, n: int = 1) -> None:
        self.spent += n

    @property
    def exhausted(self) -> bool:
        return self.max_evaluations is not None and self.spent >= self.max_evaluations