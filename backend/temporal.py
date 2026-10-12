"""temporal.py — k-of-n temporal voting for per-ROI decisions."""
from collections import deque
from typing import Dict, Hashable


class TemporalVoter:
    """Report a defect only if >= k of the last n per-frame decisions are positive.
    n=1,k=1 is the identity. One window per key (camera, roi, method)."""

    def __init__(self, n: int = 1, k: int = 1):
        if not (1 <= k <= n):
            raise ValueError("require 1 <= k <= n")
        self.n, self.k = n, k
        self._win: Dict[Hashable, deque] = {}

    def update(self, key: Hashable, decision: bool) -> bool:
        if self.n == 1:
            return bool(decision)
        w = self._win.setdefault(key, deque(maxlen=self.n))
        w.append(1 if decision else 0)
        return sum(w) >= self.k

    def reset(self, key: Hashable = None) -> None:
        if key is None:
            self._win.clear()
        else:
            self._win.pop(key, None)
