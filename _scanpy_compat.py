"""Project-local fixes for the installed Scanpy stack; no packages are modified.

Input: Scanpy's NumPy-backed igraph random-state adapter in the active process.
Output: the same adapter with an explicit 64-bit integer dtype when necessary.

NumPy RandomState.randint defaults to a 32-bit C-long on Windows. igraph can
request a range outside that dtype, making Scanpy's seeded igraph backend emit
ignored ValueErrors. This preserves the configured seed, backend, and Scanpy's
existing half-open randint bounds; only the integer storage width is repaired.
"""

from __future__ import annotations


def ensure_scanpy_compat() -> bool:
    """Repair affected Scanpy RNG adapters in memory, once; report if patched.

    A probe with a fresh seeded adapter detects the installed stack's actual
    behavior without advancing the analysis RNG. Unaffected/newer adapters are
    untouched. The helper is called explicitly after importing Scanpy.
    """
    import numpy as np
    from scanpy._utils.random import _RNGIgraph

    if getattr(_RNGIgraph, "_scp_int64_randint", False):
        return True

    try:
        _RNGIgraph(0).randint(0, 2**32 - 1)
    except ValueError as exc:
        if "out of bounds for int32" not in str(exc):
            raise
    else:
        return False

    def randint(self, low, high=None, size=None, dtype=np.int64):
        result = self._rng.randint(low, high=high, size=size, dtype=dtype)
        # igraph requires an ordinary Python integer from scalar RNG callbacks.
        return int(result) if size is None else result

    _RNGIgraph.randint = randint
    _RNGIgraph._scp_int64_randint = True
    return True
