"""
src/utils/math_utils.py

Small numeric helpers that are shared across the project.

Two helpers live here:

``to_numpy``
    Convert a PyTorch tensor (or any array-like) into a NumPy array.
    Three modules used to carry their own private copy of this three-line
    function (``src/explainability/eri.py``,
    ``src/explainability/visualization.py`` and
    ``src/explainability/degradation_viz.py``), and the copies had already
    drifted apart in whether they cast to ``float64``.

``normalize_importance``
    Turn raw importance magnitudes into a distribution that sums to one.
    The zero-total case is handled **once**, here: a vector whose total
    magnitude is zero becomes an all-zero vector.  Previously the ERI
    module returned the raw (unnormalised) magnitudes in that case while
    the visualisation module returned zeros, so the two disagreed about
    what "no importance" means.

author: me-intenzo
"""

from __future__ import annotations

import numpy as np

#: Magnitudes below this total are treated as "no importance at all".
IMPORTANCE_EPSILON: float = 1e-12


def to_numpy(
    value,
    dtype: np.dtype | type | None = None,
) -> np.ndarray:
    """Convert *value* into a NumPy array.

    Parameters
    ----------
    value:
        A PyTorch tensor, a NumPy array or any array-like object.
    dtype:
        Optional target dtype.  ``None`` keeps the original dtype, which
        matters for the plotting code paths that must not silently
        up-cast float32 arrays.

    Returns
    -------
    np.ndarray
        The converted array (detached from the autograd graph).
    """

    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()

    array = np.asarray(value)

    if dtype is not None:
        array = array.astype(dtype, copy=False)

    return array


def normalize_importance(
    values,
    dtype: np.dtype | type = np.float64,
) -> np.ndarray:
    """Normalise absolute importance magnitudes to sum to one.

    A vector whose total magnitude is zero carries no information, so it is
    returned as an all-zero vector of the same shape.  This is the single
    definition of that edge case in the project.
    """

    array = np.abs(to_numpy(values, dtype=dtype))

    total = float(array.sum())

    if total <= IMPORTANCE_EPSILON:
        return np.zeros_like(array)

    return array / total


__all__ = [
    "IMPORTANCE_EPSILON",
    "normalize_importance",
    "to_numpy",
]
