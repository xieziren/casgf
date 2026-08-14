"""Topological winding number of ``det G`` around a closed contour.

The default contour is the boundary of the right half-disc of radius ``r`` in
the complex frequency plane, traversed anticlockwise:

1. the lower-right quarter arc, from ``-i*r`` to ``+r``;
2. the upper-right quarter arc, from ``+r`` to ``+i*r``;
3. straight down the imaginary axis, from ``+i*r`` back to ``-i*r``.

By the argument principle, the number of times ``det G(z)`` winds around the
origin along this loop is (zeros - poles) of ``det G`` enclosed by it -- an
integer that cannot change without a pole and a zero colliding, which is what
makes it a topological invariant of the reaction.

Plotting the trajectory and counting its loops by eye works but does not scale;
here the phase is unwrapped into an actual integer.
"""

from __future__ import annotations

import warnings

import numpy as np

__all__ = [
    "keyhole_contour",
    "det_along_contour",
    "winding_number",
    "winding_number_of",
    "min_pole_distance",
]


def keyhole_contour(radius: float = 1.5, n_per_segment: int = 10_000) -> np.ndarray:
    """Closed half-disc contour, as complex frequencies.

    Returns ``3 * n_per_segment`` points starting and ending at ``-i*radius``.
    """
    if radius <= 0:
        raise ValueError("radius must be positive")
    if n_per_segment < 2:
        raise ValueError("n_per_segment must be at least 2")

    w_up = np.linspace(0.0, radius, n_per_segment)
    arc = np.sqrt(np.maximum(0.0, radius**2 - w_up**2))

    lower = w_up - 1j * arc  # -i*r  ->  +r
    upper = (w_up - 1j * arc)[::-1].conj()  # +r  ->  +i*r
    closing = 1j * np.linspace(radius, -radius, n_per_segment)  # +i*r -> -i*r
    return np.concatenate([lower, upper, closing])


def min_pole_distance(gf, contour) -> float:
    """Closest approach of the contour to any pole of ``gf``.

    A contour that grazes a pole makes ``det G`` swing by more than ``pi``
    between neighbouring samples, which silently corrupts the phase unwrapping.
    """
    z = np.atleast_1d(np.asarray(contour, dtype=complex))
    return float(np.abs(z[:, None] - gf.poles[None, :]).min())


def det_along_contour(gf, contour, orbitals=None) -> np.ndarray:
    """``det G(z)`` at every point of ``contour``."""
    return np.linalg.det(gf.at_complex(contour, orbitals=orbitals))


def winding_number(values, tol: float = 0.05) -> int:
    """Turns of a closed curve around the origin, from its unwrapped phase.

    Parameters
    ----------
    values:
        ``det G`` sampled along a closed contour, in order.
    tol:
        How far from an integer the raw phase accumulation may land before this
        warns.  A large residue means the contour is too coarse, or passes too
        close to a pole or a zero.
    """
    values = np.asarray(values, dtype=complex)
    if values.size < 3:
        raise ValueError("need at least 3 contour samples")
    if np.any(values == 0):
        raise ValueError("det G vanishes exactly on the contour; the winding number is undefined")

    phase = np.unwrap(np.angle(values))
    turns = (phase[-1] - phase[0]) / (2.0 * np.pi)
    nearest = int(np.rint(turns))
    if abs(turns - nearest) > tol:
        warnings.warn(
            f"phase accumulation {turns:.3f} is {abs(turns - nearest):.3f} away from the "
            f"nearest integer {nearest}; increase n_per_segment or move the contour away "
            "from poles and zeros",
            RuntimeWarning,
            stacklevel=2,
        )
    return nearest


def winding_number_of(gf, contour=None, orbitals=None, radius: float = 1.5,
                      n_per_segment: int = 10_000, tol: float = 0.05) -> int:
    """Winding number of ``det G`` for one symmetry block.

    ``contour`` defaults to :func:`keyhole_contour` with the given ``radius``.
    ``orbitals`` selects the block; resolve it with
    :func:`casgf.symmetry.block_indices`.
    """
    if contour is None:
        contour = keyhole_contour(radius=radius, n_per_segment=n_per_segment)
    return winding_number(det_along_contour(gf, contour, orbitals=orbitals), tol=tol)
