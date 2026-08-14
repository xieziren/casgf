"""Winding numbers, checked against the argument principle.

The contour is the boundary of the right half-disc ``{|z| < r, Re z > 0}``,
traversed anticlockwise, so the winding number of a meromorphic ``f`` is
(zeros - poles) of ``f`` inside that region.  Note the origin lies *on* the
contour: in the physical problem ``w = 0`` sits in the middle of the gap where
``det G`` is finite and non-zero, so nothing sits there.
"""

import numpy as np
import pytest

from casgf import keyhole_contour, lehmann, min_pole_distance, winding_number, winding_number_of


@pytest.fixture
def contour():
    return keyhole_contour(radius=1.5, n_per_segment=4000)


def test_contour_is_closed(contour):
    assert abs(contour[0] - contour[-1]) < 1e-12
    assert abs(abs(contour[0]) - 1.5) < 1e-12


def test_contour_stays_in_the_right_half_disc(contour):
    assert contour.real.min() >= -1e-12
    assert np.abs(contour).max() <= 1.5 + 1e-12


@pytest.mark.parametrize(
    "func, expected",
    [
        (lambda z: z - 0.5, 1),  # one zero inside
        (lambda z: 1 / (z - 0.5), -1),  # one pole inside
        (lambda z: (z - 0.5) ** 2, 2),  # double zero
        (lambda z: z + 0.5, 0),  # zero in the left half plane, outside
        (lambda z: z - 5.0, 0),  # zero beyond the radius
        (lambda z: (z - 0.3) * (z - 0.9) / (z - 0.6), 1),  # two zeros, one pole
    ],
)
def test_argument_principle(contour, func, expected):
    assert winding_number(func(contour)) == expected


def test_a_curve_through_the_origin_is_rejected(contour):
    """A zero sitting exactly on the contour leaves the winding number undefined."""
    values = contour - contour[100]
    assert np.abs(values).min() == 0.0
    with pytest.raises(ValueError, match="vanishes"):
        winding_number(values)


def test_too_few_samples():
    with pytest.raises(ValueError, match="at least 3"):
        winding_number(np.array([1 + 0j, 1 + 0j]))


def test_warns_when_the_phase_does_not_close(contour):
    """An open curve accumulates a non-integer phase and must say so."""
    open_curve = contour[: len(contour) // 3] - 0.5
    with pytest.warns(RuntimeWarning, match="phase accumulation"):
        winding_number(open_curve)


def test_non_interacting_green_function_winding(free_space, contour):
    """For a non-interacting ``G``, ``det G`` has no zeros: winding counts poles only.

    Each enclosed pole contributes ``-1``, so the winding number is minus the
    number of them. Only poles carrying spectral weight count -- most of the
    ``N+-1`` spectrum is orthogonal to ``c^dag|0>`` and never appears in ``G``.
    """
    gf = lehmann(free_space)
    assert min_pole_distance(gf, contour) > 1e-3

    weights = gf.pole_weights()
    inside = np.count_nonzero((gf.poles > 0) & (gf.poles < 1.5) & (weights > 1e-10))
    assert inside > 0
    assert winding_number_of(gf, contour) == -inside


def test_radius_selects_which_poles_are_enclosed(free_space):
    """Shrinking the contour past a pole must change the winding by one."""
    gf = lehmann(free_space)
    addition = np.sort(gf.poles_add[gf.poles_add > 0])
    assert addition.size >= 2

    inner = 0.5 * (addition[0] + addition[1])
    outer = addition[1] + 0.5 * (addition[1] - addition[0])
    assert winding_number_of(gf, radius=inner, n_per_segment=4000) == -1
    assert winding_number_of(gf, radius=outer, n_per_segment=4000) == -2
