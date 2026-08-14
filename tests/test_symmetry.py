"""Orbital-symmetry blocking of the Green's function.

When the active orbitals split into non-mixing symmetry blocks, ``G`` is block
diagonal, ``det G`` factorises, and the winding numbers of the blocks add up to
the winding number of the whole -- which is what lets a reaction change one
block's topology while leaving the total unchanged.
"""

import numpy as np
import pytest

from casgf import (
    ActiveSpace,
    block_indices,
    block_leakage,
    irrep_blocks,
    keyhole_contour,
    lehmann,
    min_pole_distance,
    parity_blocks,
    winding_number_of,
)


def block_diagonal_space(seed=17):
    """A 4-orbital active space where {0, 2} and {1, 3} do not mix.

    Both the one- and the two-electron terms are restricted to act within a
    block, so each block conserves its own particle number.
    """
    rng = np.random.default_rng(seed)
    blocks = (np.array([0, 2]), np.array([1, 3]))

    h1 = np.zeros((4, 4))
    eri = np.zeros((4, 4, 4, 4))
    for b in blocks:
        a = rng.normal(size=(2, 2))
        h1[np.ix_(b, b)] = 0.5 * (a + a.T)
    for b1 in blocks:
        for b2 in blocks:
            e = rng.normal(size=(2, 2, 2, 2)) * 0.3
            e = 0.5 * (e + e.transpose(1, 0, 2, 3))
            e = 0.5 * (e + e.transpose(0, 1, 3, 2))
            block = eri[np.ix_(b1, b1, b2, b2)]
            eri[np.ix_(b1, b1, b2, b2)] = block + e
    # Restore (ij|kl) = (kl|ij).
    eri = 0.5 * (eri + eri.transpose(2, 3, 0, 1))
    return ActiveSpace.from_arrays(h1, eri, nelecas=4, orbsym=[0, 1, 0, 1])


def test_parity_blocks():
    assert np.array_equal(parity_blocks(6)["even"], [0, 2, 4])
    assert np.array_equal(parity_blocks(6)["odd"], [1, 3, 5])


def test_irrep_blocks_from_labels():
    blocks = irrep_blocks([0, 1, 0, 1])
    assert set(blocks) == {"0", "1"}
    assert np.array_equal(blocks["0"], [0, 2])
    assert np.array_equal(blocks["1"], [1, 3])


def test_irrep_blocks_from_string_labels():
    """PySCF hands out irrep *names* as readily as ids; both must work."""
    blocks = irrep_blocks(np.array(["Au", "B1g", "Au", "B3u"]))
    assert set(blocks) == {"Au", "B1g", "B3u"}
    assert np.array_equal(blocks["Au"], [0, 2])
    assert np.array_equal(blocks["B1g"], [1])


def test_block_indices_accepts_string_labels():
    assert np.array_equal(block_indices("Au", 4, orbsym=["Au", "B1g", "Au", "B3u"]), [0, 2])


def test_irrep_and_parity_agree_for_alternating_orbitals():
    """Index-parity slicing is only valid when the orbitals alternate like this."""
    orbsym = [0, 1, 0, 1, 0, 1]
    by_irrep = irrep_blocks(orbsym)
    by_parity = parity_blocks(6)
    assert np.array_equal(by_irrep["0"], by_parity["even"])
    assert np.array_equal(by_irrep["1"], by_parity["odd"])


def test_irrep_and_parity_disagree_when_orbitals_are_not_alternating():
    """...and it silently gives the wrong blocks otherwise."""
    by_irrep = irrep_blocks([0, 0, 1, 1])
    assert not np.array_equal(by_irrep["0"], parity_blocks(4)["even"])


def test_block_indices_dispatch():
    assert block_indices(None, 4) is None
    assert block_indices("all", 4) is None
    assert np.array_equal(block_indices("even", 4), [0, 2])
    assert np.array_equal(block_indices([3, 1], 4), [3, 1])
    assert np.array_equal(block_indices("1", 4, orbsym=[0, 1, 0, 1]), [1, 3])


def test_block_indices_rejects_bad_input():
    with pytest.raises(ValueError, match="orbsym"):
        block_indices("Ag", 4)
    with pytest.raises(KeyError):
        block_indices("B1u", 4, orbsym=[0, 1, 0, 1])
    with pytest.raises(ValueError, match="out of range"):
        block_indices([0, 9], 4)


def test_green_function_is_block_diagonal():
    space = block_diagonal_space()
    gf = lehmann(space)
    g = gf.at(0.4, eta=0.1)
    off_diagonal = g[np.ix_([0, 2], [1, 3])]
    assert np.abs(off_diagonal).max() < 1e-12


def test_determinant_factorises_over_blocks():
    space = block_diagonal_space()
    gf = lehmann(space)
    freqs = np.linspace(-2, 2, 21)
    full = gf.det(freqs, eta=0.1)
    even = gf.det(freqs, eta=0.1, orbitals=[0, 2])
    odd = gf.det(freqs, eta=0.1, orbitals=[1, 3])
    assert np.abs(full - even * odd).max() < 1e-12


def test_leakage_vanishes_for_a_block_diagonal_green_function():
    gf = lehmann(block_diagonal_space())
    blocks = list(parity_blocks(4).values())
    assert block_leakage(gf.at(0.0, eta=0.1), blocks) < 1e-12


def test_leakage_is_visible_when_the_blocks_do_mix(interacting_space):
    """A generic Hamiltonian has no block structure, and the diagnostic says so."""
    gf = lehmann(interacting_space)
    blocks = list(parity_blocks(4).values())
    assert block_leakage(gf.at(0.0, eta=0.1), blocks) > 1e-3


def test_leakage_rejects_non_square_input():
    with pytest.raises(ValueError, match="square"):
        block_leakage(np.zeros((2, 3)), [np.array([0]), np.array([1])])


def test_block_winding_numbers_add_up():
    space = block_diagonal_space()
    gf = lehmann(space)
    contour = keyhole_contour(radius=1.5, n_per_segment=4000)
    assert min_pole_distance(gf, contour) > 1e-3

    total = winding_number_of(gf, contour)
    even = winding_number_of(gf, contour, orbitals=block_indices("even", 4))
    odd = winding_number_of(gf, contour, orbitals=block_indices("odd", 4))
    assert total == even + odd
