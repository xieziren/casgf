"""Two independent routes to the same Green's function must agree.

:func:`casgf.lehmann` diagonalises the ``N+-1`` sectors and assembles ``G`` from
poles and residues.  :func:`casgf.reference.green_function_by_resolvent` inverts
the resolvent directly and never forms a pole, a residue or an eigenvector of
those sectors, and takes its reference state from a Davidson solve rather than a
dense one.

Nothing but the Hamiltonian is shared between them, so agreement is evidence
about the physics rather than about a shared code path.  In particular this is
what pins the sign of the chemical-potential shift on each branch: flip one and
the two implementations disagree immediately.
"""

import numpy as np
import pytest
from conftest import random_eri, random_h1

from casgf import ActiveSpace, lehmann
from casgf.reference import green_function_by_resolvent

FREQUENCIES = np.array([-1.5 + 0.05j, -0.4 + 0.2j, 0.0 + 0.1j, 0.7 + 0.05j, 2.0 + 0.3j])


@pytest.mark.parametrize("norb", [2, 3, 4])
def test_agrees_on_interacting_systems(norb):
    space = ActiveSpace.from_arrays(
        random_h1(norb, seed=norb), random_eri(norb, seed=norb + 40), nelecas=norb
    )
    fast = lehmann(space).at_complex(FREQUENCIES)
    slow = green_function_by_resolvent(space, FREQUENCIES)
    assert np.abs(fast - slow).max() < 1e-10


def test_agrees_without_interaction(free_space):
    fast = lehmann(free_space).at_complex(FREQUENCIES)
    slow = green_function_by_resolvent(free_space, FREQUENCIES)
    assert np.abs(fast - slow).max() < 1e-10


@pytest.mark.parametrize("mu", [-0.6, 0.0, 0.9])
def test_agrees_at_an_imposed_chemical_potential(interacting_space, mu):
    """Both branches must pick up the same +mu, in both implementations."""
    fast = lehmann(interacting_space, mu=mu).at_complex(FREQUENCIES)
    slow = green_function_by_resolvent(interacting_space, FREQUENCIES, mu=mu)
    assert np.abs(fast - slow).max() < 1e-10


def test_agrees_on_the_default_chemical_potential(interacting_space):
    fast = lehmann(interacting_space)
    slow = green_function_by_resolvent(interacting_space, FREQUENCIES)
    assert np.abs(fast.at_complex(FREQUENCIES) - slow).max() < 1e-10


def test_agrees_for_beta_spin(interacting_space):
    fast = lehmann(interacting_space, spin="beta").at_complex(FREQUENCIES)
    slow = green_function_by_resolvent(interacting_space, FREQUENCIES, spin="beta")
    assert np.abs(fast - slow).max() < 1e-10


def test_agrees_on_an_open_shell_reference():
    """A doublet reference, where the two spin channels genuinely differ."""
    norb = 4
    space = ActiveSpace.from_arrays(
        random_h1(norb, seed=13), random_eri(norb, seed=17), nelecas=(3, 2)
    )
    for spin in ("alpha", "beta"):
        fast = lehmann(space, spin=spin).at_complex(FREQUENCIES)
        slow = green_function_by_resolvent(space, FREQUENCIES, spin=spin)
        assert np.abs(fast - slow).max() < 1e-10, spin


def test_resolvent_rejects_an_oversized_sector():
    space = ActiveSpace.from_arrays(np.eye(10), np.zeros((10,) * 4), nelecas=10)
    with pytest.raises(ValueError, match="max_dets"):
        green_function_by_resolvent(space, [0.5 + 0.1j], max_dets=100)


def test_resolvent_rejects_a_bad_spin(interacting_space):
    with pytest.raises(ValueError, match="alpha"):
        green_function_by_resolvent(interacting_space, [0.5 + 0.1j], spin="gamma")
