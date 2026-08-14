"""With the interaction switched off, the Green's function has a closed form.

``G(z) = ((z - mu) I - h1)^-1``.  Reproducing that exactly is the strongest
available check that the sector diagonalisation, the transition amplitudes, the
pole bookkeeping and the chemical-potential shift are all consistent: an error
in any one of them shows up here.
"""

import numpy as np
import pytest
from conftest import random_h1

from casgf import ActiveSpace, lehmann


@pytest.mark.parametrize("norb", [2, 3, 4])
@pytest.mark.parametrize("omega, eta", [(0.3, 0.05), (-1.2, 0.1), (0.0, 0.5)])
def test_matches_resolvent_without_mu(norb, omega, eta):
    h1 = random_h1(norb, seed=norb)
    nelec = (norb // 2, norb // 2)
    gf = lehmann(ActiveSpace.from_arrays(h1, np.zeros((norb,) * 4), nelec), mu=0.0)

    expected = np.linalg.inv((omega + 1j * eta) * np.eye(norb) - h1)
    assert np.abs(gf.at(omega, eta) - expected).max() < 1e-12


@pytest.mark.parametrize("omega, eta", [(0.3, 0.05), (-1.2, 0.1)])
def test_chemical_potential_is_a_rigid_level_shift(omega, eta):
    """A non-zero mu must act exactly like adding ``mu`` to every orbital energy."""
    norb = 4
    h1 = random_h1(norb, seed=3)
    gf = lehmann(ActiveSpace.from_arrays(h1, np.zeros((norb,) * 4), nelecas=4))

    shifted = h1 + gf.mu * np.eye(norb)
    expected = np.linalg.inv((omega + 1j * eta) * np.eye(norb) - shifted)
    assert np.abs(gf.at(omega, eta) - expected).max() < 1e-12


def test_poles_are_the_orbital_energies():
    norb = 4
    h1 = random_h1(norb, seed=5)
    gf = lehmann(ActiveSpace.from_arrays(h1, np.zeros((norb,) * 4), nelecas=4), mu=0.0)

    # Only the poles carrying spectral weight are physical; the rest of the
    # N+-1 spectrum is orthogonal to c^dag|0> and drops out.
    weights = np.einsum("in,in->n", gf.residues, gf.residues)
    live = np.sort(gf.poles[weights > 1e-10])
    assert np.allclose(live, np.linalg.eigvalsh(h1), atol=1e-10)
