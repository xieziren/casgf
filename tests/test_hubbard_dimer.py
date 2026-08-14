"""The two-site Hubbard model at half filling, where everything is analytic.

    H = -t sum_s (c^dag_1s c_2s + h.c.) + U sum_i n_i,up n_i,dn

Ground-state energies:

* ``N = 2`` (one up, one down):  ``E_0 = U/2 - sqrt((U/2)^2 + 4 t^2)``
* ``N = 1``: a single electron in the bonding/antibonding orbital, ``E = -+ t``
* ``N = 3``: exactly one doubly occupied site plus a hole hopping, ``E = U -+ t``

so the addition and removal poles are known in closed form.  This exercises the
interacting machinery -- correlation, both spin sectors, the chemical potential
-- against a result that owes nothing to this implementation.
"""

import numpy as np
import pytest

from casgf import ActiveSpace, lehmann


def hubbard_dimer(t=1.0, u=4.0):
    h1 = np.array([[0.0, -t], [-t, 0.0]])
    eri = np.zeros((2, 2, 2, 2))
    eri[0, 0, 0, 0] = eri[1, 1, 1, 1] = u
    return ActiveSpace.from_arrays(h1, eri, nelecas=(1, 1))


@pytest.mark.parametrize("t, u", [(1.0, 4.0), (1.0, 0.5), (0.7, 8.0)])
def test_ground_state_energy(t, u):
    from casgf.ed import solve_sector

    space = hubbard_dimer(t, u)
    sector = solve_sector(space.h1, space.eri, 2, (1, 1))
    expected = u / 2 - np.sqrt((u / 2) ** 2 + 4 * t**2)
    assert abs(sector.e_gs - expected) < 1e-12


@pytest.mark.parametrize("t, u", [(1.0, 4.0), (1.0, 0.5), (0.7, 8.0)])
def test_poles(t, u):
    space = hubbard_dimer(t, u)
    gf = lehmann(space)

    e0 = u / 2 - np.sqrt((u / 2) ** 2 + 4 * t**2)
    e_minus, e_plus = -t, u - t  # N-1 and N+1 ground states
    mu = 0.5 * (e_minus - e_plus)
    assert abs(gf.mu - mu) < 1e-12

    expected_add = np.sort(np.array([u - t, u + t]) - e0 + mu)
    expected_rem = np.sort(e0 - np.array([-t, t]) + mu)
    assert np.allclose(np.sort(gf.poles_add), expected_add, atol=1e-12)
    assert np.allclose(np.sort(gf.poles_rem), expected_rem, atol=1e-12)


@pytest.mark.parametrize("t, u", [(1.0, 4.0), (0.7, 8.0)])
def test_gap_grows_with_interaction(t, u):
    """At half filling the gap is ``sqrt(U^2 + 16 t^2) - 2t`` ... and it opens with U."""
    weak = lehmann(hubbard_dimer(t, 0.0)).gap
    strong = lehmann(hubbard_dimer(t, u)).gap
    assert strong > weak
    # Non-interacting limit: the gap is the bonding/antibonding splitting.
    assert abs(weak - 2 * t) < 1e-10


def test_spin_symmetry():
    """Alpha and beta Green's functions coincide for this closed-shell singlet."""
    space = hubbard_dimer()
    up = lehmann(space, spin="alpha")
    down = lehmann(space, spin="beta")
    assert np.allclose(np.sort(up.poles), np.sort(down.poles), atol=1e-12)
    freqs = np.linspace(-6, 6, 51)
    assert np.abs(up.spectral(freqs, 0.1) - down.spectral(freqs, 0.1)).max() < 1e-12
