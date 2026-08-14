"""The chemical-potential convention, pinned down.

Adding ``mu * N_op`` to the Hamiltonian sends ``E^{(M)} -> E^{(M)} + mu * M``.
An addition energy and a removal energy are both differences between sectors
that differ by exactly one particle, so **both** pick up ``+mu``.

Getting one of those two signs wrong leaves the sum rule, Hermiticity and the
non-interacting limit at ``mu = 0`` all intact, and shows up only as a gap that
is no longer symmetric about ``w = 0`` -- displaced by exactly ``2 * mu``.  That
is what these tests watch for.
"""

import numpy as np
import pytest

from casgf import ActiveSpace, lehmann, mu_particle_hole
from casgf.ed import solve_sector


def test_default_mu_centres_the_gap(interacting_space):
    gf = lehmann(interacting_space)
    lowest_addition = gf.poles_add.min()
    highest_removal = gf.poles_rem.max()
    assert lowest_addition > 0 > highest_removal
    assert abs(lowest_addition + highest_removal) < 1e-10
    assert abs(gf.gap - 2 * lowest_addition) < 1e-10


def test_mu_matches_the_closed_form(interacting_space):
    gf = lehmann(interacting_space)
    na, nb = interacting_space.nelecas
    plus = solve_sector(interacting_space.h1, interacting_space.eri, interacting_space.ncas,
                        (na + 1, nb))
    minus = solve_sector(interacting_space.h1, interacting_space.eri, interacting_space.ncas,
                         (na - 1, nb))
    assert abs(gf.mu - mu_particle_hole(minus.e_gs, plus.e_gs)) < 1e-12


def test_gap_is_independent_of_mu(interacting_space):
    """mu re-references the frequency axis; it cannot change the gap width."""
    reference = lehmann(interacting_space)
    for mu in (-0.7, 0.0, 1.3):
        assert abs(lehmann(interacting_space, mu=mu).gap - reference.gap) < 1e-10


def test_both_branches_shift_together(interacting_space):
    """A change in mu moves addition and removal poles by the same amount.

    This is the test that a sign flip on one branch fails.
    """
    base = lehmann(interacting_space, mu=0.0)
    shifted = lehmann(interacting_space, mu=0.4)
    assert np.allclose(shifted.poles_add - base.poles_add, 0.4, atol=1e-12)
    assert np.allclose(shifted.poles_rem - base.poles_rem, 0.4, atol=1e-12)


def test_with_mu_avoids_rediagonalising(interacting_space):
    """``with_mu`` must give exactly what a fresh build with that mu gives."""
    rebuilt = lehmann(interacting_space, mu=0.4)
    reused = lehmann(interacting_space, mu=0.0).with_mu(0.4)
    assert np.allclose(reused.poles, rebuilt.poles, atol=1e-12)
    assert np.allclose(reused.residues, rebuilt.residues, atol=1e-12)


def test_mu_is_required_when_a_sector_is_missing():
    """A completely filled active space has no N+1 sector, so mu cannot be inferred."""
    norb = 2
    full = ActiveSpace.from_arrays(np.eye(norb), np.zeros((norb,) * 4), nelecas=(norb, norb))
    with pytest.raises(ValueError, match="mu"):
        lehmann(full)

    gf = lehmann(full, mu=0.0)
    assert gf.poles_add.size == 0
    assert gf.poles_rem.size > 0
