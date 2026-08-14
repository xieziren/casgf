"""Extracting an active space from a real CASSCF calculation.

These run PySCF on a tiny system, so they stay in the fast suite while still
covering the path from a molecule to ``h1``/``eri``.
"""

import numpy as np
import pytest
from pyscf import gto

from casgf import ActiveSpace, lehmann, normalise_nelec, run_casscf
from casgf.ed import n_determinants, solve_sector


@pytest.fixture(scope="module")
def h4_casscf():
    """A stretched H4 chain: small, but genuinely multireference."""
    mol = gto.M(
        atom="H 0 0 0; H 0 0 1.4; H 0 0 2.8; H 0 0 4.2",
        basis="sto-3g",
        unit="B",
        verbose=0,
    )
    return run_casscf(mol, ncas=4, nelecas=4)


def test_normalise_nelec():
    assert normalise_nelec(6) == (3, 3)
    assert normalise_nelec(7) == (4, 3)
    assert normalise_nelec((5, 3)) == (5, 3)
    with pytest.raises(ValueError, match="does not fit"):
        normalise_nelec(11, ncas=4)


def test_active_space_shapes(h4_casscf):
    space = ActiveSpace.from_casscf(h4_casscf)
    assert space.ncas == 4
    assert space.h1.shape == (4, 4)
    assert space.eri.shape == (4, 4, 4, 4)
    assert space.nelecas == (2, 2)
    assert space.nelec_total == 4


def test_ed_reproduces_the_casscf_total_energy(h4_casscf):
    """``E_active(ground state) + e_core`` must be the CASSCF total energy.

    This is what certifies that ``h1``, ``eri`` and ``e_core`` were extracted
    with mutually consistent conventions.
    """
    space = ActiveSpace.from_casscf(h4_casscf)
    sector = solve_sector(space.h1, space.eri, space.ncas, space.nelecas)
    assert sector.e_gs + space.e_core == pytest.approx(h4_casscf.e_tot, abs=1e-8)


def test_green_function_from_a_molecule(h4_casscf):
    space = ActiveSpace.from_casscf(h4_casscf)
    gf = lehmann(space)
    assert abs(gf.sum_rule() - 4) < 1e-10
    assert gf.gap > 0


def test_eri_is_restored_to_four_index():
    """Packed integrals must be accepted and unpacked."""
    from pyscf import ao2mo

    full = np.zeros((4, 4, 4, 4))
    for i in range(4):
        full[i, i, i, i] = 1.5
    packed = ao2mo.restore(4, full, 4)
    space = ActiveSpace.from_arrays(np.eye(4), packed, nelecas=4)
    assert space.eri.shape == (4, 4, 4, 4)
    assert np.allclose(space.eri, full)


def test_determinant_counting():
    assert n_determinants(4, (2, 2)) == 36
    assert n_determinants(8, (4, 4)) == 4900
    assert n_determinants(8, (5, 4)) == 3920


def test_oversized_sector_is_refused():
    with pytest.raises(ValueError, match="max_dets"):
        solve_sector(np.eye(10), np.zeros((10,) * 4), 10, (5, 5), max_dets=100)


def test_bad_electron_count_is_refused():
    with pytest.raises(ValueError, match="out of range"):
        solve_sector(np.eye(4), np.zeros((4,) * 4), 4, (5, 2))
