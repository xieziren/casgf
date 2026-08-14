"""Numerical regression on real CASSCF active spaces.

``tests/data/reference.npz`` holds the active-space integrals of two converged
CASSCF calculations -- benzene CAS(6,6)/def2-SVP and a rectangular H4
CAS(4,4)/6-31G -- together with the curves this package produced from them.
Both systems are built from coordinates written out in ``make_reference.py``,
so the fixture is self-contained.

Storing the *integrals* rather than the geometries keeps this a test of the
Green's-function code: it cannot fail because CASSCF converged to a slightly
different point on another machine or PySCF release.

These also carry the cross-check against :mod:`casgf.reference` onto systems
with real, structured Hamiltonians rather than random ones.
"""

from pathlib import Path

import numpy as np
import pytest

from casgf import ActiveSpace, lehmann
from casgf.reference import green_function_by_resolvent

DATA = Path(__file__).parent / "data" / "reference.npz"
SYSTEMS = ["benzene", "h4"]


@pytest.fixture(scope="module")
def reference():
    if not DATA.exists():
        pytest.skip(f"{DATA.name} is missing; regenerate it with tests/data/make_reference.py")
    return np.load(DATA)


@pytest.fixture(scope="module", params=SYSTEMS)
def system(request, reference):
    name = request.param
    space = ActiveSpace.from_arrays(
        reference[f"{name}_h1"],
        reference[f"{name}_eri"],
        nelecas=tuple(reference[f"{name}_nelecas"]),
    )
    return {"name": name, "space": space, "gf": lehmann(space)}


def test_log_abs_det_is_unchanged(system, reference):
    freqs, eta = reference["freqs"], float(reference["eta"])
    computed = system["gf"].log_abs_det(freqs, eta=eta)
    assert np.abs(computed - reference[f"{system['name']}_logdet"]).max() < 1e-10


def test_spectral_function_is_unchanged(system, reference):
    freqs, eta = reference["freqs"], float(reference["eta"])
    computed = system["gf"].spectral(freqs, eta=eta)
    assert np.abs(computed - reference[f"{system['name']}_spectral"]).max() < 1e-10


def test_chemical_potential_and_gap_are_unchanged(system, reference):
    gf = system["gf"]
    assert gf.mu == pytest.approx(float(reference[f"{system['name']}_mu"]), abs=1e-10)
    assert gf.gap == pytest.approx(float(reference[f"{system['name']}_gap"]), abs=1e-10)


def test_sum_rule(system):
    assert abs(system["gf"].sum_rule() - system["space"].ncas) < 1e-9


def test_gap_is_centred(system):
    gf = system["gf"]
    assert abs(gf.poles_add.min() + gf.poles_rem.max()) < 1e-10


def test_matches_the_independent_resolvent(system):
    """The two implementations agree on a structured, physical Hamiltonian."""
    z = np.array([-0.9 + 0.05j, -0.3 + 0.1j, 0.0 + 0.1j, 0.35 + 0.05j, 1.1 + 0.2j])
    fast = system["gf"].at_complex(z)
    slow = green_function_by_resolvent(system["space"], z)
    assert np.abs(fast - slow).max() < 1e-10
