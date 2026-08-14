"""Generate ``reference.npz``, the numerical regression fixture.

Runs CASSCF on two systems built from coordinates written out in this file, and
stores their active-space integrals together with the Green's-function curve
this package currently produces from them.

The test then rebuilds the curve from the stored integrals, so it is a check
that the numbers do not drift as the code changes -- it does not depend on
CASSCF converging bit-for-bit identically on another machine.

Regenerate (only when the numbers are *meant* to change) with::

    python make_reference.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pyscf import gto

from casgf import ActiveSpace, lehmann

# Idealised D6h benzene: a regular hexagon of carbons with radial C-H bonds.
CC, CH = 1.39, 1.09


def benzene() -> str:
    atoms = []
    for k in range(6):
        angle = 2 * np.pi * k / 6
        c, s = np.cos(angle), np.sin(angle)
        atoms.append(f"C {CC * c:.10f} {CC * s:.10f} 0.0")
        atoms.append(f"H {(CC + CH) * c:.10f} {(CC + CH) * s:.10f} 0.0")
    return "; ".join(atoms)


def h4_rectangle(width: float, height: float = 2.0) -> str:
    """Four hydrogens at the corners of a rectangle, in Bohr.

    ``width == height`` is the square, where the two frontier orbitals become
    degenerate; away from it the rectangle has a gap.
    """
    x, y = width / 2, height / 2
    return "; ".join(
        f"H {sx * x:.10f} {sy * y:.10f} 0.0" for sx, sy in ((1, 1), (-1, 1), (-1, -1), (1, -1))
    )


SYSTEMS = {
    "benzene": dict(atom=benzene(), basis="def2-SVP", unit="A", ncas=6, nelecas=6),
    "h4": dict(atom=h4_rectangle(2.6), basis="6-31g", unit="B", ncas=4, nelecas=4),
}

FREQS = np.linspace(-1.0, 1.0, 301)
ETA = 1e-3


def main(out: Path) -> None:
    payload = {"freqs": FREQS, "eta": ETA}
    for name, spec in SYSTEMS.items():
        mol = gto.M(atom=spec["atom"], basis=spec["basis"], unit=spec["unit"], verbose=0)
        space = ActiveSpace.from_molecule(mol, ncas=spec["ncas"], nelecas=spec["nelecas"])
        gf = lehmann(space)

        payload[f"{name}_h1"] = space.h1
        payload[f"{name}_eri"] = space.eri
        payload[f"{name}_nelecas"] = np.asarray(space.nelecas)
        payload[f"{name}_logdet"] = gf.log_abs_det(FREQS, eta=ETA)
        payload[f"{name}_spectral"] = gf.spectral(FREQS, eta=ETA)
        payload[f"{name}_mu"] = gf.mu
        payload[f"{name}_gap"] = gf.gap
        print(f"{name}: CASSCF {space.meta['e_tot']:.9f} Ha  mu {gf.mu:+.9f}  gap {gf.gap:.9f}")

    np.savez_compressed(out, **payload)
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main(Path(__file__).with_name("reference.npz"))
