"""casgf -- Green's functions and topological winding numbers from CASSCF active spaces.

Typical use::

    from casgf import ActiveSpace, lehmann

    asp = ActiveSpace.from_arrays(h1, eri, nelecas=8)
    gf = lehmann(asp)
    logdet = gf.log_abs_det(np.linspace(-1, 1, 301), eta=1e-5)
"""

from __future__ import annotations

from .active_space import (
    ActiveSpace,
    active_orbsym,
    build_mol,
    mp2_natural_orbitals,
    normalise_nelec,
    run_casscf,
)
from .chempot import mu_particle_hole, mu_relaxed_casscf
from .ed import Sector, n_determinants, solve_sector, transition_amplitudes
from .greens import Lehmann, lehmann
from .irc import IRCTable, read_geometry_blocks, read_irc_table
from .scan import ScanStep, collect_integrals, irc_scan
from .symmetry import block_indices, block_leakage, irrep_blocks, parity_blocks
from .winding import (
    det_along_contour,
    keyhole_contour,
    min_pole_distance,
    winding_number,
    winding_number_of,
)

__version__ = "0.1.0"

__all__ = [
    "ActiveSpace",
    "IRCTable",
    "Lehmann",
    "ScanStep",
    "Sector",
    "__version__",
    "active_orbsym",
    "block_indices",
    "block_leakage",
    "build_mol",
    "collect_integrals",
    "det_along_contour",
    "irc_scan",
    "irrep_blocks",
    "keyhole_contour",
    "lehmann",
    "min_pole_distance",
    "mp2_natural_orbitals",
    "mu_particle_hole",
    "mu_relaxed_casscf",
    "n_determinants",
    "normalise_nelec",
    "parity_blocks",
    "read_geometry_blocks",
    "read_irc_table",
    "run_casscf",
    "solve_sector",
    "transition_amplitudes",
    "winding_number",
    "winding_number_of",
]
