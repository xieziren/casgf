# casgf

One-particle Green's functions, spectral functions and topological winding numbers from
CASSCF active spaces — by exact diagonalisation, in pure Python.

PySCF will give you a converged CASSCF wavefunction. Getting from there to `G(ω)` means
diagonalising the `N±1` particle-number sectors, projecting creation and annihilation
operators onto their eigenstates, choosing a chemical potential and keeping every sign
straight. `casgf` does that, and gives you the things you actually want to look at:

```python
gf.spectral(freqs, eta)        # -Im Tr G, the density of states
gf.log_abs_det(freqs, eta)     # poles *and* zeros of det G
winding_number_of(gf, contour) # the topological invariant, as an integer
```

## Why exact diagonalisation

Active spaces are small — CAS(4,4) is 36 determinants, CAS(8,8) is 4900 — so each sector
can simply be diagonalised in full. That is more than affordable, and unlike an iterative
solver it hands you *every* eigenpair, which is exactly what the Lehmann representation
needs:

```
G_ij(z) = Σ_n residues[i,n] · residues[j,n] / (z − poles[n])
```

Once that is built, evaluating `G` at any frequency is a rational sum. A 301-point spectrum
and a 30 000-point contour integral cost about the same, which is what makes winding-number
contours practical rather than aspirational.

## Install

```bash
pip install -e ".[dev,plot]"
```

Python ≥ 3.10, NumPy, SciPy and PySCF. No compiler, no external solver.

## Quickstart

```python
import numpy as np
from pyscf import gto
from casgf import ActiveSpace, lehmann

mol = gto.M(atom="H 0 0 0; H 0 0 1.4; H 0 0 2.8; H 0 0 4.2", basis="6-31g", unit="B")
gf = lehmann(ActiveSpace.from_molecule(mol, ncas=4, nelecas=4))

freqs = np.linspace(-1, 1, 401)
spectrum = gf.spectral(freqs, eta=0.01)
determinant = gf.log_abs_det(freqs, eta=0.01)

print(gf.gap, gf.mu, gf.sum_rule())   # gap, chemical potential, spectral weight
```

`ActiveSpace.from_molecule` runs RHF → MP2 natural orbitals → CASSCF and extracts the
active-space integrals. If you already have integrals — from an earlier scan, or from
another program — feed them straight in:

```python
space = ActiveSpace.from_arrays(h1, eri, nelecas=8)   # eri in chemist notation
```

### Spectral function or determinant?

`spectral` peaks at the **poles** of `G`. `log_abs_det` peaks at those same poles and also
dips at the **zeros** of `det G`. Zeros carry no spectral weight, so the density of states
is blind to them — but they are half of what fixes the topology of `G`, and they exist only
because of the two-electron interaction. See
[`examples/01_spectral_function.ipynb`](examples/01_spectral_function.ipynb).

### Symmetry blocks and winding numbers

When the active orbitals fall into irreducible representations that do not mix, `G` is
block diagonal, `det G` factorises, and each block carries its own winding number:

```python
from casgf import block_leakage, irrep_blocks, keyhole_contour, winding_number_of

blocks = irrep_blocks(space.orbsym, mol)
contour = keyhole_contour(radius=1.5, n_per_segment=10_000)

for name, orbitals in blocks.items():
    print(name, winding_number_of(gf, contour, orbitals=orbitals))
```

By the argument principle that integer is (zeros − poles) of `det G` enclosed by the
contour, so it cannot change continuously — only by a pole and a zero colliding.

`block_leakage` reports how far `G` actually is from block diagonal, which is worth
measuring rather than assuming: a geometry that is only approximately symmetric gets
assigned to C1, and then no symmetry adaptation happens at all. Blocks can also be taken by
index parity (`"even"` / `"odd"`) when point-group symmetry is unavailable.

### Along a reaction path

```python
from casgf import collect_integrals, irc_scan, read_geometry_blocks

geometries = read_geometry_blocks("path_clean_awk")
data = collect_integrals(irc_scan(geometries, ncas=8, nelecas=8))
np.save("h1.npy", data["h1"])
```

`irc_scan` carries the converged orbitals from one geometry to the next with
`mcscf.project_init_guess`, which is what keeps the active space continuous along the path.
Readers for `--`-separated geometry blocks and for GAMESS IRC tables are in `casgf.irc`.

## Command line

```bash
casgf gf --h1 h1.npy --eri eri.npy --index 0 --nelec 8 --eta 1e-5 --out curve.npy
casgf winding --h1 h1.npy --eri eri.npy --index 0 --nelec 8 --block blocks
casgf scan --geometries path_clean_awk --ncas 8 --nelec 8 --out scan/
```

## Examples

The three run through one system: **cyclobutadiene**, whose automerization — the two double
bonds trading places through a square transition state — is the textbook
orbital-symmetry-controlled process, and whose antiaromatic four-electron $\pi$ system is a
standard multireference test case.

| | what it does | cost |
| --- | --- | --- |
| [`01_spectral_function.ipynb`](examples/01_spectral_function.ipynb) | spectral function vs `log\|det G\|` for cyclobutadiene CAS(4,4) and benzene CAS(6,6), with and without the interaction | seconds |
| [`02_symmetry_and_winding.ipynb`](examples/02_symmetry_and_winding.ipynb) | D2h irrep blocking, leakage, determinant factorisation and per-block winding numbers on the rectangle — then the square, where the frontier pair goes degenerate and the blocking stops being defined at all | seconds |
| [`03_green_function_map.py`](examples/03_green_function_map.py) | the whole automerization path: energy profile (barrier ≈ 5.8 kcal/mol) and `log\|det G\|` as an image | ~30 s |

Every example generates its own geometries from bond lengths written in the file; nothing
depends on an external data file.

## Correctness

104 tests, under a second, run on Python 3.10/3.11/3.12 in CI. What they actually pin down:

- **Non-interacting limit** — with `eri = 0` the answer is known in closed form,
  `G(z) = ((z − μ)I − h1)⁻¹`. Reproduced to 1e-12. An error anywhere in the sector
  diagonalisation, the transition amplitudes, the pole bookkeeping or the chemical-potential
  shift breaks this.
- **Two independent implementations** — [`casgf.reference`](src/casgf/reference.py) computes
  the same `G` by inverting the resolvent directly, never forming a pole or a residue, with
  the reference state from a Davidson solve rather than a dense one. Nothing but the
  Hamiltonian is shared between the two routes, and they agree to 1e-10 on random,
  non-interacting, open-shell and real CASSCF Hamiltonians alike.
- **Hubbard dimer** — poles checked against the analytic two-site result.
- **Sum rules** — total spectral weight equals the orbital count to 1e-10; `G(ω*)† = G(ω)`;
  the spectral function is non-negative and correctly normalised.
- **Chemical potential** — the default choice centres the gap exactly, which is what pins
  the sign convention discussed in [`docs/theory.md`](docs/theory.md).
- **Argument principle** — winding numbers reproduce the known values for `z`, `1/z`, `z²`,
  and add up correctly over symmetry blocks.
- **Regression** — stored curves for benzene CAS(6,6)/def2-SVP and rectangular H4
  CAS(4,4)/6-31G, so the numbers cannot drift unnoticed.

```bash
pytest -q
```

## Documentation

[`docs/theory.md`](docs/theory.md) covers the active-space Hamiltonian, the Lehmann
representation, both chemical-potential conventions (including the sign that is easy to get
wrong), symmetry blocking, and the winding number.

## License

MIT — see [`LICENSE`](LICENSE).
