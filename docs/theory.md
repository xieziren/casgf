# Theory and conventions

Enough to read the code and to know what the numbers mean.

## 1. The active-space Hamiltonian

A CASSCF calculation splits the orbitals into a frozen core, an active space of `ncas`
orbitals holding `nelecas` electrons, and empty virtuals. Everything here lives in the
active space:

$$
H = \sum_{ij\sigma} h_{ij}\, c^{\dagger}_{i\sigma} c_{j\sigma}
  + \tfrac{1}{2} \sum_{ijkl}\sum_{\sigma\sigma'} (ij|kl)\,
    c^{\dagger}_{i\sigma} c^{\dagger}_{k\sigma'} c_{l\sigma'} c_{j\sigma}
$$

`h1` is PySCF's `h1e_for_cas`, which already folds in the mean field of the frozen core,
and `eri` is `(ij|kl)` in **chemist notation**, the full four-index array
(`ao2mo.restore(1, mc.get_h2eff(), ncas)`). `ActiveSpace` carries exactly these two arrays
plus `nelecas`.

`h1e_for_cas` also returns a constant `e_core` — the frozen-core plus nuclear-repulsion
energy. It is stored on `ActiveSpace` but never enters the Green's function: it is the
*same* constant in the `N`, `N+1` and `N-1` sectors, so it cancels from every energy
difference below. It is needed only to reconstruct total energies
(`test_active_space.py` checks `E_gs + e_core == mc.e_tot`).

The active orbitals are natural orbitals (`mc.natorb = True`). That matters: `G` is a
matrix *in the active-orbital basis*, so the basis has to be defined reproducibly for the
matrix elements — and hence `det G` — to mean anything.

## 2. Full diagonalisation, and why it is enough

`casgf` diagonalises each particle-number sector **completely**, with dense `eigh` on the
FCI Hamiltonian from `pyscf.fci.direct_spin1.pspace`:

| active space | sector | determinants | dense `eigh` |
| --- | --- | --- | --- |
| CAS(4,4) | (2,2) | 36 | instant |
| CAS(6,6) | (3,3) | 400 | instant |
| CAS(8,8) | (4,4) | 4 900 | ~7 s |
| CAS(6,10) | (3,3) | 14 400 | ~1 GB, minutes — out of scope |

Iterative solvers get the ground state faster, but the Lehmann representation needs *every*
eigenpair of the `N±1` sectors, which is precisely what a dense diagonalisation hands you
for free. `solve_sector` refuses anything above `max_dets = 20000` rather than quietly
starting an unaffordable job.

## 3. Lehmann representation

For the spin-up Green's function, with `|0>` the `N`-electron ground state:

$$
G_{ij}(z) = \sum_{n} \frac{\langle 0|c_i|n\rangle\langle n|c^{\dagger}_j|0\rangle}
                          {z - (E^{N+1}_n - E^{N}_0 + \mu)}
          + \sum_{m} \frac{\langle 0|c^{\dagger}_j|m\rangle\langle m|c_i|0\rangle}
                          {z - (E^{N}_0 - E^{N-1}_m + \mu)}
$$

Both branches have the same shape — a residue matrix over a pole — so `Lehmann` stores
them in one array, addition poles first:

```
G_ij(z) = sum_n residues[i, n] * residues[j, n] / (z - poles[n])
```

The residues are real because the CI vectors are real. `n_add` marks where the addition
branch ends.

Building this costs three diagonalisations. Evaluating it afterwards is a rational sum, so
a 301-point spectrum and a 30 000-point contour cost essentially the same as each other —
which is what makes the winding-number contours practical at all.

The amplitudes come from `pyscf.fci.addons.cre_a` / `des_a` applied to the ground-state CI
vector and projected onto the `N±1` eigenvectors. Their overall sign convention is
irrelevant: flipping the sign of orbital `i` sends `G -> S G S` with `S` diagonal and
`S² = 1`, which leaves `det G` untouched.

## 4. The chemical potential — and the sign that is easy to get wrong

The frequency axis is referenced to a chemical potential, added to the Hamiltonian as
`H -> H + mu * N_op`. That shifts every sector rigidly:

$$
E^{(M)} \longrightarrow E^{(M)} + \mu M
$$

An **addition** energy is `E^{(N+1)} - E^{(N)}`; a **removal** energy is
`E^{(N)} - E^{(N-1)}`. Both are differences between sectors that differ by exactly one
particle, so **both pick up `+mu`** — not `+mu` and `-mu`.

This is the one trap in the whole calculation. A sign error on the removal branch leaves
the sum rule exact, Hermiticity exact, and the non-interacting limit at `mu = 0` exact; the
only symptom is a gap displaced by `2 * mu` instead of centred on `ω = 0`. `test_chempot.py`
exists to catch precisely that, and the independent implementation in `casgf.reference`
catches it a second way.

The same statement on the resolvent side: there the shift appears as `z - mu` in **both**
branches,

```
(z - mu + E_0 - H_{N+1}) x = c†|0>        (particle)
(z - mu - E_0 + H_{N-1}) y = c|0>         (hole)
```

which is a useful cross-check, because the two forms look different but must agree.

Two ways to choose `mu`:

- **`mu_particle_hole`** (default): `mu = (E^{N-1}_0 - E^{N+1}_0) / 2` from the ground
  states of the *same* active-space Hamiltonian. Free once the sectors are diagonalised,
  and it puts the lowest addition pole at `+gap/2` and the highest removal pole at
  `-gap/2` exactly.
- **`mu_relaxed_casscf`**: the same formula with `E^{N±1}` from separate CASSCF runs on the
  cation and anion, so the ions' orbitals relax. Two extra CASSCF calculations, no longer
  exactly gap-centring, but it is the true IP/EA midpoint.

Since `mu` only shifts poles, `Lehmann.with_mu` switches between them without
re-diagonalising anything.

## 5. What gets plotted

Two different quantities, and the difference is the whole point:

- **Spectral function** `A(ω) = -Im Tr G(ω + iη)` — the usual density of states. Peaks at
  the poles of `G`. Computed straight from the pole weights, so it is cheap on dense grids.
- **`log |det G(ω + iη)|`** — peaks at the **poles** of `G` *and* dips at the **zeros** of
  `det G`. The zeros carry no spectral weight and are invisible in `A(ω)`, but they are
  half of the topological content: a winding number changes only when a pole and a zero
  exchange places.

Without the two-electron interaction, `det G` is a product of `1/(z - ε_k)` and has no
zeros at all — so every zero is produced by correlation.
[`examples/02_symmetry_and_winding.ipynb`](../examples/02_symmetry_and_winding.ipynb)
counts them explicitly and gets exactly zero in the non-interacting case.

Small `η` makes these maps very sharp: with `η = 1e-5` on a 301-point grid over `[-1, 1]`,
most sample points sit far from any pole and the ones that do not shoot up by an order of
magnitude. That is a feature when the goal is to locate poles and zeros, and a nuisance if
you wanted a smooth curve.

## 6. Symmetry blocks

When the active orbitals fall into irreducible representations that do not mix, `G` is
block diagonal and `det G` factorises. The winding number is then defined per block, and
block winding numbers add up to the total (`test_symmetry.py`).

Two ways to find the blocks:

- **`irrep_blocks(orbsym)`** — group orbitals by point-group irrep. This is the meaningful
  definition, and needs the CASSCF to have been run with `mol.symmetry = True`.
- **`parity_blocks(ncas)`** — even- and odd-numbered active orbitals, i.e. `G[::2, ::2]`
  and `G[1::2, 1::2]`. A fallback for when point-group symmetry is unavailable. It gives
  the right answer only if the orbitals happen to come out alternating between two irreps;
  it encodes no symmetry information of its own.

### How exact is the blocking, really?

Worth checking rather than assuming, because it is easy for it to be *nearly* true.

PySCF assigns a point group from the geometry with a tolerance. A structure that is only
approximately symmetric — an optimised path traced by another program, say, where the
symmetry axis is respected to a few times `1e-2` Bohr rather than to numerical precision —
is classified as **C1**, and then `mol.symmetry = True` does nothing at all. The orbitals
may still carry recognisable symmetry character, and index-parity blocking may still work,
but "may" is doing real work in that sentence.

`block_leakage(G, blocks)` puts a number on it: the largest
`|G_ij| / sqrt(|G_ii| |G_jj|)` connecting different blocks. It is exactly zero when `G` is
block diagonal and `det G` factorises exactly. On a properly symmetric geometry it comes
out at `1e-13`; on a nearly-symmetric one, `1e-3` to `1e-2` — small enough that the block
winding numbers are unambiguous, but large enough to be worth reporting next to them.

## 7. The winding number

The contour is the boundary of the right half-disc of radius `r` in the complex frequency
plane, traversed anticlockwise, in three segments:

1. the lower-right quarter arc, `-ir → +r`;
2. the upper-right quarter arc, `+r → +ir`;
3. straight down the imaginary axis, `+ir → -ir`.

By the argument principle, the number of times `det G(z)` circles the origin along this
loop is

```
winding number = (zeros of det G inside) - (poles of det G inside)
```

an integer that cannot change continuously. Only poles carrying spectral weight count: most
of the `N±1` spectrum is orthogonal to `c†|0>` and never appears in `G` at all.

`winding_number` gets the integer by unwrapping `arg det G` along the contour and dividing
the total change by `2π`; it warns if the accumulated phase is not close to an integer,
which is what happens if the contour is sampled too coarsely or passes too near a pole.
`min_pole_distance` reports the clearance so this can be checked in advance.

Note the origin lies *on* the contour, on segment 3. That is harmless whenever `mu` puts
`ω = 0` in the middle of a gap, where `det G` is finite and non-zero; `winding_number`
raises if `det G` vanishes exactly on the contour.

## 8. Two implementations

`casgf.lehmann` is the one to use. `casgf.reference.green_function_by_resolvent` computes
the same object by inverting the resolvent at each frequency, and exists only to disagree
with it if something is wrong:

| | `lehmann` | `reference` |
| --- | --- | --- |
| reference state | dense `eigh` on the `N` sector | PySCF Davidson |
| `N±1` sectors | full spectrum, all eigenvectors | never diagonalised |
| `G` | sum over poles and residues | one dense complex solve per frequency |
| cost | three diagonalisations, then nearly free per frequency | linear in the number of frequencies, with a large constant |

They share the Hamiltonian construction and nothing else, so agreement is evidence about
the physics rather than about a shared code path. `tests/test_reference.py` and
`tests/test_regression.py` check it on random, non-interacting, open-shell and real CASSCF
Hamiltonians, to 1e-10.
