# Reference data

`reference.npz` is the regression fixture for `test_regression.py`. It holds, for two
systems:

| key | contents |
| --- | --- |
| `<name>_h1`, `<name>_eri`, `<name>_nelecas` | converged CASSCF active-space integrals |
| `<name>_logdet`, `<name>_spectral` | the curves this package produces from them |
| `<name>_mu`, `<name>_gap` | chemical potential and fundamental gap |
| `freqs`, `eta` | the frequency grid and broadening used |

The two systems are **benzene** at an idealised D6h geometry, CAS(6,6)/def2-SVP, and a
rectangular **H4**, CAS(4,4)/6-31G. Both geometries are generated from coordinates written
out in `make_reference.py`, so nothing here depends on an external data file.

Storing the *integrals* rather than the geometries is deliberate: the test then checks the
Green's-function code and cannot fail merely because CASSCF converged to a slightly
different point on another machine or PySCF release.

Regenerate — only when the numbers are meant to change — with:

```bash
python make_reference.py
```
