# Qiskit Ecosystem submission

Submitted via [Issue #1449](https://github.com/Qiskit/ecosystem/issues/1449) and tracked in [PR #1450](https://github.com/Qiskit/ecosystem/pull/1450).

## Before submitting

- [x] The repository `github.com/QoroQuantum/qiskit-maestro` is public, with
      this code on `main`.
- [x] `qiskit-maestro` is published on PyPI (v0.1.0 is live at https://pypi.org/project/qiskit-maestro/).
- [x] The Tests workflow is green on `main` for Python 3.10 to 3.13 with Qiskit
      1.x and 2.x.
- [x] The Maestro docs with the updated "Coming from Qiskit Aer" page are live at
      https://qoroquantum.github.io/maestro/getting-started/coming-from-qiskit.html.
- [x] Someone at Qoro owns the issue and will answer reviewers' questions.

## How the project meets the criteria

| Criterion | Evidence |
|---|---|
| Builds on, interfaces with, or extends Qiskit | `MaestroBackend` is a `BackendV2` with a `Target`; `MaestroSampler` and `MaestroEstimator` implement `BaseSamplerV2` and `BaseEstimatorV2`. |
| Compatible with Qiskit 2.0 or newer | CI tests the latest Qiskit 2.x and 1.x on Python 3.10 to 3.13, and the oldest supported version, 1.1. `pyproject.toml` declares `qiskit>=1.1,<3`. |
| OSI-approved licence | GPL-3.0-or-later, with SPDX headers in every source file. |
| Follows Qiskit's Code of Conduct | `CODE_OF_CONDUCT.md` is the Contributor Covenant 2.1, which Qiskit's code of conduct is also based on. Contributors also follow Qiskit's. |
| Maintainer activity within six months | First release in November 2026, with Maestro itself released regularly (0.3.10 in October 2026). |

## Form answers

**Project name**

```text
qiskit-maestro
```

**Description** (under 135 characters)

```text
Run Qiskit circuits, Sampler and Estimator on the Maestro simulator: statevector, MPS, stabilizer and more, on CPU or GPU.
```

**Contact email**

```text
team@qoroquantum.de
```

**Category**: Circuit simulator

**Labels** (up to five): tensor networks, GPU, HPC, utility-scale, Hamiltonian
simulation

**Interface/API**: Python

**Stability and support expectations**: experimental

The first release has a complete, tested Qiskit surface, but it is 0.x, and
its option names may still change as the Maestro core adds features (see
`docs/maestro-api-notes.md`). Change this to production-ready after the first
minor release without breaking changes.

**GitHub repository**

```text
https://github.com/QoroQuantum/qiskit-maestro
```

**Home page**

```text
https://maestro.qoroquantum.net
```

**Documentation**

```text
https://qoroquantum.github.io/maestro/getting-started/coming-from-qiskit.html
```

**Packages**

```text
https://pypi.org/project/qiskit-maestro
https://github.com/QoroQuantum/qiskit-maestro/blob/main/pyproject.toml
```

**Reference paper**: none. Maestro's `CITATION.cff` cites the software, not a
paper. Leave the field empty unless a paper is out by the time this is
submitted.

**Code of Conduct**: tick "I agree to follow the Qiskit Code of Conduct".

## Optional comment for reviewers

```text
qiskit-maestro lets Qiskit users switch from Aer to the Maestro simulator by
changing an import: MaestroBackend (BackendV2), MaestroSampler and
MaestroEstimator (V2 primitives). It exposes Maestro's statevector, matrix
product state, stabilizer, extended stabilizer, density matrix, tensor network and Pauli propagation
methods, and its licensed GPU backend. Results use Qiskit's bit order, and the
tests compare them with StatevectorSampler, StatevectorEstimator and Aer on
Qiskit 1.x and 2.x. docs/guide.md lists what isn't supported yet.
```
