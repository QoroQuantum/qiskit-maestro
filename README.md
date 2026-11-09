# qiskit-maestro

[![Tests](https://github.com/QoroQuantum/qiskit-maestro/actions/workflows/tests.yml/badge.svg)](https://github.com/QoroQuantum/qiskit-maestro/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/qiskit-maestro.svg)](https://pypi.org/project/qiskit-maestro/)
[![License: GPL v3+](https://img.shields.io/badge/License-GPLv3%2B-blue.svg)](LICENSE)

Run your Qiskit code on the [Maestro](https://github.com/QoroQuantum/maestro)
quantum circuit simulator by changing one or two lines. You get Maestro's
statevector, matrix product state (MPS), stabilizer, extended stabilizer,
density matrix, tensor network and Pauli propagation methods, on CPU or GPU,
through the Qiskit interfaces you already use: a `BackendV2`, a `SamplerV2`
and an `EstimatorV2`.

## Install

```bash
pip install qiskit-maestro
```

This installs Maestro (`qoro-maestro`) as well. It needs Python 3.10 or later and
works with Qiskit 1.1 and later, including 2.x.

## Switch from Aer

### `AerSimulator` → `MaestroBackend`

```python
from qiskit import QuantumCircuit, transpile
from qiskit_maestro import MaestroBackend   # was: from qiskit_aer import AerSimulator

qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
qc.measure_all()

backend = MaestroBackend()                  # was: AerSimulator()
counts = backend.run(transpile(qc, backend), shots=1000).result().get_counts()
```

`MaestroBackend` has a `Target`, so `transpile` and
`generate_preset_pass_manager(backend=backend)` produce circuits it runs.
Results come back as Aer returns them: counts, `memory=True` for per-shot
bitstrings, and classical registers split by spaces. Qiskit's
`BackendSamplerV2(backend=MaestroBackend())` works too.

Aer's save instructions work as well: `save_statevector`,
`save_statevector_dict`, `save_probabilities`, `save_probabilities_dict`,
`save_expectation_value` and `save_expectation_value_variance`, so
`result.get_statevector()` and `result.data()` return what they return on Aer.

### `StatevectorSampler` or Aer's `SamplerV2` → `MaestroSampler`

```python
from qiskit_maestro import MaestroSampler as Sampler   # was: from qiskit.primitives import StatevectorSampler as Sampler

result = Sampler(seed=1).run([qc], shots=1000).result()
counts = result[0].data.meas.get_counts()
```

PUBs, broadcasting, per-PUB shots and the result shapes are the same as
`StatevectorSampler`'s. Every parameter set of a PUB reuses one converted
circuit, and the sets run in parallel.

### `StatevectorEstimator` or Aer's `EstimatorV2` → `MaestroEstimator`

```python
import numpy as np
from qiskit.circuit.library import efficient_su2
from qiskit.quantum_info import SparsePauliOp
from qiskit_maestro import MaestroEstimator as Estimator   # was: from qiskit.primitives import StatevectorEstimator as Estimator

ansatz = efficient_su2(2, reps=1)
parameter_values = np.random.default_rng(0).uniform(0, np.pi, (10, ansatz.num_parameters))
hamiltonian = SparsePauliOp(["ZZ", "XI"], [1.0, 0.5])
evs = Estimator().run([(ansatz, hamiltonian, parameter_values)]).result()[0].data.evs  # shape (10,)
```

Expectation values are computed from the simulated state, not from shots, for
every method. `precision` works as it does in `StatevectorEstimator`: `0` (the
default) is exact, and a positive value adds Gaussian noise with that standard
deviation, which is then reported in `stds`.

Coming from Aer's primitives? Their `options={"backend_options": {...},
"run_options": {...}}` argument is accepted as is.

### Noise models

Pass an Aer `NoiseModel`, such as one built from a fake device, as you would to
Aer. It works on `MaestroBackend`, `MaestroSampler` and `MaestroEstimator`:

```python
from qiskit import transpile
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
from qiskit_maestro import MaestroBackend               # was: from qiskit_aer import AerSimulator

noise_model = NoiseModel.from_backend(FakeSherbrooke())
backend = MaestroBackend(noise_model=noise_model)      # was: AerSimulator(noise_model=noise_model)
counts = backend.run(transpile(qc, FakeSherbrooke()), shots=4000).result().get_counts()
```

Each error acts where Aer applies it: after the instruction it's attached to
(before it, for `measure`), on that instruction's qubits, so an `rz` with no
error stays noiseless. Readout errors, and the relaxation `from_backend` adds
to delays, apply too. As on Aer, the backend's target then uses the noise
model's basis gates, so `transpile(qc, backend)` produces the gates the noise
is attached to.

The results agree with Aer's: exactly with `method="density_matrix"`, which
applies every channel exactly, and within statistical error on pure-state
methods such as the default statevector, which sample one noise trajectory per
shot. Expectation values on pure-state methods average 1,024 trajectories, as
Aer's `EstimatorV2` does. `method="automatic"` picks the density matrix for
small noisy circuits, as Aer does.

Depolarizing, Pauli, bit- and phase-flip, phase-damping, amplitude-damping,
thermal-relaxation (any T2 ≤ 2·T1) and reset errors are supported, and so are
their compositions on one or two qubits, which covers `from_backend`. Each
translation is checked against Aer's channel. An error Maestro can't reproduce
exactly raises `UnsupportedNoiseError` naming the instruction and qubits, and
no noise is ever dropped. [examples/aer_noise_model.py](examples/aer_noise_model.py)
is a complete Aer script moved over by changing two lines.

## Choose a method and a device

Pass options to `MaestroBackend`, `MaestroSampler` or `MaestroEstimator`.
The names follow Aer's, and Aer's longer MPS option names work as aliases.

```python
from qiskit_maestro import MaestroEstimator

estimator = MaestroEstimator(
    method="matrix_product_state",   # or "mps"
    max_bond_dimension=64,           # Aer: matrix_product_state_max_bond_dimension
    truncation_threshold=1e-10,      # Aer: matrix_product_state_truncation_threshold
)
```

| Option | Values | Default |
|---|---|---|
| `method` | `"statevector"`, `"matrix_product_state"` (`"mps"`), `"stabilizer"`, `"extended_stabilizer"`, `"density_matrix"`, `"tensor_network"`, `"pauli_propagation"`, `"automatic"` | `"statevector"` |
| `device` | `"CPU"`, `"GPU"` | `"CPU"` |
| `precision` | `"double"`, `"single"` | `"double"` |
| `seed_simulator` (`seed` on the primitives) | integer | random |
| `max_bond_dimension` | integer, MPS only | no cap |
| `truncation_threshold` | discarded weight, MPS only | Maestro's default |
| `max_parallel_experiments` | threads for independent circuits and parameter sets; `0` chooses | `0` |
| `maestro_options` | dict of any other [`maestro.SimulatorConfig`](https://qoroquantum.github.io/maestro/api/configuration.html) field | none |
| `noise_model` | a `qiskit_aer.noise.NoiseModel`; see [Noise models](#noise-models) | none |

`"automatic"` takes Aer's name for its default. Noiseless circuits run on the
statevector, Maestro's default. A noisy circuit whose measurements all come at
the end runs on the density matrix when the shots outnumber the 2ⁿ amplitudes
of its n active qubits, as on Aer.

Maestro picks the fastest engine for a circuit by itself, so it may run a
Clifford circuit you sent to `"statevector"` on the stabilizer engine. Results
are unaffected, and each result's metadata records what actually ran.

### Time evolution

`MaestroEstimator` runs time evolution on one persistent simulator, using
Maestro's incremental evolution. If the PUBs of one `run` call are the same
initial circuit followed by 0, 1, 2, ... repetitions of one step, as when each
time point of a Trotter evolution is its own circuit, Maestro simulates the
longest circuit once and reads every time point on the way. Measuring at every
one of N steps then costs N steps instead of about N²/2. Circuits built with
`PauliEvolutionGate(..., synthesis=LieTrotter(reps=k))` are recognised too.

```python
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_maestro import MaestroEstimator

n = 50
step = QuantumCircuit(n)  # One Trotter step of an Ising chain.
for start in (0, 1):
    for q in range(start, n - 1, 2):
        step.rzz(-0.2, q, q + 1)
step.rx(-0.14, range(n))
magnetisation = SparsePauliOp.from_sparse_list([("Z", [q], 1 / n) for q in range(n)], n)
estimator = MaestroEstimator(method="matrix_product_state")

# One circuit per time point, as you'd write it for any estimator.
circuits, circuit = [], QuantumCircuit(n)
for k in range(21):
    circuits.append(circuit.copy())
    circuit.compose(step, inplace=True)
results = estimator.run([(c, magnetisation) for c in circuits]).result()

# Or give the step directly, without building a circuit per time point.
evolution = estimator.run_evolution(QuantumCircuit(n), step, magnetisation, steps=range(21))
evs = evolution.result()[0].data.evs  # One value per step: shape (21,).
```

Each result's metadata says whether incremental evolution was used. Pass
`incremental_evolution=False` to simulate every circuit from the start. It
isn't used for PUBs with parameter arrays, circuits that reset qubits, or the
tensor network and Pauli propagation methods, which gain nothing from it.

### GPU

`device="GPU"` runs on Maestro's GPU backend (statevector, MPS, tensor network
and Pauli propagation). You need GPU access: see
[Get GPU access](https://qoroquantum.github.io/maestro/getting-started/gpu-access.html).
Maestro reads the licence key from `MAESTRO_LICENSE_KEY` itself. If no GPU is
available or the licence isn't accepted, qiskit-maestro raises
`MaestroGpuError`; it never falls back to the CPU silently.

## Bit order

Maestro puts qubit 0 on the left of bitstrings and Pauli strings; Qiskit puts it
on the right. qiskit-maestro converts at the boundary, so everything you see
(counts, memory, `BitArray`s, observables) uses Qiskit's order.

## What's supported, and what isn't yet

Supported:

- Every Qiskit gate. Maestro's 28 native gates run directly, and others (`rzz`,
  `ecr`, `unitary`, `PauliEvolutionGate`, library circuits and so on) are
  expanded through their Qiskit definitions.
- Mid-circuit measurement and `reset`.
- `if_test` with an optional `else`, on one clbit or a register value, with gates
  and measurements in the branches. In Qiskit 1.x, `c_if` as well.
- `for_loop`, unrolled when the circuit is converted. Its index set is fixed
  when the circuit is built, so every loop Qiskit can express unrolls, unless
  its body uses `break` or `continue`.
- `switch`, on a clbit, a register or a cast of one, rewritten as one `if` block
  per case value. A default case becomes a block for each value no case names,
  up to 64 of them.
- Aer noise models: see [Noise models](#noise-models).
- Aer's `save_statevector`, `save_statevector_dict`, `save_probabilities`,
  `save_probabilities_dict`, `save_expectation_value` and
  `save_expectation_value_variance`, anywhere before the first measurement, on
  `MaestroBackend`. With noise or a reset before it, a saved statevector is one
  random trajectory's, as on Aer; saved probabilities and expectation values
  are averaged over the shots, or exact on the density matrix.
- `method="automatic"`, as described under [Choose a method and a device](#choose-a-method-and-a-device).
- Parameterised circuits, with Qiskit's broadcasting rules.
- `SparseObservable` projector terms (`0`, `1`, `+`, `-`, `r`, `l`) in the
  Estimator (Qiskit 2.x).

Not supported yet:

- **Some noise.** Coherent (unitary) errors, general Kraus channels, errors on
  three or more qubits, correlated readout errors on several qubits, and Aer's
  nonlocal errors raise `UnsupportedNoiseError`. So does noise on a circuit
  with classical control, other than readout errors, and noise in
  `run_evolution`. Relaxation can't run on the stabilizer method.
- **Some classical control.** `while_loop`, `break`, `continue`, and classical
  expressions beyond "bit is set" and "register equals value" aren't supported.
  Casts are read as Qiskit does: a cast to a narrower `Uint` keeps the low bits.
  A cast of a multi-bit register to `Bool` (a non-zero test) isn't supported.
  Neither are nested `if_test` blocks (including a `switch` inside one), a
  `reset` inside `if_test`, an `else` branch on a multi-bit condition, or a
  `switch` case that measures into the switch's own target. Each raises
  `UnsupportedInstructionError` naming the instruction.
- **Measurements in Estimator circuits.** As in `StatevectorEstimator`, they
  raise an error. Resets are allowed: statevector and MPS follow one random
  outcome, as `StatevectorEstimator` does, and density matrix is exact.
- **Other save instructions**, such as `save_density_matrix`, `save_amplitudes`
  and `save_state`, and saves that are per shot, conditional, unnormalized, after
  a measurement or inside control flow. The Sampler and Estimator don't take
  save instructions.
- **Native `unitary` gates.** Maestro's core has generic one-, two- and
  three-qubit gates, but its circuit API doesn't expose them yet, so a
  `UnitaryGate` is expanded through its Qiskit definition.
- **Aer methods without a Maestro equivalent**: `unitary`, `superop`.
- **`dt` timing.** Delays are dropped, since they don't change a noiseless
  simulation. With a noise model whose delays relax, they apply that relaxation.
- **Shot order.** Maestro returns counts. Per-shot `memory` and `BitArray`
  shots are those counts in random order. That's exact in distribution, since
  shots are independent, but it isn't the order Maestro simulated them in.

## Links

- [Maestro documentation](https://qoroquantum.github.io/maestro/), including
  [Coming from Qiskit Aer](https://qoroquantum.github.io/maestro/getting-started/coming-from-qiskit.html)
- [Example: a VQE loop on Aer and on Maestro MPS](examples/vqe_aer_vs_maestro.ipynb)
- [Notes on the Maestro Python API](docs/maestro-api-notes.md), for contributors
- [Contributing](CONTRIBUTING.md) and [Code of Conduct](CODE_OF_CONDUCT.md)

## License

GPL-3.0-or-later. See [LICENSE](LICENSE).
