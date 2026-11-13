# Noise models

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

[The Aer noise-model example](../examples/aer_noise_model.md) is a complete
Aer script moved over by changing two lines.

## How noise is applied

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

## Supported errors

Depolarizing, Pauli, bit- and phase-flip, phase-damping, amplitude-damping,
thermal-relaxation (any T2 ≤ 2·T1) and reset errors are supported, and so are
their compositions on one or two qubits, which covers `from_backend`. Each
translation is checked against Aer's channel. An error Maestro can't reproduce
exactly raises `UnsupportedNoiseError` naming the instruction and qubits, and
no noise is ever dropped. [Supported features](supported.md#what-isnt-supported-yet)
lists the noise that isn't supported yet.
