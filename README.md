# qiskit-maestro

[![Tests](https://github.com/QoroQuantum/qiskit-maestro/actions/workflows/tests.yml/badge.svg)](https://github.com/QoroQuantum/qiskit-maestro/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/qiskit-maestro.svg)](https://pypi.org/project/qiskit-maestro/)
[![License: GPL v3+](https://img.shields.io/badge/License-GPLv3%2B-blue.svg)](https://github.com/QoroQuantum/qiskit-maestro/blob/main/LICENSE)

Run your Qiskit code on the [Maestro](https://github.com/QoroQuantum/maestro)
quantum simulator, on your machine or in the Qoro cloud, by changing one import.

- **Drop-in for Aer**: a `BackendV2`, a `SamplerV2` and an `EstimatorV2`, with
  Aer's option names, noise models and save instructions.
- **Beyond statevector**: matrix product state (MPS) for hundreds of qubits,
  plus stabilizer, density matrix, tensor network and Pauli propagation.
- **CPU, GPU or cloud**: run locally, on your own GPU, or on Qoro's GPUs with
  no hardware of your own.

## Install

```bash
pip install qiskit-maestro            # local simulation, Python 3.10+
pip install "qiskit-maestro[cloud]"   # adds the Qoro cloud, Python 3.11+
```

## Quick start

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

The Sampler and Estimator swap the same way:

| You use | Switch to |
|---|---|
| `AerSimulator` | `MaestroBackend` |
| `StatevectorSampler`, Aer's `SamplerV2` | `MaestroSampler` |
| `StatevectorEstimator`, Aer's `EstimatorV2` | `MaestroEstimator` |

Results come back in Qiskit's usual form and bit order.

## Run on the Qoro cloud

1. Sign up at [dash.qoroquantum.net](https://dash.qoroquantum.net) to get an
   API key. New accounts get 1,000 free tokens, with no credit card.
2. Install the cloud extra and set your key:

   ```bash
   pip install "qiskit-maestro[cloud]"
   export QORO_API_KEY=...   # or put it in a .env file
   ```

3. Pass `service=` to the backend, Sampler or Estimator. Nothing else changes:

   ```python
   from divi.backends import QoroService
   from qiskit_maestro import MaestroBackend

   backend = MaestroBackend(service=QoroService(), method="mps")
   counts = backend.run(transpile(qc, backend), shots=1000).result().get_counts()
   ```

Add `device="GPU"` to run on Qoro's GPU cluster.
[examples/qoro_cloud.py](https://github.com/QoroQuantum/qiskit-maestro/blob/main/examples/qoro_cloud.py)
runs all three on the cloud. A few features, such as noise models and classical
control, run locally only for now: see the
[guide](https://github.com/QoroQuantum/qiskit-maestro/blob/main/docs/guide.md#qoro-cloud).

## Pick a simulation method

```python
backend = MaestroBackend(method="mps", max_bond_dimension=64)
```

| `method` | Use it for |
|---|---|
| `"statevector"` (default) | exact results; memory doubles with each qubit |
| `"mps"` | many qubits with limited entanglement |
| `"stabilizer"` | Clifford circuits, any size |
| `"density_matrix"` | small noisy circuits, exact |
| `"automatic"` | let Maestro pick, as Aer does |

Pass an Aer `NoiseModel` as `noise_model=`, and `device="GPU"` for your own GPU
([get GPU access](https://qoroquantum.github.io/maestro/getting-started/gpu-access.html)).
The [guide](https://github.com/QoroQuantum/qiskit-maestro/blob/main/docs/guide.md#options)
lists every method and option.

## Learn more

- [Guide](https://github.com/QoroQuantum/qiskit-maestro/blob/main/docs/guide.md):
  every option, noise models, time evolution, the cloud in detail, and what
  isn't supported yet.
- [Examples](https://github.com/QoroQuantum/qiskit-maestro/tree/main/examples):
  a two-line Aer noise-model port, a VQE loop on MPS, time evolution, and the cloud.
- [Maestro docs](https://qoroquantum.github.io/maestro/), including
  [Coming from Qiskit Aer](https://qoroquantum.github.io/maestro/getting-started/coming-from-qiskit.html).
- [Contributing](https://github.com/QoroQuantum/qiskit-maestro/blob/main/CONTRIBUTING.md)
  and [Code of Conduct](https://github.com/QoroQuantum/qiskit-maestro/blob/main/CODE_OF_CONDUCT.md).

## License

GPL-3.0-or-later. See [LICENSE](https://github.com/QoroQuantum/qiskit-maestro/blob/main/LICENSE).
