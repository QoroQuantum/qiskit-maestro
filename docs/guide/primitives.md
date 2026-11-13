# Backend, Sampler and Estimator

## `MaestroBackend`

Replaces `AerSimulator`. `MaestroBackend` has a `Target`, so `transpile` and
`generate_preset_pass_manager(backend=backend)` produce circuits it runs.
Results come back as Aer returns them: counts, `memory=True` for per-shot
bitstrings, and classical registers split by spaces. Qiskit's
`BackendSamplerV2(backend=MaestroBackend())` works too.

Aer's save instructions work as well: `save_statevector`,
`save_statevector_dict`, `save_probabilities`, `save_probabilities_dict`,
`save_expectation_value` and `save_expectation_value_variance`, so
`result.get_statevector()` and `result.data()` return what they return on Aer.

## `MaestroSampler`

Replaces `StatevectorSampler` or Aer's `SamplerV2`. PUBs, broadcasting,
per-PUB shots and the result shapes are the same as `StatevectorSampler`'s.
Every parameter set of a PUB reuses one converted circuit, and the sets run in
parallel.

## `MaestroEstimator`

Replaces `StatevectorEstimator` or Aer's `EstimatorV2`.

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

## Bit order

Maestro puts qubit 0 on the left of bitstrings and Pauli strings; Qiskit puts it
on the right. qiskit-maestro converts at the boundary, so everything you see
(counts, memory, `BitArray`s, observables) uses Qiskit's order.
