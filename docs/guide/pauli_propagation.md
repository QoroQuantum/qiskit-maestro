# Pauli propagation

`method="pauli_propagation"` computes expectation values by running the
circuit backwards on the observable, a sum of Pauli strings, instead of
forwards on the state. A Clifford gate maps each string to one string, and a
rotation can split it in two. The cost depends mostly on how many strings
there are, not on how many qubits, so it reaches circuits too wide for a
statevector, including lattices an MPS handles badly, such as the heavy-hex
layout of IBM's devices. Use it with `MaestroEstimator`:

```python
from qiskit_maestro import MaestroEstimator

estimator = MaestroEstimator(
    method="pauli_propagation",
    maestro_options={"pp_coefficient_threshold": 1e-4},
)
evs = estimator.run([(circuit, observables)]).result()[0].data.evs
```

## Truncation

With no threshold, every string is kept and the values are exact, but the
number of strings can grow exponentially with depth. For deep circuits, set
`pp_coefficient_threshold`: strings whose coefficients fall below it are
dropped. A smaller threshold is more accurate and slower, so lower it until
the values stop changing. `stds` doesn't include the truncation error.

These [`maestro.SimulatorConfig`](https://qoroquantum.github.io/maestro/api/configuration.html)
fields tune the truncation. Pass them in `maestro_options`:

| Field | Effect |
|---|---|
| `pp_coefficient_threshold` | Drop strings whose coefficient is at most this, in magnitude |
| `pp_max_pauli_weight` | Drop strings that act on more qubits than this |
| `pp_gates_between_trims` | Apply the two limits every this many gates. `1` trims after every gate, as IBM's `pauli-prop` does |

## GPU, cloud and noise

Pauli propagation runs on the GPU with `device="GPU"`, and on the Qoro cloud
with `service=`, on its CPUs or GPUs, with the same `maestro_options`. Noise
models work with it too. [The Pauli propagation example](../examples/pauli_propagation.md)
runs the kicked Ising circuit of IBM's utility experiment on the 115-qubit
heavy-hex lattice, exactly and at three thresholds.
`benchmarks/pauli_propagation.py` compares accuracy against time with IBM's
[`pauli-prop`](https://github.com/Qiskit/pauli-prop), as in
[Maestro's benchmarks](https://qoroquantum.github.io/maestro/benchmarks.html#pauli-propagation).
