# Options and simulation methods

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
| `method` | `"statevector"`, `"matrix_product_state"` (`"mps"`), `"stabilizer"`, `"extended_stabilizer"`, `"density_matrix"`, `"tensor_network"`, `"pauli_propagation"` (see [Pauli propagation](pauli_propagation.md)), `"automatic"` | `"statevector"` |
| `device` | `"CPU"`, `"GPU"` | `"CPU"` |
| `precision` | `"double"`, `"single"` | `"double"` |
| `seed_simulator` (`seed` on the primitives) | integer | random |
| `max_bond_dimension` | integer, MPS only | no cap |
| `truncation_threshold` | discarded weight, MPS only | Maestro's default |
| `max_parallel_experiments` | threads for independent circuits and parameter sets; `0` chooses | `0` |
| `maestro_options` | dict of any other [`maestro.SimulatorConfig`](https://qoroquantum.github.io/maestro/api/configuration.html) field | none |
| `noise_model` | a `qiskit_aer.noise.NoiseModel`; see [Noise models](noise_models.md) | none |
| `service` | a `divi.backends.QoroService`; see [Qoro cloud](gpu_and_cloud.md#qoro-cloud) | none (runs locally) |

## Automatic method selection

`"automatic"` takes Aer's name for its default. Noiseless circuits run on the
statevector, Maestro's default. A noisy circuit whose measurements all come at
the end runs on the density matrix when the shots outnumber the 2ⁿ amplitudes
of its n active qubits, as on Aer.

Maestro picks the fastest engine for a circuit by itself, so it may run a
Clifford circuit you sent to `"statevector"` on the stabilizer engine. Results
are unaffected, and each result's metadata records what actually ran.
[The simulation methods example](../examples/simulation_methods.md) runs one
circuit on three methods.
