# Changelog

## [0.1.0]

First release.

- `MaestroBackend`, a `BackendV2` with a `Target` of Maestro's native gates and
  all-to-all connectivity. `run` returns counts and, with `memory=True`,
  per-shot results, so `BackendSamplerV2` works with it.
- `MaestroSampler`, a native `SamplerV2` with full PUB support, which runs the
  parameter sets of a PUB in parallel.
- `MaestroEstimator`, a native `EstimatorV2` that computes expectation values
  from the simulated state for every method, with `StatevectorEstimator`'s
  `precision` semantics.
- Methods `statevector`, `matrix_product_state`, `stabilizer`,
  `density_matrix`, `tensor_network` and `pauli_propagation`, on `CPU` or `GPU`,
  with Aer's option names.
- Mid-circuit measurement, reset, and `if_test`/`else` classical control.
- Time evolution on one persistent simulator, through Maestro's incremental
  evolution: `MaestroEstimator.run_evolution`, and automatic recognition of
  PUBs that repeat one step, such as one circuit per Trotter time point. Switch
  the automatic part off with `incremental_evolution=False`.
- Supports Qiskit 1.1 and later, including 2.x, on Python 3.10 to 3.13.
