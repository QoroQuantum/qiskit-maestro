# Changelog

## Unreleased

- Aer noise models: `noise_model=` on `MaestroBackend` (constructor, `set_options`
  or `run`), `MaestroSampler` and `MaestroEstimator`, including through Aer's
  `options={"backend_options": {...}}`. Each error acts where Aer applies it,
  per instruction name and qubits: after the instruction, or before it for
  `measure`. Readout errors and `from_backend`'s delay relaxation apply too, and
  the backend's target uses the noise model's basis gates, as Aer's does.
  Errors are written as a Pauli channel followed by amplitude damping,
  which covers depolarizing, Pauli, phase-damping, amplitude-damping,
  thermal-relaxation and reset errors and their compositions, including
  `NoiseModel.from_backend`'s. Each translation is checked against Aer's channel
  to 1e-9. On `density_matrix` they're exact: results match Aer's to about
  1e-15. On pure-state methods they're sampled per shot, with amplitude damping
  applied exactly through an ancilla qubit. Idle qubits are dropped from noisy
  circuits, as Aer does, so circuits transpiled for a 127-qubit device run
  quickly. Anything else, such as coherent errors, raises the new
  `UnsupportedNoiseError`, a subclass of `UnsupportedInstructionError`, naming
  the instruction and qubits.
- Aer's `save_statevector`, `save_statevector_dict`, `save_probabilities`,
  `save_probabilities_dict`, `save_expectation_value` and
  `save_expectation_value_variance` on `MaestroBackend.run`, returning the types
  Aer returns, so `result.get_statevector()` and `result.data()` work as on Aer.
  Saved statevectors include the circuit's global phase. With noise, a saved
  statevector is one random trajectory's, as on Aer, and saved probabilities and
  expectation values are averaged over the shots.
- `method="automatic"`, Aer's default: the statevector for noiseless circuits,
  and the density matrix for small noisy ones, by Aer's rule.
- `for_loop` is unrolled, and `switch` becomes one `if` block per case value.
  Both are in the backend's target, so `transpile` keeps them.
- Noisy sampling on pure-state methods runs its shots in parallel, 1,024 per task.

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
  Conditions are evaluated with Qiskit's semantics:
  - A cast to a narrower `Uint` keeps the low bits, so a two-bit register
    holding 3 satisfies `cast(register, Uint(1)) == 1`. A cast of a multi-bit
    register to `Bool`, which tests for non-zero, raises
    `UnsupportedInstructionError` instead of being read as a different test.
  - A condition that can never hold, such as a clbit compared with 2 or a
    register compared with a value its width can't represent, is never true. Its
    `if` body is dropped and only an `else` branch runs. A clbit compared with 2
    is no longer read as `clbit == 1`.
- Time evolution on one persistent simulator, through Maestro's incremental
  evolution: `MaestroEstimator.run_evolution`, and automatic recognition of
  PUBs that repeat one step, such as one circuit per Trotter time point. Switch
  the automatic part off with `incremental_evolution=False`.
- `MaestroSampler` returns an empty `BitArray` of the same shape as
  `StatevectorSampler` for an empty parameter batch, such as bindings of shape
  `(0, 1)`.
- Stabilizer estimates run with Maestro's `auto_reduce` off. Maestro 0.3.10
  fails some Clifford estimates with "The stabilizer simulator does not support
  the Rx gate" when `auto_reduce` is on, for example `X` on qubit 0 of a
  two-qubit circuit with the observable `ZZ`. Set `auto_reduce` in
  `maestro_options` to override this. The workaround can go once the minimum
  `qoro-maestro` has a fix.
- The source distribution includes the whole test suite, `conftest.py` and
  `tests/__init__.py` too, so it can be tested after extracting. CI builds
  both distributions and checks them with `scripts/verify_dist.py`.
- Supports Qiskit 1.1 and later, including 2.x, on Python 3.10 to 3.13.
