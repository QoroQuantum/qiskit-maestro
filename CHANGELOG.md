# Changelog

## Unreleased

- Documentation at
  [qoroquantum.github.io/qiskit-maestro](https://qoroquantum.github.io/qiskit-maestro/),
  built with Sphinx and the Qiskit ecosystem theme: the README, the guide (one
  page per topic), the examples, an API reference generated from the docstrings, and this
  changelog. `pip install -e ".[docs]"` installs what it needs, and CI builds
  it on every push and publishes it from `main`. The package's Documentation
  link on PyPI now points there.
- The README has the Qiskit Ecosystem badge.
- Pauli propagation is documented: a row in the README's method table, and a
  page of the guide on setting it up with `MaestroEstimator`, choosing a
  `pp_coefficient_threshold`, and running it on the GPU and the Qoro cloud.
- Three new examples, simpler than the others:
  `examples/getting_started.py` (a Bell state on the backend, the Sampler
  and the Estimator), `examples/simulation_methods.py` (one GHZ circuit on
  the statevector, MPS and stabilizer methods, from 20 to 1,000 qubits) and
  `examples/pauli_propagation.py` (the kicked Ising circuit on IBM's
  115-qubit heavy-hex lattice). The test suite runs all three.
- `benchmarks/pauli_propagation.py`, a short version of the Pauli propagation
  benchmark in Maestro's documentation: accuracy against time for
  `MaestroEstimator` and IBM's `pauli-prop` on a kicked Ising chain.
- Each Aer error is one `maestro.QuantumChannel`, built once and shared by every
  instruction it follows, on `density_matrix`, `statevector` and
  `matrix_product_state` (CPU). The density matrix applies the channel in one
  pass instead of a Pauli channel and a delay. The statevector and the matrix
  product state sample it from their own state on every shot, as Aer does:
  noisy circuits no longer carry an ancilla qubit, and relaxation towards a
  thermal population (`from_backend(..., temperature=...)`) is sampled too.
  The other methods and the GPU keep the Pauli channel and the ancilla.
  With a fake IBM device's full noise model (`benchmarks/noise_vs_aer.py`, Aer
  0.18.2, a 24-core desktop): the 12-qubit density matrix takes 2.1 s at 1,000
  shots, where it took 29 s and Aer takes 2.9 s; the statevector is 2.7 to 7.9
  times as fast as Aer at 6 to 20 qubits; and the matrix product state at a
  bond limit of 64 is 5 to 15 times as fast at 6 to 16 qubits, and takes two
  minutes at 20 qubits, where Aer was stopped after fifteen.
- Coherent errors, mixtures of unitaries and general Kraus channels on one or
  two qubits run on `density_matrix`, `statevector` and `matrix_product_state`
  (CPU), from Aer's description of the error: as a mixture of unitaries where
  its circuits are unitary, and by its Kraus operators otherwise. They raised
  `UnsupportedNoiseError` before, and still do on the other methods. An error
  with a single unitary branch, such as a coherent over-rotation, is a gate: a
  circuit whose errors are all of that kind is evolved once for all its shots.
- Requires the Maestro release that adds `QuantumChannel` and
  `QuantumCircuit.channel` (after 0.3.14).
- `benchmarks/noise_vs_aer.py` times `MaestroBackend` against `AerSimulator`
  under a fake IBM device's noise model, after checking that they agree: in
  the full distribution on the density matrix, and on the sampled methods in
  every qubit's marginal and in the parities of consecutive pairs and of all
  qubits. A matrix product state whose bond limit truncates the state is not
  compared with Aer's, which truncates differently: up to 20 qubits, each
  simulator's error in those statistics is reported against the exact ones,
  from Maestro's statevector under the same noise. Each run records the
  versions of Qiskit, Aer, Maestro and the plugin, the QCSim revision Maestro
  was built against, the threads they could use, the circuit's seed, the bond
  limit and the workers Maestro ran the noise realizations on.
- Result metadata gains `metadata["maestro"]["realization_workers"]`: the
  workers Maestro ran a call's noise realizations on, for sampled noise.

## [0.2.0]

- CI runs on every push to every branch, not only on `main`, and adds Python
  3.14, macOS, the Qoro cloud tests (skipped in CI before, since Divi wasn't
  installed), a job with every dependency at the lowest version allowed, a
  coverage floor, and `actionlint` on the workflows. The save-instruction and
  OpenQASM 2 tests now pass on Qiskit 1.1 too. Dependabot keeps the workflows'
  actions up to date.
- A shorter README, with the Qoro cloud near the top. The details moved to
  `docs/guide.md`, and two new examples cover the cloud
  (`examples/qoro_cloud.py`) and time evolution (`examples/time_evolution.py`).
- Run on the Qoro cloud: `service=QoroService()` (from Divi 0.16 or later,
  installed with `pip install "qiskit-maestro[cloud]"`, which needs Python 3.11
  or later and Qiskit 2.6 or later) on `MaestroBackend`, `MaestroSampler`
  and `MaestroEstimator`. Each `run()` submits one cloud job and returns a
  `QoroCloudJob` (backend) or `QoroCloudPrimitiveJob` (primitives) carrying the
  cloud's job ID, with `status()`, `cancel()`, and `result()` that waits. The
  estimator uses the cloud's expectation-value jobs, so its values are exact.
  Circuits are converted and checked locally, then sent as OpenQASM 2.0 with
  the simulator options as a Maestro config; results come back in the same
  form and bit order as local runs. Failed jobs, cloud timeouts and running out
  of credits raise the new `QoroCloudError`. `device="GPU"` runs on the
  `qoro_gpu` cluster. Parameter sweeps go as one template per circuit with a
  table of parameter values, bound by the cloud, when the job's circuits share
  their shots or observables. Noise models, classical control,
  save instructions, estimator resets and `run_evolution` raise a clear error
  on the cloud for now. `from_backend` carries the service over.
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
- Requires `qoro-maestro` 0.3.14, which runs a call's noise realizations on its
  own workers. Noisy sampling on pure-state methods hands each parameter set's
  shots to one Maestro call, with `return_samples=True` so that seeded runs take
  Maestro's direct path, and the plugin no longer runs such calls on threads of
  its own. Seeded results don't depend on the number of workers. With a
  device's depolarizing and readout errors, sampling is 2.3 to 3.6 times as
  fast as Aer's at 8 to 12 qubits.
- `MaestroEstimator`: a circuit with a `reset` on a pure-state method averages
  1,024 trajectories, as Aer's `EstimatorV2` averages over shots, instead of
  following one random outcome. Density matrix stays exact.
- Noise models translate about 13 times faster: the plugin computes each
  error's Pauli transfer matrix from its instructions, instead of through
  Qiskit's `SuperOp`, and falls back to Qiskit for instructions it doesn't know.
  Each error's translation is also kept for later runs while the error exists;
  adding to a noise model makes a new error, which is translated afresh. With
  the leaner conversion and outcome counting, the plugin's own time in a noisy
  run fell from 6–12 ms to 3–6 ms.

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
