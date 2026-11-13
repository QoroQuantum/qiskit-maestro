# Supported features

## What's supported

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
- Aer noise models: see [Noise models](noise_models.md).
- Aer's `save_statevector`, `save_statevector_dict`, `save_probabilities`,
  `save_probabilities_dict`, `save_expectation_value` and
  `save_expectation_value_variance`, anywhere before the first measurement, on
  `MaestroBackend`. With noise or a reset before it, a saved statevector is one
  random trajectory's, as on Aer; saved probabilities and expectation values
  are averaged over the shots, or exact on the density matrix.
- `method="automatic"`, as described under
  [Automatic method selection](options.md#automatic-method-selection).
- Parameterised circuits, with Qiskit's broadcasting rules.
- `SparseObservable` projector terms (`0`, `1`, `+`, `-`, `r`, `l`) in the
  Estimator (Qiskit 2.x).

## What isn't supported yet

- **Some noise.** Errors on three or more qubits, correlated readout errors on
  several qubits, and Aer's nonlocal errors raise `UnsupportedNoiseError`. So
  do coherent (unitary) errors and general Kraus channels on the GPU and on
  methods other than `density_matrix`, `statevector` and
  `matrix_product_state`. So does noise on a circuit with classical control,
  other than readout errors, and noise in `run_evolution`. Relaxation can't
  run on the stabilizer method.
- **Some classical control.** `while_loop`, `break`, `continue`, and classical
  expressions beyond "bit is set" and "register equals value" aren't supported.
  Casts are read as Qiskit does: a cast to a narrower `Uint` keeps the low bits.
  A cast of a multi-bit register to `Bool` (a non-zero test) isn't supported.
  Neither are nested `if_test` blocks (including a `switch` inside one), a
  `reset` inside `if_test`, an `else` branch on a multi-bit condition, or a
  `switch` case that measures into the switch's own target. Each raises
  `UnsupportedInstructionError` naming the instruction.
- **Measurements in Estimator circuits.** As in `StatevectorEstimator`, they
  raise an error. Resets are allowed: pure-state methods average 1,024
  trajectories, as Aer's `EstimatorV2` averages over shots, and density matrix
  is exact.
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
- **On the Qoro cloud**: see [Not on the cloud yet](gpu_and_cloud.md#not-on-the-cloud-yet).
