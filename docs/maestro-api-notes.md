# Maestro Python API: notes for the Qiskit plugin

Studied against `qoro-maestro==0.3.10` (PyPI), Qiskit 2.6.1 and Qiskit Aer 0.18.2,
on macOS arm64 (no GPU). Every claim below was checked by running code, and
the ones the plugin relies on are covered by its test suite. Gap 13 and the
`reset` semantics in `estimate` were rechecked against 0.3.14 (core `main` at
`d09d9ae`, built from source), which the plugin now requires; the other gaps
have not been rechecked yet.

## How the core works

**Circuits.** `maestro.circuits.QuantumCircuit()` takes no width. Gates are
methods taking plain floats (`qc.rx(qubit, theta)`, qubit first); there are no
symbolic parameters. The width is the highest qubit index used plus one
(`num_qubits`). Native gate methods: `x y z h s sdg t tdg sx sxdg k p rx ry rz u
cx cy cz ch csx csxdg swap cp crx cry crz cu ccx cswap`, plus `measure`
(list of `(qubit, clbit)` pairs), `reset`, `reset_qubits`, `delay` (seconds),
`measure_all` and `pauli_channel`. All 29 Qiskit-equivalent gates match Qiskit's
matrices exactly, global phase included (`cu` takes `theta, phi, lambda, gamma`
in Qiskit's order). `k` has no Qiskit equivalent.

**Importing Qiskit circuits.** There is no Qiskit importer. The two routes are
building a `QuantumCircuit` through the gate methods, or OpenQASM text through
`maestro.QasmToCirc().parse_and_translate(text, params)` or
`simple_execute(text)`. The parser covers QASM 2 and a static subset of QASM 3,
including `input` parameters bound at parse time and classical control:
`if(c==3) op;`, and QASM 3 `if (c[0]) {...} else {...}`, `if (!c[0])`,
`if (c[0] && !c[1]) {...}`.

**Configuration.** `SimulatorConfig(simulator_type=..., simulation_type=...,
max_bond_dimension=..., singular_value_threshold=..., precision=..., seed=...,
gpu_device=..., ...)`, 38 keyword-only fields. `simulation_type` values:
Statevector 0, MatrixProductState 1, Stabilizer 2, TensorNetwork 3,
PauliPropagator 4, ExtendedStabilizer 5, PathIntegral 6, DensityMatrix 7,
MatrixProductOperator 8. `simulator_type` values: QiskitAer 0, QCSim 1,
CompositeQiskitAer 2, CompositeQCSim 3, Gpu 4, QuestSim 5, DistributedGpu 6,
DistributedMpiGpu 7. `precision` accepts only `"single"` and `"double"`, the
same values as Aer.

**Sampling.** `qc.execute(config, shots)` returns
`{"counts", "time_taken", "simulator", "method", "sampling_components",
"largest_component_qubits", ["gpu_device"], ["max_bond_dim_reached"]}`.
`simulator` and `method` are the enum values that actually ran, which can
differ from what was asked for (see gaps 1 and 2). Counts are keyed by bitstrings
with **clbit 0 on the left**, and the string is only as long as the highest
clbit written plus one: measuring only into clbit 4 gives `"00001"`, and a
circuit with no measurement gives `{"0": shots}`. Mid-circuit measurement and
reset work; Maestro switches to per-shot execution by itself. Seeds are
reproducible, including across threads.

**Expectation values.** `qc.estimate(list_of_pauli_strings, config)` returns
`{"expectation_values", "expectation_error_bounds", "simulator", "method",
"cluster_methods", ...}`. Character *i* of a Pauli string acts on **qubit i**
(qubit 0 on the left), the reverse of Qiskit's labels. Identity and lowercase
labels work, and strings longer than the circuit are fine. The values are
computed from the state (no shots) for every method: statevector, MPS,
stabilizer, density matrix, tensor network and Pauli propagation. There are no
coefficients and no standard deviations. `AdjointProgram` does accept
`(coefficient, string)` terms and many parameter rows at once, but only on the
CPU statevector.

**Parallelism.** `execute` and `estimate` release the GIL. Calling them from 8
Python threads ran 200 twelve-qubit estimates 6.4× faster than calling them
one after another, with identical results. `AdjointProgram` with 200 rows was
about 2× faster again on that workload, and about even at 20 qubits.

**Time evolution.** `maestro.incremental_evolve(init, step, measure_at_steps,
paulis, config)` runs `init` once, applies `step` repeatedly on one simulator,
and evaluates Pauli strings at each requested step count, so n points of an
N-step evolution cost N steps instead of about n·N/2. It agreed with `estimate`
to 1e-13 on statevector, MPS, density matrix and stabilizer. On a 120-qubit MPS
Trotter run measured at 20 points it was 5.8× faster than one `estimate` per
point (8.6× at 200 qubits and 40 points), 3.2× on a 20-qubit statevector, and
no faster on tensor network or Pauli propagation, which recompute from the
start anyway. It honours the requested backend and method (no automatic
selection), and raises on an unavailable GPU rather than falling back. The
plugin uses it for `MaestroEstimator.run_evolution` and for PUB lists that
repeat one step.

**Bit order.** Maestro puts qubit 0 and clbit 0 on the left of bitstrings and
Pauli strings. Qiskit puts them on the right. Statevector amplitude order is the
same in both (qubit 0 is the least significant bit of the index).

## Gaps in the core, for the Maestro team

Ordered by how much they cost Qiskit users. "Workaround" is what the plugin does
today; none of these are hidden from users.

1. **GPU requests fall back to the CPU silently.** With no GPU (or no licence),
   `SimulatorConfig(simulator_type=Gpu)` runs on QCSim and may also switch the
   method (a Clifford circuit ran as `method=2`, stabilizer). GPU with
   `DensityMatrix` deliberately runs CPU MPO. Python has no way to ask for
   *fixed* selection: `fixed_backend` exists in C++ and the JSON request API
   but not on `SimulatorConfig`. Workaround: the plugin checks
   `maestro.is_gpu_available()` before running, rejects GPU with stabilizer or
   density matrix, and checks `result["simulator"]` afterwards. Ask: raise on an
   unavailable GPU, or expose `selection="fixed"` in Python.
2. **The requested method can be replaced.** Python configs always use
   automatic selection, so a Clifford circuit asked for as statevector may run
   as stabilizer. Results are still exact, so the plugin only reports the method
   that ran in result metadata. Same ask as 1.
3. **Measurements in `estimate` are handled two different ways, silently.** A
   measurement with nothing after it on that qubit is removed
   (`RemoveEndMeasurements`), so `h(0); measure(0)` gives ⟨X₀⟩ = 1, the value
   before the measurement. A measurement followed by more gates collapses to one
   sampled outcome, so `h(0); measure(0); h(0)` gives ⟨X⟩ = ±1. That happens on
   density matrix too, where the exact ensemble value (0) is available.
   Workaround: the estimator rejects circuits that measure, as Qiskit's
   `StatevectorEstimator` does. Ask: document the two behaviours, and treat a
   mid-circuit measurement as dephasing on density-matrix and MPO methods, as
   `reset` already is.
4. **No symbolic parameters and no batched execute/estimate.** Each parameter
   set needs a new `QuantumCircuit`. Workaround: the plugin converts the Qiskit
   circuit once into a list of gate calls with parameter expressions, evaluates
   the expressions for every row, rebuilds the Maestro circuit (cheap), and runs
   the rows in a thread pool. Ask: a parameterised circuit with a
   `run(params[batch, n])` for every method, so the core can schedule and size
   the batch (the plugin's thread pool can't see memory use).
5. **No per-shot results.** `execute` returns counts only. Workaround: `memory`
   and `BitArray` shots are the counts expanded and shuffled with a seeded RNG.
   This is exact in distribution because shots are independent, but it isn't
   the order the shots ran in. Ask: an optional `memory=True` on `execute`.
6. **Classical control only through QASM.** `QuantumCircuit` has no
   conditional methods. Workaround: circuits with `if_test` are written out as
   OpenQASM 3 and parsed each run. The parser then limits what works:
   - no conditional `reset` ("the circuit representation has no conditional
     reset operation");
   - no `else` on a condition over more than one bit;
   - no nested `if`, and no `while`, `for` or `switch`;
   - only `==` and bit tests, no classical expressions;
   - no noise channels, so a circuit with both quantum noise and classical
     control can't run (readout errors work, through the `NoiseModel`).

   The plugin unrolls `for` loops and rewrites `switch` as one `if` per case
   value, and rejects the other cases with an error naming them. Ask:
   conditional gate/measure/reset methods on `QuantumCircuit`.
7. **Missing gate methods.** There's no `unitary` (the low-level `Simulator`
   has `ApplyGenericOneQubitGate` and the 2- and 3-qubit versions, but
   `QuantumCircuit` doesn't, and the plugin runs everything through
   `QuantumCircuit.execute`/`estimate`, so `UnitaryGate` is still expanded
   through its Qiskit definition; still true in 0.3.13). Ask: a
   `QuantumCircuit.unitary(qubits, matrix)` for one to three qubits. `rzz`
   and `rxx` exist only in the QASM parser, and
   `ryy`, `rzx`, `ecr`, `iswap` and `xx_plus_yy` aren't available at all.
   Workaround: those gates are decomposed through their Qiskit definitions, so
   `rzz` becomes `cx, rz, cx`. That's correct, but it doubles the two-qubit gates
   an MPS has to apply.
8. **Width is inferred.** There's no way to declare 10 qubits, and
   `get_statevector` drops untouched qubits. Counts and estimates aren't
   affected (checked with idle qubits in the middle). The plugin pads saved
   statevectors with the untouched high qubits, and bitstrings to the circuit's
   clbit count.
9. **No standard deviations.** `estimate` has no variance, and
   `expectation_error_bounds` is non-zero only when `observable_error_tolerance`
   is set. With `precision > 0` the plugin adds Gaussian noise as
   `StatevectorEstimator` does and reports `stds = precision`. Ask: a truncation
   error estimate for MPS and Pauli propagation that the estimator could report
   as `stds`.
10. **`incremental_evolve` doesn't reset qubits in superposition.** With a step
    of `h(0); cx(0, 1); reset(0)`, ⟨Z₀⟩ after one step is 0 on statevector, MPS
    and density matrix; it should be 1, and `estimate` gets 1. A reset of a
    qubit in a basis state works. Workaround: `run_evolution` rejects resets,
    and the automatic path leaves PUBs with resets to `estimate`. Ask: apply
    resets as `estimate` does. Smaller points: it takes Pauli strings only (no
    coefficients), returns steps sorted, and reports the configured
    `simulator` and `method` enums where `execute` and `estimate` report ints.
11. **`NoiseModel` noise is per qubit, not per instruction.** It acts after
    every gate on a qubit (with separate one- and two-qubit layers), whatever
    the gate. Aer attaches errors per instruction name and qubits, so IBM
    device models leave `rz` noiseless and put one error on each `ecr`, which
    the plugin expands into several Maestro gates. In Python only `h`, `s` and
    `sdg` take `noiseless=True`, and circuits have no Kraus or
    thermal-relaxation instruction, only `pauli_channel` (one or two qubits).
    The C++ circuit has both (`SetNoiseless` on every operation,
    `QuantumChannelOperation` with Kraus operators), and the native request
    API exposes them (`"noiseless"` on gates, `"kraus"` instructions), but the
    Python bindings don't. Workaround: the plugin doesn't map Aer errors onto
    `NoiseModel` setters. It writes each error as a Pauli channel plus
    amplitude damping and places them where Aer applies the error: a
    `pauli_channel`, and on density matrix a `delay` under `set_idle_noise`
    (T1 = 1, T2 = 2, the excited population), which Maestro applies as an
    exact thermal channel. On pure-state methods the damping runs on an
    ancilla (`cry`, `cx`, `reset`), because the sampled thermal noise
    approximates T2 > T1 by T2 = T1 (with a `RuntimeWarning`). Coherent errors
    and general Kraus channels raise. Ask: `noiseless=` on every gate method,
    and `kraus_channel(qubits, operators)` on `QuantumCircuit`.
12. **Noisy expectation values simulate every qubit.** With a channel in the
    circuit, `noisy_estimate_montecarlo` skips the causal-cone reduction, so a
    3-qubit circuit transpiled for a 127-qubit device didn't finish in
    5 minutes (sampling it with `noisy_execute` takes milliseconds).
    Workaround: noisy circuits drop the qubits no operation touches, as Aer's
    truncation does, and observables are remapped (X or Y on a dropped qubit
    gives 0). Ask: drop untouched qubits before simulating.
13. **Fixed in 0.3.14: `noisy_execute` ran realizations one after another.**
    In 0.3.10 to 0.3.13, `execute_noise_batches` looped over the realizations
    in one thread, holding the GIL to inject noise and build each one's
    counts, so 200 shots of a 10-qubit device circuit took 93 ms against
    Aer's 30 ms, and threads in the plugin stopped helping at 8 (the fitted
    serial part was 26–56 µs per shot). 0.3.14 (`ExecuteNoisy`) runs them on
    up to 16 workers with the GIL released: the same 200 shots take 6.7 ms,
    and 20,000 take 0.58 s against Aer's 2.2 s, scaling almost linearly to 8
    workers. Seeded results don't depend on the worker count. The plugin
    makes one call per parameter set and doesn't add threads of its own,
    which would oversubscribe the CPU (splitting into 64 calls made a
    4,000-shot run 13 times slower). Remaining friction: seeded calls that
    return counts skip the direct path that unseeded calls and
    `return_samples=True` take (12.6 against 6.5 ms at 200 shots), so the
    plugin asks for samples and counts them itself.
14. **`get_statevector` needs a statevector or MPS.** The stabilizer and
    tensor-network methods raise ("Invalid simulation type for obtaining the
    amplitude"). The plugin rejects `save_statevector` on them.
15. **Small API frictions.** `estimate` accepts only a `list` of strings, not a
    tuple or other sequence. `delay` takes seconds, but Qiskit delays are often
    in `dt`, which a simulator target doesn't define. The plugin drops delays,
    which is exact without noise.

## Semantics that match Qiskit's references

- A mid-circuit `reset` inside `estimate`: in 0.3.14, pure-state methods
  average 100 trajectories (`estimate_evaluation: "trajectory_average"`) and
  density matrix is exact; 0.3.10 followed one seeded trajectory. The plugin
  averages 1,024 through `noisy_estimate_montecarlo` with an empty noise
  model, as Aer's `EstimatorV2` averages over shots (checked: ⟨Z₁⟩ after
  resetting half a Bell pair is within 0.16 of 0 on statevector, MPS and
  stabilizer, and exactly 0 on density matrix).
- `StatevectorEstimator` reports `stds = 0` even when `precision > 0` adds
  noise. The plugin reports `stds = precision` in that case, because that is the
  standard deviation of the value it returns. This is the one intentional
  difference from the reference.
