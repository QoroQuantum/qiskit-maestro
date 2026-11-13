# Time evolution

`MaestroEstimator` runs time evolution on one persistent simulator, using
Maestro's incremental evolution. If the PUBs of one `run` call are the same
initial circuit followed by 0, 1, 2, ... repetitions of one step, as when each
time point of a Trotter evolution is its own circuit, Maestro simulates the
longest circuit once and reads every time point on the way. Measuring at every
one of N steps then costs N steps instead of about N²/2. Circuits built with
`PauliEvolutionGate(..., synthesis=LieTrotter(reps=k))` are recognised too.

You can also give the step directly, without building a circuit per time point:
`estimator.run_evolution(initial_state, step, observables, steps=range(21))`.
[The time evolution example](../examples/time_evolution.md) does it both ways.

Each result's metadata says whether incremental evolution was used. Pass
`incremental_evolution=False` to simulate every circuit from the start. It
isn't used for PUBs with parameter arrays, circuits that reset qubits, or the
tensor network and Pauli propagation methods, which gain nothing from it.
