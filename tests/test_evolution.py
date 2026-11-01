# SPDX-FileCopyrightText: 2026 Qoro Quantum
# SPDX-License-Identifier: GPL-3.0-or-later

"""Time evolution on one persistent simulator (Maestro's incremental_evolve)."""

from __future__ import annotations

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.synthesis import LieTrotter

from qiskit_maestro import MaestroEstimator, UnsupportedInstructionError
from qiskit_maestro.converter import convert
from qiskit_maestro.evolution import find_chain


def _ising(n: int, j: float = 1.0, h: float = 0.7) -> SparsePauliOp:
    terms = [("ZZ", [q, q + 1], -j) for q in range(n - 1)]
    terms += [("X", [q], -h) for q in range(n)]
    return SparsePauliOp.from_sparse_list(terms, n)


def _z(n: int, qubit: int) -> SparsePauliOp:
    return SparsePauliOp.from_sparse_list([("Z", [qubit], 1.0)], n)


def _neel(n: int) -> QuantumCircuit:
    circuit = QuantumCircuit(n)
    circuit.x(range(0, n, 2))
    return circuit


def _step(n: int, dt: float = 0.1) -> QuantumCircuit:
    circuit = QuantumCircuit(n)
    for start in (0, 1):
        for q in range(start, n - 1, 2):
            circuit.rzz(-2 * dt, q, q + 1)
    circuit.rx(-2 * 0.7 * dt, range(n))
    return circuit


def _evolved(n: int, k: int, step: QuantumCircuit | None = None) -> QuantumCircuit:
    circuit = _neel(n)
    for _ in range(k):
        circuit.compose(step if step is not None else _step(n), inplace=True)
    return circuit


def _ops(circuits):
    return [convert(c, estimator=True).ops for c in circuits]


# ---- Finding the chain ---------------------------------------------------------


def test_finds_initial_state_step_and_counts_in_any_order():
    n = 4
    chain = find_chain(_ops([_evolved(n, k) for k in (0, 3, 1, 2)]))
    assert chain is not None
    assert chain.steps == [0, 3, 1, 2]
    assert len(chain.initial) == len(_ops([_neel(n)])[0])
    assert len(chain.step) == len(_ops([_step(n)])[0])


def test_the_shortest_circuit_is_the_initial_state():
    n = 3
    chain = find_chain(_ops([_evolved(n, k) for k in (2, 4, 7)]))
    assert chain.steps == [0, 2, 5]


@pytest.mark.parametrize(
    "build",
    [
        # Not prefixes of one another.
        lambda n: [_evolved(n, 1), _evolved(n, 2, _step(n, dt=0.2))],
        # Prefixes, but the added parts aren't repetitions of one step.
        lambda n: [_evolved(n, 1), _evolved(n, 2), _evolved(n, 2).compose(_step(n, dt=0.3))],
        # An angle that differs by more than the tolerance.
        lambda n: [_evolved(n, 1), _evolved(n, 2, _step(n, dt=0.1 + 1e-9))],
        # All the same length.
        lambda n: [_evolved(n, 2), _evolved(n, 2)],
        # One circuit.
        lambda n: [_evolved(n, 3)],
    ],
    ids=["not-prefixes", "not-periodic", "angle-beyond-tolerance", "same-length", "single"],
)
def test_no_chain(build):
    assert find_chain(_ops(build(4))) is None


def test_angles_within_tolerance_match():
    n = 3
    nudged = _step(n, dt=0.1 + 1e-14)
    assert find_chain(_ops([_evolved(n, 1), _evolved(n, 3, nudged)])) is not None


# ---- Automatically, in run() ------------------------------------------------------


def test_trotter_pubs_run_as_one_evolution():
    n, ks = 8, range(9)
    # Different observables per PUB, including arrays, so each PUB's values
    # must come from its own observables.
    observables = [_ising(n), [_z(n, 0), _z(n, n - 1)], [[_ising(n)], [_z(n, 3)]]]
    pubs = [(_evolved(n, k), observables[k % 3]) for k in ks]
    mine = MaestroEstimator().run(pubs).result()
    reference = StatevectorEstimator().run(pubs).result()
    one_by_one = MaestroEstimator(incremental_evolution=False).run(pubs).result()
    for m, r, o in zip(mine, reference, one_by_one):
        assert m.data.evs.shape == r.data.evs.shape
        np.testing.assert_allclose(m.data.evs, r.data.evs, atol=1e-10)
        np.testing.assert_allclose(m.data.evs, o.data.evs, atol=1e-10)
        assert m.metadata["maestro"]["incremental_evolution"] is True
        assert "incremental_evolution" not in o.metadata["maestro"]


def test_pauli_evolution_gate_with_lie_trotter_reps():
    n = 8
    hamiltonian = _ising(n)
    circuits = [_neel(n)]
    for k in range(1, 7):
        circuit = _neel(n)
        gate = PauliEvolutionGate(hamiltonian, time=0.1 * k, synthesis=LieTrotter(reps=k))
        circuit.append(gate, range(n))
        circuits.append(circuit)
    observables = [hamiltonian, _z(n, 0)]
    results = MaestroEstimator().run([(c, observables) for c in circuits]).result()
    for circuit, result in zip(circuits, results):
        state = Statevector(circuit.decompose(reps=4))
        exact = [state.expectation_value(o).real for o in observables]
        np.testing.assert_allclose(result.data.evs, exact, atol=1e-10)
        assert result.metadata["maestro"]["incremental_evolution"] is True


@pytest.mark.parametrize("method", ["density_matrix", "matrix_product_state"])
def test_other_methods(method):
    n = 6
    pubs = [(_evolved(n, k), [_ising(n), _z(n, 2)]) for k in range(6)]
    mine = MaestroEstimator(method=method).run(pubs).result()
    reference = StatevectorEstimator().run(pubs).result()
    for m, r in zip(mine, reference):
        np.testing.assert_allclose(m.data.evs, r.data.evs, atol=1e-10)
        assert m.metadata["maestro"]["incremental_evolution"] is True


def test_stabilizer_clifford_steps():
    n = 5
    step = QuantumCircuit(n)
    step.h(0)
    for q in range(n - 1):
        step.cx(q, q + 1)
    step.s(range(n))
    labels = ["ZZIII", "IXXII", "YIIIY", "XXXXX", "IIZII"]
    pubs = [(_evolved(n, k, step), labels) for k in range(5)]
    mine = MaestroEstimator(method="stabilizer").run(pubs).result()
    reference = StatevectorEstimator().run(pubs).result()
    for m, r in zip(mine, reference):
        np.testing.assert_allclose(m.data.evs, r.data.evs, atol=1e-12)


@pytest.mark.slow
def test_mps_at_120_qubits_matches_one_by_one():
    n = 120
    pubs = [(_evolved(n, k), [_ising(n), _z(n, 60)]) for k in range(11)]
    options = {"method": "mps", "max_bond_dimension": 256}
    mine = MaestroEstimator(**options).run(pubs).result()
    one_by_one = MaestroEstimator(incremental_evolution=False, **options).run(pubs).result()
    for m, o in zip(mine, one_by_one):
        np.testing.assert_allclose(m.data.evs, o.data.evs, atol=1e-8)
    assert mine[-1].metadata["maestro"]["incremental_evolution"] is True
    assert mine[-1].metadata["maestro"]["bond_dimensions"][0] > 1


def test_precision_noise_is_the_same_on_both_paths():
    n = 5
    pubs = [(_evolved(n, k), [_ising(n), _z(n, 0)]) for k in range(4)]
    mine = MaestroEstimator(seed=9).run(pubs, precision=0.1).result()
    one_by_one = MaestroEstimator(seed=9, incremental_evolution=False).run(pubs, precision=0.1)
    for m, o in zip(mine, one_by_one.result()):
        np.testing.assert_allclose(m.data.evs, o.data.evs, atol=1e-10)
        np.testing.assert_array_equal(m.data.stds, [0.1, 0.1])


def _parametrised_pubs(n):
    theta = Parameter("θ")
    circuit = _evolved(n, 1)
    circuit.rz(theta, 0)
    return [(circuit, _z(n, 0), [0.1]), (_evolved(n, 2), _z(n, 0))]


def _reset_pubs(n):
    first, second = _evolved(n, 1), _evolved(n, 2)
    first.reset(0)
    second.reset(0)
    return [(first, _z(n, 0)), (second, _z(n, 0))]


def _not_a_chain(n):
    return [(_evolved(n, 1), _z(n, 0)), (_evolved(n, 2, _step(n, 0.3)), _z(n, 0))]


@pytest.mark.parametrize(
    "make_pubs,options",
    [
        (_not_a_chain, {}),
        (_parametrised_pubs, {}),
        (_reset_pubs, {}),
        (lambda n: [(_evolved(n, k), _z(n, 0)) for k in (1, 2)], {"incremental_evolution": False}),
        (lambda n: [(_evolved(n, k), _z(n, 0)) for k in (1, 2)], {"method": "tensor_network"}),
    ],
    ids=["not-a-chain", "parameter-values", "resets", "switched-off", "tensor-network"],
)
def test_one_by_one_when_incremental_evolution_does_not_apply(make_pubs, options):
    results = MaestroEstimator(seed=1, **options).run(make_pubs(4)).result()
    for result in results:
        assert "incremental_evolution" not in result.metadata.get("maestro", {})


# ---- Explicitly, with run_evolution() ---------------------------------------------------


def test_run_evolution_matches_one_circuit_per_step():
    n = 6
    steps = [5, 0, 12, 5, 3]
    observables = [[_ising(n), _z(n, 0)], [_z(n, 5), SparsePauliOp("X" * n)]]
    result = MaestroEstimator().run_evolution(_neel(n), _step(n), observables, steps).result()[0]
    assert result.data.evs.shape == (5, 2, 2)
    assert result.data.stds.shape == (5, 2, 2)
    for t, k in enumerate(steps):
        reference = StatevectorEstimator().run([(_evolved(n, k), observables)]).result()[0]
        np.testing.assert_allclose(result.data.evs[t], reference.data.evs, atol=1e-10)
    assert result.metadata["steps"] == steps
    assert result.metadata["maestro"]["incremental_evolution"] is True


def test_run_evolution_with_an_empty_initial_state_and_mps():
    n = 10
    result = (
        MaestroEstimator(method="mps")
        .run_evolution(QuantumCircuit(n), _step(n), _ising(n), range(0, 21, 5))
        .result()[0]
    )
    for t, k in enumerate(range(0, 21, 5)):
        circuit = QuantumCircuit(n)
        for _ in range(k):
            circuit.compose(_step(n), inplace=True)
        reference = StatevectorEstimator().run([(circuit, _ising(n))]).result()[0]
        assert result.data.evs[t] == pytest.approx(reference.data.evs, abs=1e-10)
    assert len(result.metadata["maestro"]["bond_dimensions"]) == 5


def test_run_evolution_precision():
    n = 3
    result = (
        MaestroEstimator(seed=2)
        .run_evolution(_neel(n), _step(n), _z(n, 0), [1, 2], precision=0.05)
        .result()[0]
    )
    np.testing.assert_array_equal(result.data.stds, [0.05, 0.05])
    assert result.metadata["target_precision"] == 0.05


def test_two_prefix_circuits_are_one_step_apart():
    # Any circuit and a longer one that starts with it form a chain: the
    # difference is a single step.
    n = 4
    pubs = [(_evolved(n, 1), _ising(n)), (_evolved(n, 2).compose(_step(n, 0.3)), _ising(n))]
    mine = MaestroEstimator().run(pubs).result()
    reference = StatevectorEstimator().run(pubs).result()
    for m, r in zip(mine, reference):
        assert m.data.evs == pytest.approx(r.data.evs, abs=1e-10)
        assert m.metadata["maestro"]["incremental_evolution"] is True


def test_every_native_gate_through_run_evolution():
    from qiskit.circuit.random import random_circuit

    n = 4
    step = random_circuit(n, 6, max_operands=3, seed=21)
    observables = [SparsePauliOp(label) for label in ["XYZI", "ZZZZ", "IXIY", "YIIX"]]
    result = MaestroEstimator().run_evolution(_neel(n), step, observables, [1, 3]).result()[0]
    for t, k in enumerate([1, 3]):
        reference = StatevectorEstimator().run([(_evolved(n, k, step), observables)]).result()[0]
        np.testing.assert_allclose(result.data.evs[t], reference.data.evs, atol=1e-10)


def test_run_evolution_rejects_resets():
    # Maestro's incremental_evolve doesn't reset a qubit that is in superposition.
    step = QuantumCircuit(2)
    step.h(0)
    step.cx(0, 1)
    step.reset(0)
    with pytest.raises(UnsupportedInstructionError, match="reset"):
        MaestroEstimator().run_evolution(QuantumCircuit(2), step, ["IZ", "ZI"], [1])


@pytest.mark.parametrize(
    "arguments,error,match",
    [
        ((QuantumCircuit(2), QuantumCircuit(3), "ZZ", [1]), ValueError, "qubits"),
        ((QuantumCircuit(2), QuantumCircuit(2), "ZZ", []), ValueError, "non-empty"),
        ((QuantumCircuit(2), QuantumCircuit(2), "ZZ", [1, -1]), ValueError, "non-negative"),
        ((QuantumCircuit(2), QuantumCircuit(2), "ZZ", [1.5]), ValueError, "integers"),
        ((QuantumCircuit(2), QuantumCircuit(2), "ZZZ", [1]), ValueError, ""),
    ],
    ids=["width-mismatch", "no-steps", "negative", "not-integers", "observable-width"],
)
def test_run_evolution_rejects_bad_arguments(arguments, error, match):
    with pytest.raises(error, match=match):
        MaestroEstimator().run_evolution(*arguments)


def test_run_evolution_rejects_unbound_parameters_and_measurements():
    step = QuantumCircuit(1)
    step.rx(Parameter("dt"), 0)
    with pytest.raises(ValueError, match="step has unbound parameters: dt"):
        MaestroEstimator().run_evolution(QuantumCircuit(1), step, "Z", [1])
    measured = QuantumCircuit(1, 1)
    measured.measure(0, 0)
    with pytest.raises(UnsupportedInstructionError, match="measure"):
        MaestroEstimator().run_evolution(QuantumCircuit(1, 1), measured, "Z", [1])
