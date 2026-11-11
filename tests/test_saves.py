# SPDX-FileCopyrightText: 2026 Qoro Quantum
# SPDX-License-Identifier: GPL-3.0-or-later

"""Aer's save instructions on MaestroBackend, compared with Aer."""

from __future__ import annotations

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import Pauli, SparsePauliOp, Statevector
from qiskit.result import ProbDistribution

from qiskit_maestro import (
    MaestroBackend,
    MaestroEstimator,
    MaestroSampler,
    UnsupportedInstructionError,
)

from .conftest import HAS_AER, qft_gate

pytestmark = pytest.mark.skipif(not HAS_AER, reason="qiskit-aer isn't installed")

if HAS_AER:
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, amplitude_damping_error, depolarizing_error


def _state_circuit() -> QuantumCircuit:
    """Gates with global phases in their definitions, and an idle top qubit."""
    circuit = QuantumCircuit(4, 2)
    circuit.h(0)
    circuit.rzz(0.4, 0, 1)
    circuit.ecr(1, 2)
    circuit.append(qft_gate(3), [0, 1, 2])
    circuit.rx(0.3, 1)
    circuit.global_phase = 0.25
    return circuit


def _aer_data(circuit, **options):
    # Aer doesn't know QFTGate, or before Qiskit 1.2 the QFT circuit as a gate, named
    # "QFT", whose definition is itself a "QFT" gate. The definitions hold the same
    # global phase.
    circuit = circuit.decompose(gates_to_decompose=["qft", "QFT"], reps=2)
    return AerSimulator(**options).run(circuit, shots=200, seed_simulator=1).result().data(0)


def _mine(circuit, **options):
    return MaestroBackend(seed_simulator=1, **options).run(circuit, shots=200).result()


@pytest.mark.parametrize("method", ["statevector", "matrix_product_state"])
def test_save_statevector_matches_aer_with_global_phase(method):
    circuit = _state_circuit()
    circuit.save_statevector()
    circuit.save_statevector_dict(label="dict")
    circuit.measure([0, 1], [0, 1])
    result = _mine(circuit, method=method)
    aer = _aer_data(circuit)
    state = result.get_statevector()
    assert isinstance(state, Statevector)
    np.testing.assert_allclose(state.data, aer["statevector"].data, atol=1e-12)
    data = result.data(0)
    assert data["dict"].keys() == aer["dict"].keys()
    for key, amplitude in aer["dict"].items():
        assert data["dict"][key] == pytest.approx(amplitude, abs=1e-12)
    assert sum(result.get_counts().values()) == 200


def test_mid_circuit_saves_see_the_state_at_that_point():
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.save_statevector(label="before")
    circuit.cx(0, 1)
    circuit.save_statevector(label="after")
    data = _mine(circuit).data(0)
    np.testing.assert_allclose(data["before"].data, [2**-0.5, 2**-0.5, 0, 0], atol=1e-12)
    np.testing.assert_allclose(data["after"].data, [2**-0.5, 0, 0, 2**-0.5], atol=1e-12)


def test_save_statevector_with_bound_parameters():
    theta = Parameter("theta")
    circuit = QuantumCircuit(2)
    circuit.ry(theta, 0)
    circuit.crx(theta, 0, 1)
    circuit.save_statevector()
    bound = circuit.assign_parameters({theta: 0.8})
    np.testing.assert_allclose(
        _mine(bound).get_statevector().data, _aer_data(bound)["statevector"].data, atol=1e-12
    )


@pytest.mark.parametrize("method", ["statevector", "matrix_product_state", "density_matrix"])
def test_probabilities_and_expectation_values_match_aer(method):
    circuit = _state_circuit()
    circuit.save_probabilities(label="all")
    circuit.save_probabilities([2, 0], label="some")
    circuit.save_probabilities_dict([1, 3], label="dict")
    circuit.save_expectation_value(
        SparsePauliOp(["ZZII", "IXIY", "YIIX"], [0.5, -0.2, 0.3]), [0, 1, 2, 3], label="e"
    )
    circuit.save_expectation_value(Pauli("XZ"), [2, 0], label="e_order")
    circuit.save_expectation_value_variance(
        SparsePauliOp(["ZX", "YY"], [1.0, 0.5]), [1, 2], label="v"
    )
    data = _mine(circuit, method=method).data(0)
    aer = _aer_data(circuit)
    for label in ("all", "some"):
        assert isinstance(data[label], np.ndarray)
        np.testing.assert_allclose(data[label], aer[label], atol=1e-12)
    assert isinstance(data["dict"], ProbDistribution)
    assert set(data["dict"]) == set(aer["dict"])
    for key, value in aer["dict"].items():
        assert data["dict"][key] == pytest.approx(value, abs=1e-12)
    for label in ("e", "e_order"):
        assert isinstance(data[label], float)
        assert data[label] == pytest.approx(aer[label], abs=1e-12)
    np.testing.assert_allclose(data["v"], aer["v"], atol=1e-12)


def test_default_labels_follow_aer():
    circuit = QuantumCircuit(1)
    circuit.h(0)
    circuit.save_statevector()
    circuit.save_probabilities()
    circuit.save_expectation_value(Pauli("X"), [0])
    circuit.save_expectation_value_variance(Pauli("X"), [0])
    assert set(_mine(circuit).data(0)) == set(_aer_data(circuit))


def test_probabilities_of_many_qubits():
    circuit = QuantumCircuit(14)
    circuit.h(range(14))
    for q in range(13):
        circuit.cry(0.3 * (q + 1), q, q + 1)
    qubits = [13, 2, 7, 0, 5, 9, 11, 1, 3, 4, 6, 8, 10]
    circuit.save_probabilities(qubits, label="p")
    np.testing.assert_allclose(_mine(circuit).data(0)["p"], _aer_data(circuit)["p"], atol=1e-12)


def test_saves_after_resets_on_the_density_matrix_are_exact():
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.reset(0)
    circuit.save_probabilities()
    circuit.save_expectation_value(Pauli("ZZ"), [0, 1])
    data = _mine(circuit, method="density_matrix").data(0)
    aer = _aer_data(circuit, method="density_matrix")
    np.testing.assert_allclose(data["probabilities"], aer["probabilities"], atol=1e-12)
    assert data["expectation_value"] == pytest.approx(aer["expectation_value"], abs=1e-12)


def test_noisy_statevector_save_is_one_trajectory_like_aer():
    model = NoiseModel()
    model.add_all_qubit_quantum_error(depolarizing_error(0.75, 1), ["x"])
    model.add_all_qubit_quantum_error(amplitude_damping_error(0.5), ["h"])
    circuit = QuantumCircuit(1)
    circuit.h(0)
    circuit.x(0)
    circuit.save_statevector()
    states = set()
    for seed in range(30):
        result = MaestroBackend(noise_model=model, seed_simulator=seed).run(circuit).result()
        state = result.get_statevector()
        assert state.is_valid()
        states.add(tuple(np.round(state.data, 6)))
    # Different seeds give different trajectories, each a normalized state.
    assert len(states) > 2


def test_noisy_probabilities_average_over_shots_like_aer():
    model = NoiseModel()
    model.add_all_qubit_quantum_error(depolarizing_error(0.2, 1), ["h"])
    model.add_all_qubit_quantum_error(amplitude_damping_error(0.3), ["x"])
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.x(1)
    circuit.save_probabilities()
    exact = _aer_data(circuit, noise_model=model, method="density_matrix")["probabilities"]
    shots = 4000
    result = MaestroBackend(noise_model=model, seed_simulator=1).run(circuit, shots=shots)
    np.testing.assert_allclose(
        result.result().data(0)["probabilities"], exact, atol=5 * 0.5 / np.sqrt(shots)
    )


@pytest.mark.parametrize(
    "build,match",
    [
        (lambda c: (c.measure_all(), c.save_statevector()), "after a measurement"),
        (lambda c: c.save_statevector(pershot=True), "pershot"),
        (lambda c: c.save_density_matrix(), "supported save instructions"),
        (lambda c: c.save_expectation_value(Pauli("Z"), [0], unnormalized=True), "unnormalized"),
    ],
)
def test_unsupported_saves_raise(build, match):
    circuit = QuantumCircuit(1)
    circuit.h(0)
    build(circuit)
    with pytest.raises(UnsupportedInstructionError, match=match):
        MaestroBackend().run(circuit)


def test_saves_inside_control_flow_raise():
    circuit = QuantumCircuit(1, 1)
    circuit.measure(0, 0)
    with circuit.if_test((circuit.clbits[0], 1)):
        circuit.save_statevector()
    with pytest.raises(UnsupportedInstructionError, match="save"):
        MaestroBackend().run(circuit)


def test_statevector_save_needs_a_pure_state_method():
    circuit = QuantumCircuit(1)
    circuit.h(0)
    circuit.save_statevector()
    with pytest.raises(
        UnsupportedInstructionError, match="density_matrix method has no statevector"
    ):
        MaestroBackend(method="density_matrix").run(circuit)


def test_duplicate_labels_raise():
    circuit = QuantumCircuit(1)
    circuit.save_probabilities(label="x")
    circuit.save_statevector(label="x")
    with pytest.raises(ValueError, match="label 'x'"):
        MaestroBackend().run(circuit)


def test_primitives_reject_saves():
    circuit = QuantumCircuit(1)
    circuit.h(0)
    circuit.save_statevector()
    with pytest.raises(UnsupportedInstructionError, match="MaestroBackend.run"):
        MaestroEstimator().run([(circuit, "Z")])
    circuit.measure_all()
    with pytest.raises(UnsupportedInstructionError, match="MaestroBackend.run"):
        MaestroSampler().run([circuit])


def test_circuits_without_saves_keep_their_result_format():
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure_all()
    data = _mine(circuit).data(0)
    assert set(data) == {"counts"}
