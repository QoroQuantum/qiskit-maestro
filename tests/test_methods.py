# SPDX-FileCopyrightText: 2026 Qoro Quantum
# SPDX-License-Identifier: GPL-3.0-or-later

"""Maestro's simulation methods through the Qiskit interfaces."""

from __future__ import annotations

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp, random_clifford

from qiskit_maestro import MaestroBackend, MaestroEstimator, MaestroSampler

from .conftest import assert_same_distribution, ghz, seeded_random_circuits


def _low_entanglement(num_qubits: int, layers: int, seed: int) -> QuantumCircuit:
    """Weakly entangling brickwork: small rotations and shallow CZ layers."""
    rng = np.random.default_rng(seed)
    circuit = QuantumCircuit(num_qubits)
    for layer in range(layers):
        for q in range(num_qubits):
            circuit.ry(rng.uniform(0, np.pi), q)
            circuit.rz(rng.uniform(0, 2 * np.pi), q)
        for q in range(layer % 2, num_qubits - 1, 2):
            circuit.crx(rng.uniform(0, 0.6), q, q + 1)
    return circuit


def _local_observables(num_qubits: int) -> list[SparsePauliOp]:
    terms = []
    for q in range(num_qubits - 1):
        terms.append(SparsePauliOp.from_sparse_list([("ZZ", [q, q + 1], 1.0)], num_qubits))
        terms.append(SparsePauliOp.from_sparse_list([("X", [q], 0.5)], num_qubits))
    return terms


# ---- MPS -----------------------------------------------------------------------------


def test_mps_matches_statevector_on_low_entanglement_circuits():
    circuit = _low_entanglement(12, 6, seed=1)
    observables = _local_observables(12)
    mps = MaestroEstimator(method="matrix_product_state").run([(circuit, observables)]).result()[0]
    exact = StatevectorEstimator().run([(circuit, observables)]).result()[0]
    np.testing.assert_allclose(mps.data.evs, exact.data.evs, atol=1e-8)
    assert mps.metadata["maestro"]["methods"] == ["matrix_product_state"]


def test_mps_sampling_matches_statevector():
    circuit = _low_entanglement(8, 4, seed=2)
    circuit.measure_all()
    mps = MaestroSampler(method="mps", seed=1).run([circuit], shots=20_000).result()[0]
    sv = MaestroSampler(seed=1).run([circuit], shots=20_000).result()[0]
    assert_same_distribution(mps.data.meas.get_counts(), sv.data.meas.get_counts())


def test_mps_truncation_options_are_honoured():
    circuit = ghz(10, measure=False)
    observable = SparsePauliOp("X" * 10)
    exact = MaestroEstimator(method="mps").run([(circuit, observable)]).result()[0].data.evs
    truncated = (
        MaestroEstimator(method="mps", max_bond_dimension=1)
        .run([(circuit, observable)])
        .result()[0]
        .data.evs
    )
    assert exact == pytest.approx(1.0)
    # A bond dimension of 1 can't hold a GHZ state.
    assert abs(truncated - 1.0) > 0.5


@pytest.mark.slow
@pytest.mark.parametrize("num_qubits", [101, 150])
def test_mps_samples_large_ghz(num_qubits):
    circuit = ghz(num_qubits, measure=True)
    counts = MaestroSampler(method="mps", seed=3).run([circuit], shots=2000).result()[0].data
    counts = counts.meas.get_counts()
    assert set(counts) <= {"0" * num_qubits, "1" * num_qubits}
    assert abs(counts.get("0" * num_qubits, 0) / 2000 - 0.5) < 0.06


@pytest.mark.slow
def test_mps_large_ghz_through_the_backend():
    counts = MaestroBackend(method="mps").run(ghz(120, measure=True), shots=500).result()
    assert set(counts.get_counts()) == {"0" * 120, "1" * 120}


@pytest.mark.slow
def test_mps_estimator_at_120_qubits():
    n = 120
    rng = np.random.default_rng(4)
    angles = rng.uniform(0, np.pi, n)
    circuit = QuantumCircuit(n)
    for q, a in enumerate(angles):
        circuit.ry(a, q)
    entangled = ghz(n, measure=False)
    z = [SparsePauliOp.from_sparse_list([("Z", [q], 1.0)], n) for q in (0, 57, 119)]
    zz = SparsePauliOp.from_sparse_list([("ZZ", [0, n - 1], 1.0)], n)
    estimator = MaestroEstimator(method="mps")
    product, cat = estimator.run([(circuit, z), (entangled, [zz, SparsePauliOp("X" * n)])]).result()
    np.testing.assert_allclose(product.data.evs, np.cos(angles[[0, 57, 119]]), atol=1e-10)
    np.testing.assert_allclose(cat.data.evs, [1.0, 1.0], atol=1e-10)


# ---- Stabilizer --------------------------------------------------------------------------


def _clifford_layers(num_qubits: int, layers: int, seed: int) -> QuantumCircuit:
    rng = np.random.default_rng(seed)
    circuit = QuantumCircuit(num_qubits)
    for _ in range(layers):
        for q in range(num_qubits):
            getattr(circuit, rng.choice(["h", "s", "sdg", "x", "sx"]))(q)
        order = rng.permutation(num_qubits)
        for a, b in zip(order[::2], order[1::2]):
            circuit.cx(int(a), int(b))
    return circuit


def test_stabilizer_matches_statevector_on_small_cliffords():
    for seed in range(4):
        circuit = random_clifford(5, seed=seed).to_circuit()
        observables = [SparsePauliOp(label) for label in ["ZZIII", "IXIYI", "XXXXX", "IIZIY"]]
        stab = MaestroEstimator(method="stabilizer").run([(circuit, observables)]).result()[0]
        exact = StatevectorEstimator().run([(circuit, observables)]).result()[0]
        np.testing.assert_allclose(stab.data.evs, exact.data.evs, atol=1e-12)


@pytest.mark.slow
def test_stabilizer_runs_1000_qubit_cliffords():
    n = 1000
    circuit = _clifford_layers(n, 4, seed=5)
    measured = circuit.copy()
    measured.measure_all()
    counts = (
        MaestroBackend(method="stabilizer", seed_simulator=1)
        .run(measured, shots=100)
        .result()
        .get_counts()
    )
    assert sum(counts.values()) == 100
    assert all(len(bits) == n for bits in counts)
    sampler = MaestroSampler(method="stabilizer", seed=1).run([measured], shots=100).result()[0]
    assert sampler.data.meas.num_bits == n
    # A Clifford state's Pauli expectation values are 0 or ±1.
    paulis = [SparsePauliOp.from_sparse_list([("Z", [q], 1.0)], n) for q in range(0, n, 97)]
    evs = MaestroEstimator(method="stabilizer").run([(circuit, paulis)]).result()[0].data.evs
    assert np.all(np.isin(np.round(evs, 12), [-1.0, 0.0, 1.0]))


@pytest.mark.slow
def test_stabilizer_1000_qubit_ghz():
    n = 1000
    counts = MaestroSampler(method="stabilizer", seed=2).run([ghz(n, measure=True)], shots=200)
    assert set(counts.result()[0].data.meas.get_counts()) == {"0" * n, "1" * n}
    zz = SparsePauliOp.from_sparse_list([("ZZ", [0, n - 1], 1.0)], n)
    evs = MaestroEstimator(method="stabilizer").run([(ghz(n, measure=False), zz)]).result()[0]
    assert evs.data.evs == pytest.approx(1.0)


def test_stabilizer_rejects_non_clifford_gates():
    # (Before a Z measurement Maestro removes a T gate as diagonal, so measure X.)
    circuit = QuantumCircuit(1)
    circuit.h(0)
    circuit.t(0)
    with pytest.raises(RuntimeError, match="non-clifford"):
        MaestroEstimator(method="stabilizer").run([(circuit, "X")]).result()


# ---- The other methods ---------------------------------------------------------------------


@pytest.mark.parametrize("method", ["density_matrix", "tensor_network", "pauli_propagation"])
def test_exact_methods_match_statevector_estimator(method):
    circuits = seeded_random_circuits(3, 4, 5, measure=False, seed=800)
    rng = np.random.default_rng(6)
    labels = ["".join(rng.choice(list("IXYZ"), 4)) for _ in range(8)]
    pubs = [(c, SparsePauliOp(labels, rng.normal(size=8))) for c in circuits]
    mine = MaestroEstimator(method=method).run(pubs).result()
    reference = StatevectorEstimator().run(pubs).result()
    for m, r in zip(mine, reference):
        np.testing.assert_allclose(m.data.evs, r.data.evs, atol=1e-8)


@pytest.mark.parametrize("method", ["density_matrix", "tensor_network", "pauli_propagation"])
def test_sampling_with_other_methods(method):
    circuit = seeded_random_circuits(1, 4, 5, measure=True, seed=900)[0]
    mine = MaestroSampler(method=method, seed=2).run([circuit], shots=20_000).result()[0]
    reference = MaestroSampler(seed=2).run([circuit], shots=20_000).result()[0]
    assert_same_distribution(mine.data.c.get_counts(), reference.data.c.get_counts())


def test_single_precision():
    circuit = _low_entanglement(6, 3, seed=7)
    observables = _local_observables(6)
    single = MaestroEstimator(precision="single").run([(circuit, observables)]).result()[0]
    exact = StatevectorEstimator().run([(circuit, observables)]).result()[0]
    np.testing.assert_allclose(single.data.evs, exact.data.evs, atol=1e-5)
