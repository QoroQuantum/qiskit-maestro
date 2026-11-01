# SPDX-FileCopyrightText: 2026 Qoro Quantum
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import numpy as np
import pytest
import qiskit
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import Pauli, SparsePauliOp, Statevector

from qiskit_maestro import MaestroBackend, MaestroEstimator, UnsupportedInstructionError, _execution

from .conftest import efficient_su2, requires_aer, seeded_random_circuits


def _random_observable(num_qubits: int, num_terms: int, seed: int) -> SparsePauliOp:
    rng = np.random.default_rng(seed)
    labels = ["".join(rng.choice(list("IXYZ"), num_qubits)) for _ in range(num_terms)]
    return SparsePauliOp(labels, rng.normal(size=num_terms))


def _ansatz(num_qubits: int = 4) -> QuantumCircuit:
    return efficient_su2(num_qubits, reps=2)


def test_random_circuits_match_statevector_estimator():
    circuits = seeded_random_circuits(8, 5, 8, measure=False, seed=600)
    pubs = [(c, _random_observable(5, 12, seed=i)) for i, c in enumerate(circuits)]
    mine = MaestroEstimator().run(pubs).result()
    reference = StatevectorEstimator().run(pubs).result()
    for m, r in zip(mine, reference):
        np.testing.assert_allclose(m.data.evs, r.data.evs, atol=1e-10)
        np.testing.assert_array_equal(m.data.stds, np.zeros_like(r.data.stds))


@requires_aer
def test_random_circuits_match_aer_estimator():
    from qiskit_aer import AerSimulator
    from qiskit_aer.primitives import EstimatorV2 as AerEstimator

    circuits = seeded_random_circuits(5, 5, 8, measure=False, seed=700)
    observables = [_random_observable(5, 10, seed=10 + i) for i in range(5)]
    mine = MaestroEstimator().run(list(zip(circuits, observables))).result()
    aer_circuits = transpile(circuits, AerSimulator())
    aer = AerEstimator().run(list(zip(aer_circuits, observables))).result()
    for m, a in zip(mine, aer):
        np.testing.assert_allclose(m.data.evs, a.data.evs, atol=1e-10)


# Broadcasting cases: (parameter-values shape, observables shape).
_BROADCASTS = [
    ((), ()),
    ((), (3,)),
    ((5,), ()),
    ((5,), (5,)),
    ((5,), (1,)),
    ((1, 6), (4, 1)),
    ((2, 3), (3,)),
    ((3, 1), (1, 2)),
    ((2, 2, 2), (2, 1, 2)),
]


@pytest.mark.parametrize("param_shape,obs_shape", _BROADCASTS, ids=str)
def test_broadcasting_matches_statevector_estimator(param_shape, obs_shape):
    circuit = _ansatz(3)
    rng = np.random.default_rng(1)
    values = rng.uniform(0, np.pi, size=param_shape + (circuit.num_parameters,))
    labels = ["XYZ", "ZZI", "IXI", "YIZ", "ZIZ", "XXX", "IIY", "ZYX"]
    pool = [SparsePauliOp([labels[i % 8], labels[(i + 3) % 8]], [1.0, -0.5]) for i in range(64)]
    observables = np.empty(obs_shape, dtype=object)
    for k, index in enumerate(np.ndindex(obs_shape)):
        observables[index] = pool[k]
    if obs_shape == ():
        observables = pool[0]
    pub = (circuit, observables, values)
    mine = MaestroEstimator().run([pub]).result()[0]
    reference = StatevectorEstimator().run([pub]).result()[0]
    assert mine.data.shape == reference.data.shape
    assert mine.data.evs.shape == reference.data.evs.shape
    assert mine.data.stds.shape == reference.data.stds.shape
    np.testing.assert_allclose(mine.data.evs, reference.data.evs, atol=1e-10)


@pytest.mark.parametrize(
    "observable",
    [
        "XIZ",
        Pauli("XIZ"),
        SparsePauliOp("XIZ"),
        {"XIZ": 1.0, "ZZI": -0.25},
        [["XIZ"], ["IIZ"]],
        [SparsePauliOp(["XXX", "ZII"], [0.3, 0.7]), "IZI"],
    ],
    ids=["str", "pauli", "sparse-pauli-op", "dict", "nested-list", "mixed-list"],
)
def test_observable_forms(observable):
    circuit = _ansatz(3).assign_parameters(np.linspace(0.1, 2.0, _ansatz(3).num_parameters))
    mine = MaestroEstimator().run([(circuit, observable)]).result()[0].data.evs
    reference = StatevectorEstimator().run([(circuit, observable)]).result()[0].data.evs
    np.testing.assert_allclose(mine, reference, atol=1e-10)


def test_identity_terms_need_no_simulation():
    circuit = QuantumCircuit(2)
    circuit.h(0)
    observable = SparsePauliOp(["II", "ZI", "IX"], [2.0, 1.0, 0.5])
    evs = MaestroEstimator().run([(circuit, observable)]).result()[0].data.evs
    assert evs == pytest.approx(2.0 + 1.0 + 0.5)


def test_one_maestro_call_per_parameter_set(monkeypatch):
    calls = []
    original = _execution.estimate

    def spy(circuit, angle_rows, labels, options, seeds):
        calls.append([list(row_labels) for row_labels in labels])
        return original(circuit, angle_rows, labels, options, seeds)

    monkeypatch.setattr(_execution, "estimate", spy)
    circuit = _ansatz(3)
    values = np.random.default_rng(2).uniform(size=(4, circuit.num_parameters))
    # Two observables sharing the term ZZI, broadcast against 4 parameter sets.
    observables = [[SparsePauliOp(["ZZI", "XIX"])], [SparsePauliOp(["ZZI", "YYI"])]]
    MaestroEstimator().run([(circuit, observables, values)]).result()
    assert len(calls) == 1
    labels = calls[0]
    assert len(labels) == 4  # One call's worth of strings per parameter set...
    for row_labels in labels:  # ...each term listed once, in Maestro's order.
        assert sorted(row_labels) == sorted(["IZZ", "XIX", "IYY"])


def test_precision_follows_statevector_estimator():
    circuit = _ansatz(3)
    values = np.random.default_rng(3).uniform(size=(5, circuit.num_parameters))
    pub = (circuit, SparsePauliOp(["ZZI", "XIX"], [0.5, 1.0]), values)
    mine = MaestroEstimator(seed=12).run([pub], precision=0.05).result()[0]
    reference = StatevectorEstimator(seed=12).run([pub], precision=0.05).result()[0]
    exact = StatevectorEstimator().run([pub]).result()[0].data.evs
    # Same seed, same noise: the values match the reference draw for draw.
    np.testing.assert_allclose(mine.data.evs, reference.data.evs, atol=1e-10)
    assert not np.allclose(mine.data.evs, exact)
    assert mine.metadata["target_precision"] == 0.05
    # Qiskit 1.1 named this key "precision"; later versions say "target_precision".
    assert reference.metadata.get("target_precision", reference.metadata.get("precision")) == 0.05
    # The one intentional difference: stds reports the noise that was added.
    np.testing.assert_array_equal(mine.data.stds, np.full(5, 0.05))


def test_default_precision():
    circuit = QuantumCircuit(1)
    circuit.h(0)
    estimator = MaestroEstimator(default_precision=0.1, seed=1)
    result = estimator.run([(circuit, "X")]).result()[0]
    assert result.metadata["target_precision"] == 0.1
    assert result.data.stds == 0.1
    exact = MaestroEstimator().run([(circuit, "X")]).result()[0]
    assert exact.data.evs == pytest.approx(1.0)
    assert exact.data.stds == 0.0


@pytest.mark.skipif(
    qiskit.__version__.startswith("1."),
    reason="Estimator PUBs take SparseObservable from Qiskit 2.0",
)
def test_projector_terms():
    from qiskit.quantum_info import SparseObservable

    circuit = QuantumCircuit(2)
    circuit.ry(0.8, 0)
    circuit.ry(1.9, 1)
    circuit.cx(0, 1)
    # Qiskit order: "0+" is |0><0| on qubit 1 and |+><+| on qubit 0.
    observable = SparseObservable.from_list([("0+", 1.0), ("1r", -2.0), ("lZ", 0.5)])
    evs = MaestroEstimator().run([(circuit, observable)]).result()[0].data.evs

    single = {
        "0": np.diag([1, 0]),
        "1": np.diag([0, 1]),
        "+": np.full((2, 2), 0.5),
        "r": 0.5 * np.array([[1, -1j], [1j, 1]]),
        "l": 0.5 * np.array([[1, 1j], [-1j, 1]]),
        "Z": np.diag([1, -1]),
    }
    matrix = sum(
        c * np.kron(single[label[0]], single[label[1]])
        for label, c in [("0+", 1.0), ("1r", -2.0), ("lZ", 0.5)]
    )
    state = Statevector(circuit).data
    assert evs == pytest.approx(np.real(state.conj() @ matrix @ state), abs=1e-10)


def test_reset_follows_one_trajectory_on_pure_state_methods():
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.reset(0)
    values = {
        seed: MaestroEstimator(seed=seed).run([(circuit, "ZI")]).result()[0].data.evs
        for seed in range(8)
    }
    assert all(abs(abs(v) - 1) < 1e-10 for v in values.values())
    assert {np.sign(v) for v in values.values()} == {-1.0, 1.0}


def test_reset_is_exact_on_density_matrix():
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.reset(0)
    evs = MaestroEstimator(method="density_matrix").run([(circuit, ["ZI", "IZ"])]).result()[0]
    np.testing.assert_allclose(evs.data.evs, [0.0, 1.0], atol=1e-12)


def test_measurements_are_rejected():
    circuit = QuantumCircuit(1)
    circuit.h(0)
    circuit.measure_all()
    with pytest.raises(UnsupportedInstructionError, match="remove_final_measurements"):
        MaestroEstimator().run([(circuit, "Z")])


def test_observable_width_must_match():
    circuit = QuantumCircuit(2)
    with pytest.raises(ValueError):
        MaestroEstimator().run([(circuit, "ZZZ")])


def test_aer_options_form_and_from_backend():
    estimator = MaestroEstimator(
        options={"default_precision": 0.02, "backend_options": {"method": "mps"}}
    )
    assert estimator.default_precision == 0.02
    assert estimator.options["method"] == "matrix_product_state"
    backend = MaestroBackend(method="stabilizer", seed_simulator=5)
    assert MaestroEstimator.from_backend(backend).options["method"] == "stabilizer"


def test_metadata():
    circuit = QuantumCircuit(1)
    circuit.h(0)
    circuit.t(0)
    circuit.metadata = {"tag": 1}
    result = MaestroEstimator().run([(circuit, "X")]).result()
    assert result.metadata == {"version": 2}
    assert result[0].metadata["circuit_metadata"] == {"tag": 1}
    assert result[0].metadata["maestro"]["methods"] == ["statevector"]


def test_parameter_expressions_in_a_vqe_style_loop():
    theta = Parameter("θ")
    circuit = QuantumCircuit(2)
    circuit.ry(theta, 0)
    circuit.cx(0, 1)
    circuit.rz(theta / 2 + 0.1, 1)
    hamiltonian = SparsePauliOp(["ZZ", "XI", "IY"], [1.0, 0.4, -0.3])
    estimator, reference = MaestroEstimator(), StatevectorEstimator()
    for value in np.linspace(0, np.pi, 5):
        mine = estimator.run([(circuit, hamiltonian, [value])]).result()[0].data.evs
        ref = reference.run([(circuit, hamiltonian, [value])]).result()[0].data.evs
        np.testing.assert_allclose(mine, ref, atol=1e-12)


# ---- Stabilizer estimates with Maestro's auto_reduce ----------------------------------


def _all_paulis(num_qubits: int) -> list[str]:
    import itertools

    return ["".join(label) for label in itertools.product("IXYZ", repeat=num_qubits)]


def test_stabilizer_x_gate_with_an_idle_qubit():
    # Maestro 0.3.10 raised "The stabilizer simulator does not support the Rx gate" here.
    circuit = QuantumCircuit(2)
    circuit.x(0)
    result = MaestroEstimator(method="stabilizer").run([(circuit, "ZZ")]).result()[0]
    reference = StatevectorEstimator().run([(circuit, "ZZ")]).result()[0]
    assert float(result.data.evs) == pytest.approx(-1.0)
    np.testing.assert_allclose(result.data.evs, reference.data.evs, atol=1e-12)


@pytest.mark.parametrize("gate", ["x", "y", "z", "h", "s"])
def test_stabilizer_matches_statevector_for_every_pauli_with_an_idle_qubit(gate):
    # Qubit 1 is idle. Every Pauli string on three qubits, including the ones that
    # failed under Maestro's auto_reduce.
    circuit = QuantumCircuit(3)
    getattr(circuit, gate)(0)
    observables = _all_paulis(3)
    mine = MaestroEstimator(method="stabilizer").run([(circuit, observables)]).result()[0]
    reference = StatevectorEstimator().run([(circuit, observables)]).result()[0]
    np.testing.assert_allclose(mine.data.evs, reference.data.evs, atol=1e-12)


def test_stabilizer_workaround_applies_to_from_backend_and_aer_options():
    circuit = QuantumCircuit(2)
    circuit.x(0)
    backend = MaestroBackend(method="stabilizer")
    for estimator in (
        MaestroEstimator.from_backend(backend),
        MaestroEstimator(options={"backend_options": {"method": "stabilizer"}}),
    ):
        evs = estimator.run([(circuit, ["ZZ", "ZI", "IZ"])]).result()[0].data.evs
        np.testing.assert_allclose(evs, [-1, 1, -1], atol=1e-12)


def test_auto_reduce_is_only_switched_off_for_stabilizer_estimates():
    default = MaestroEstimator(method="stabilizer").options
    assert default["maestro_options"] is None  # The user's options aren't rewritten.

    def reduced(options):
        return _execution._estimate_options(options)["maestro_options"]

    base = MaestroEstimator(method="stabilizer").options
    assert reduced(base) == {"auto_reduce": False}
    # An explicit choice wins, in either direction.
    for value in (True, False):
        options = MaestroEstimator(
            method="stabilizer", maestro_options={"auto_reduce": value, "gate_fusion": True}
        ).options
        assert reduced(options) == {"auto_reduce": value, "gate_fusion": True}
    # Other options are kept, and other methods are untouched.
    options = MaestroEstimator(method="stabilizer", maestro_options={"gate_fusion": True}).options
    assert reduced(options) == {"gate_fusion": True, "auto_reduce": False}
    for method in ("statevector", "matrix_product_state", "density_matrix"):
        options = MaestroEstimator(method=method).options
        assert _execution._estimate_options(options) is options
