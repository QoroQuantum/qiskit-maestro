# SPDX-FileCopyrightText: 2026 Qoro Quantum
# SPDX-License-Identifier: GPL-3.0-or-later

"""Aer noise models on Maestro, compared with Aer itself.

Density-matrix results must match Aer's to numerical precision, and sampled
results within statistical error. The noise in these tests is strong enough
that dropping, repeating or misplacing any error would fail them.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from qiskit import QuantumCircuit, transpile
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.quantum_info import PTM, SparsePauliOp

from qiskit_maestro import (
    MaestroBackend,
    MaestroEstimator,
    MaestroSampler,
    UnsupportedInstructionError,
    UnsupportedNoiseError,
)

from .conftest import HAS_AER, assert_same_distribution

pytestmark = pytest.mark.skipif(not HAS_AER, reason="qiskit-aer isn't installed")

if HAS_AER:
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import (
        NoiseModel,
        QuantumError,
        ReadoutError,
        amplitude_damping_error,
        coherent_unitary_error,
        depolarizing_error,
        kraus_error,
        pauli_error,
        phase_amplitude_damping_error,
        phase_damping_error,
        reset_error,
        thermal_relaxation_error,
    )

    from qiskit_maestro.noise import decompose, error_ptm

try:
    from qiskit_ibm_runtime.fake_provider import FakeSherbrooke

    HAS_RUNTIME = True
except ImportError:
    HAS_RUNTIME = False

_BASIS = ["rz", "sx", "x", "h", "cx", "id"]
_PAIRS = [(0, 1), (1, 0), (1, 2), (2, 1)]


def _strong_noise_model() -> NoiseModel:
    """Errors of every kind the translation handles, about 100 times a device's."""
    model = NoiseModel(basis_gates=_BASIS)
    t1s, t2s = [40e-6, 50e-6, 60e-6], [30e-6, 90e-6, 60e-6]  # T2 < T1, T2 > T1, T2 = T1.
    for q in range(3):
        relax = thermal_relaxation_error(t1s[q], t2s[q], 2e-6)
        model.add_quantum_error(depolarizing_error(0.03, 1).compose(relax), ["sx", "x"], [q])
        model.add_quantum_error(thermal_relaxation_error(t1s[q], t2s[q], 4e-6), ["id"], [q])
        model.add_quantum_error(reset_error(0.04, 0.01), ["reset"], [q])
        model.add_readout_error(ReadoutError([[0.95, 0.05], [0.12, 0.88]]), [q])
    model.add_all_qubit_quantum_error(
        amplitude_damping_error(0.05).compose(phase_damping_error(0.04)), ["h"]
    )
    for a, b in _PAIRS:
        relax = thermal_relaxation_error(t1s[a], t2s[a], 8e-6).expand(
            thermal_relaxation_error(t1s[b], t2s[b], 8e-6)
        )
        model.add_quantum_error(depolarizing_error(0.06, 2).compose(relax), ["cx"], [a, b])
    model.add_all_qubit_quantum_error(pauli_error([("X", 0.03), ("I", 0.97)]), ["measure"])
    return model


def _circuit(*, measure: bool) -> QuantumCircuit:
    """A circuit on the noisy gates, with noiseless rz and a reset in the middle."""
    circuit = QuantumCircuit(3)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.sx(2)
    circuit.rz(0.7, 2)
    circuit.cx(1, 2)
    circuit.id(0)
    circuit.reset(1)
    circuit.x(1)
    circuit.rz(-0.4, 0)
    circuit.sx(0)
    circuit.cx(2, 1)
    circuit.h(2)
    if measure:
        circuit.measure_all()
    return circuit


def _saves(circuit: QuantumCircuit) -> QuantumCircuit:
    circuit = circuit.copy()
    circuit.save_probabilities(label="p")
    circuit.save_probabilities([2, 0], label="p20")
    circuit.save_expectation_value(
        SparsePauliOp(["ZZI", "XIY", "IZX"], [0.4, 0.3, 0.3]), [0, 1, 2], label="e"
    )
    circuit.save_expectation_value_variance(SparsePauliOp(["XX"]), [0, 2], label="v")
    return circuit


def _ptm(error) -> np.ndarray:
    return np.asarray(PTM(error.to_quantumchannel()).data).real


# ---- The channel decomposition ------------------------------------------------------


_STANDARD_ERRORS = [
    lambda: depolarizing_error(0.1, 1),
    lambda: depolarizing_error(0.1, 2),
    lambda: pauli_error([("XZ", 0.1), ("YI", 0.05), ("II", 0.85)]),
    lambda: phase_damping_error(0.3),
    lambda: amplitude_damping_error(0.3),
    lambda: phase_amplitude_damping_error(0.2, 0.1),
    lambda: reset_error(0.1, 0.05),
    lambda: thermal_relaxation_error(50e-6, 30e-6, 3e-6),
    lambda: thermal_relaxation_error(50e-6, 95e-6, 3e-6),
    lambda: thermal_relaxation_error(50e-6, 70e-6, 3e-6, 0.1),
    lambda: depolarizing_error(0.05, 1).compose(thermal_relaxation_error(5e-5, 8e-5, 3e-6)),
    lambda: thermal_relaxation_error(5e-5, 8e-5, 3e-6).compose(depolarizing_error(0.05, 1)),
    lambda: depolarizing_error(0.05, 2).compose(
        thermal_relaxation_error(5e-5, 8e-5, 3e-6).expand(amplitude_damping_error(0.1))
    ),
]


@pytest.mark.parametrize("build", _STANDARD_ERRORS)
def test_standard_errors_decompose_exactly(build):
    error = build()
    channel = decompose(_ptm(error), (0.0,) * error.num_qubits)
    assert not isinstance(channel, str), channel
    # Rebuild the channel from the parts and compare it with Aer's.
    n = error.num_qubits
    paulis = channel.paulis or (1.0,) + (0.0,) * (4**n - 1)
    labels = ["".join("IXYZ"[i // 4**k % 4] for k in reversed(range(n))) for i in range(4**n)]
    rebuilt = pauli_error([(label, p) for label, p in zip(labels, paulis) if p > 0])
    for k, a in channel.decays:
        damping = [amplitude_damping_error(1 - a) if j == k else None for j in range(n)]
        layer = None
        for j in range(n):  # a.expand(b) puts b on the next qubit up.
            factor = damping[j] or QuantumError([(QuantumCircuit(1), 1.0)])
            layer = factor if layer is None else layer.expand(factor)
        rebuilt = rebuilt.compose(layer)
    np.testing.assert_allclose(_ptm(rebuilt), _ptm(error), atol=1e-12)


def test_coherent_errors_dont_decompose():
    from qiskit.circuit.library import RXGate

    error = coherent_unitary_error(RXGate(0.1).to_matrix())
    assert "coherent" in decompose(_ptm(error), (0.0,))


def _unusual_errors():
    """Errors whose instructions have no Pauli structure, or act on reversed qubits."""
    from qiskit.circuit.library import CXGate, ECRGate, RXGate, RZZGate

    swapped = QuantumCircuit(2)
    swapped.cx(1, 0)
    swapped.append(RXGate(0.3), [1])
    flip = np.kron(np.diag([1, -1]), np.array([[0, 1], [1, 0]]))
    return [
        coherent_unitary_error(RXGate(0.1).to_matrix()),
        coherent_unitary_error(ECRGate().to_matrix()),
        QuantumError([(CXGate(), 0.3), (RZZGate(0.2), 0.7)]),
        QuantumError([(swapped, 0.4), (QuantumCircuit(2), 0.6)]),
        kraus_error([np.sqrt(0.8) * np.eye(4), np.sqrt(0.2) * flip]),
        thermal_relaxation_error(5e-5, 8e-5, 3e-6).expand(reset_error(0.2, 0.1)),
    ]


def test_pauli_transfer_matrices_match_qiskit():
    """The plugin's own PTM computation agrees with Qiskit's for every kind of error."""
    errors = [build() for build in _STANDARD_ERRORS] + _unusual_errors()
    model = NoiseModel.from_backend(GenericBackendV2(4, seed=11))
    errors += [
        error for by_qubits in model._local_quantum_errors.values() for error in by_qubits.values()
    ]
    for error in errors:
        np.testing.assert_allclose(error_ptm(error), _ptm(error), atol=1e-12)


# ---- Exact agreement on the density matrix ------------------------------------------


def test_density_matrix_matches_aer_exactly():
    model = _strong_noise_model()
    circuit = _saves(_circuit(measure=False))
    aer = AerSimulator(noise_model=model, method="density_matrix").run(circuit).result().data(0)
    mine = MaestroBackend(noise_model=model, method="density_matrix").run(circuit).result()
    data = mine.data(0)
    for label in ("p", "p20"):
        np.testing.assert_allclose(data[label], aer[label], atol=1e-12)
    assert data["e"] == pytest.approx(aer["e"], abs=1e-12)
    np.testing.assert_allclose(data["v"], aer["v"], atol=1e-12)
    assert mine.results[0].metadata["method"] == "density_matrix"
    # The noise is strong enough to matter.
    ideal = MaestroBackend(method="density_matrix").run(circuit).result().data(0)
    assert np.max(np.abs(ideal["p"] - aer["p"])) > 0.01


def test_density_matrix_estimator_matches_aer_exactly():
    from qiskit_aer.primitives import EstimatorV2 as AerEstimator

    model = _strong_noise_model()
    circuit = _circuit(measure=False)
    observable = SparsePauliOp(["ZZI", "XIY", "IZX"], [0.4, 0.3, 0.3])
    options = {"backend_options": {"noise_model": model, "method": "density_matrix"}}
    aer = AerEstimator(options=options).run([(circuit, observable)]).result()[0].data.evs
    mine = MaestroEstimator(options=options).run([(circuit, observable)]).result()[0].data.evs
    assert mine == pytest.approx(aer, abs=1e-12)


def test_automatic_method_picks_the_density_matrix_like_aer():
    from qiskit_aer.primitives import EstimatorV2 as AerEstimator

    model = _strong_noise_model()
    circuit = _circuit(measure=False)
    observable = SparsePauliOp(["ZZI", "XIY"], [0.5, 0.5])
    aer = AerEstimator(options={"backend_options": {"noise_model": model}})
    mine = MaestroEstimator(noise_model=model, method="automatic")
    assert mine.run([(circuit, observable)]).result()[0].data.evs == pytest.approx(
        aer.run([(circuit, observable)]).result()[0].data.evs, abs=1e-12
    )
    result = (
        MaestroBackend(noise_model=model, method="automatic")
        .run(_circuit(measure=True), shots=100)
        .result()
    )
    assert result.results[0].metadata["method"] == "density_matrix"


# ---- Statistical agreement when sampling ----------------------------------------------


@pytest.mark.parametrize("method", ["density_matrix", "statevector", "matrix_product_state"])
def test_sampled_counts_match_aer(method):
    model = _strong_noise_model()
    circuit = _circuit(measure=True)
    shots = 8000
    aer = AerSimulator(noise_model=model, method="density_matrix")
    expected = aer.run(circuit, shots=shots, seed_simulator=1).result().get_counts()
    backend = MaestroBackend(noise_model=model, method=method, seed_simulator=2)
    counts = backend.run(circuit, shots=shots).result().get_counts()
    assert_same_distribution(counts, expected)


def test_statevector_expectation_values_average_trajectories():
    model = _strong_noise_model()
    circuit = _saves(_circuit(measure=False))
    exact = MaestroBackend(noise_model=model, method="density_matrix").run(circuit).result()
    shots = 4000
    sampled = MaestroBackend(noise_model=model, seed_simulator=3).run(circuit, shots=shots)
    data, reference = sampled.result().data(0), exact.data(0)
    # Each value averages `shots` trajectories of a quantity bounded by 1.
    assert data["e"] == pytest.approx(reference["e"], abs=5 / np.sqrt(shots))
    np.testing.assert_allclose(data["p"], reference["p"], atol=5 * 0.5 / np.sqrt(shots))


def test_sampler_matches_aer_sampler():
    from qiskit_aer.primitives import SamplerV2 as AerSampler

    model = _strong_noise_model()
    circuit = _circuit(measure=True)
    options = {"backend_options": {"noise_model": model}}
    aer = AerSampler(seed=4, options=options).run([circuit], shots=8000).result()
    mine = MaestroSampler(seed=5, options=options).run([circuit], shots=8000).result()
    assert_same_distribution(mine[0].data.meas.get_counts(), aer[0].data.meas.get_counts())


def test_seeded_noisy_runs_repeat():
    model = _strong_noise_model()
    circuit = _circuit(measure=True)
    runs = [
        MaestroBackend(noise_model=model, seed_simulator=9).run(circuit, shots=3000).result()
        for _ in range(2)
    ]
    assert runs[0].get_counts() == runs[1].get_counts()


@pytest.mark.parametrize("shots", [5, 200, 3000])
def test_seeded_noisy_runs_dont_depend_on_the_thread_count(shots):
    model = _strong_noise_model()
    circuits = [_circuit(measure=True), _circuit(measure=True).reverse_bits()]
    counts = [
        MaestroBackend(noise_model=model, seed_simulator=9, max_parallel_experiments=workers)
        .run(circuits, shots=shots)
        .result()
        .get_counts()
        for workers in (1, 3, 0)
    ]
    assert counts[0] == counts[1] == counts[2]


def test_noisy_shots_split_by_the_shot_count_alone():
    from qiskit_maestro._execution import noisy_chunks

    for shots in (1, 15, 16, 33, 200, 1024, 4097, 10**6):
        chunks = noisy_chunks(shots)
        assert sum(chunks) == shots
        assert 1 <= len(chunks) <= 64
        assert max(chunks) - min(chunks) <= 1
        assert len(chunks) == 1 or min(chunks) >= 16


def test_noisy_parameter_sets_keep_their_own_angles():
    """Parameter sets split into tasks that share each set's circuit, never another's."""
    from qiskit.circuit import Parameter
    from qiskit_aer.primitives import SamplerV2 as AerSampler

    theta = Parameter("theta")
    circuit = QuantumCircuit(2)
    circuit.rx(theta, 0)
    circuit.cx(0, 1)
    circuit.measure_all()
    model = NoiseModel()
    model.add_all_qubit_quantum_error(depolarizing_error(0.1, 2), ["cx"])
    values = [[0.0], [np.pi], [np.pi / 2], [2.0]]
    options = {"backend_options": {"noise_model": model}}
    mine = MaestroSampler(seed=1, options=options).run([(circuit, values)], shots=4000).result()
    aer = AerSampler(seed=2, options=options).run([(circuit, values)], shots=4000).result()
    for k in range(len(values)):
        assert_same_distribution(
            mine[0].data.meas[k].get_counts(), aer[0].data.meas[k].get_counts()
        )


# ---- Where Aer attaches noise -------------------------------------------------------


def _flip(name: str, qubits=None) -> NoiseModel:
    """A model that flips the qubits of every ``name`` instruction, so each one shows."""
    model = NoiseModel()
    error = pauli_error([("X", 1.0)])
    if qubits is None:
        model.add_all_qubit_quantum_error(error, [name])
    else:
        model.add_quantum_error(error, [name], qubits)
    return model


def _counts(circuit, model, method="statevector"):
    return (
        MaestroBackend(noise_model=model, method=method)
        .run(circuit, shots=50)
        .result()
        .get_counts()
    )


@pytest.mark.parametrize("method", ["statevector", "density_matrix"])
def test_noise_is_attached_per_instruction(method):
    # rz has no error, so only the x is noisy: two flips cancel.
    circuit = QuantumCircuit(1)
    circuit.rz(0.3, 0)
    circuit.x(0)
    circuit.rz(0.2, 0)
    circuit.measure_all()
    assert _counts(circuit, _flip("x"), method) == {"0": 50}
    # An error on measure acts before the measurement, so a second measurement sees it again.
    circuit = QuantumCircuit(1, 2)
    circuit.measure(0, 0)
    circuit.measure(0, 1)
    assert _counts(circuit, _flip("measure"), method) == {"01": 50}
    # Errors on id and reset apply too; barriers never get noise.
    circuit = QuantumCircuit(2)
    circuit.id(0)
    circuit.reset(1)
    circuit.barrier()
    circuit.measure_all()
    model = _flip("id")
    model.add_all_qubit_quantum_error(pauli_error([("X", 1.0)]), ["reset", "barrier"])
    assert _counts(circuit, model, method) == {"11": 50}


def test_local_errors_override_all_qubit_errors():
    model = _flip("x")
    model.add_quantum_error(pauli_error([("Z", 1.0)]), ["x"], [0])
    circuit = QuantumCircuit(2)
    circuit.x([0, 1])
    circuit.measure_all()
    assert _counts(circuit, model) == {"01": 50}


def test_local_errors_apply_only_to_their_qubits():
    circuit = QuantumCircuit(2)
    circuit.cx(0, 1)
    circuit.cx(1, 0)
    circuit.measure_all()
    model = NoiseModel()
    model.add_quantum_error(pauli_error([("XX", 1.0)]), ["cx"], [1, 0])
    assert _counts(circuit, model) == {"11": 50}


def test_noise_attaches_to_a_gate_as_a_whole():
    # rzz expands into cx gates, which must not pick up cx noise; Aer runs rzz natively.
    circuit = QuantumCircuit(2)
    circuit.rzz(0.3, 0, 1)
    circuit.ecr(0, 1)
    circuit.ecr(0, 1)
    circuit.measure_all()
    model = NoiseModel()
    model.add_all_qubit_quantum_error(pauli_error([("XX", 1.0)]), ["cx"])
    model.add_all_qubit_quantum_error(pauli_error([("XI", 1.0)]), ["ecr"])
    expected = AerSimulator(noise_model=model).run(circuit, shots=50).result().get_counts()
    assert _counts(circuit, model) == expected


def test_noise_inside_unrolled_loops():
    circuit = QuantumCircuit(1)
    with circuit.for_loop(range(3)):
        circuit.x(0)
    circuit.measure_all()
    assert _counts(circuit, _flip("x")) == {"0": 50}


def test_from_backend_delay_relaxation_matches_aer():
    device = GenericBackendV2(2, seed=5)
    model = NoiseModel.from_backend(device)
    circuit = QuantumCircuit(2)
    circuit.x(0)
    circuit.sx(1)
    circuit.delay(40, 0, unit="us")
    circuit.delay(30_000, 1, unit="ns")
    circuit.save_probabilities()
    aer = AerSimulator(noise_model=model, method="density_matrix").run(circuit).result()
    mine = MaestroBackend(noise_model=model, method="density_matrix").run(circuit).result()
    np.testing.assert_allclose(
        mine.data(0)["probabilities"], aer.data(0)["probabilities"], atol=1e-12
    )
    assert mine.data(0)["probabilities"][0] > 0.1  # The delay relaxed qubit 0.


def test_readout_errors_with_classical_control():
    model = NoiseModel()
    model.add_all_qubit_readout_error(ReadoutError([[0.9, 0.1], [0.3, 0.7]]))
    circuit = QuantumCircuit(2, 2)
    circuit.x(0)
    circuit.measure(0, 0)
    with circuit.if_test((circuit.clbits[0], 1)):
        circuit.x(1)
    circuit.measure(1, 1)
    shots = 8000
    aer = AerSimulator(noise_model=model).run(circuit, shots=shots, seed_simulator=1)
    mine = MaestroBackend(noise_model=model, seed_simulator=2).run(circuit, shots=shots)
    assert_same_distribution(mine.result().get_counts(), aer.result().get_counts())


def test_errors_on_unused_instructions_dont_matter():
    from qiskit.circuit.library import RXGate

    model = _flip("x")
    model.add_all_qubit_quantum_error(coherent_unitary_error(RXGate(0.1).to_matrix()), ["sx"])
    circuit = QuantumCircuit(1)
    circuit.x(0)
    circuit.measure_all()
    assert _counts(circuit, model) == {"0": 50}


@pytest.mark.parametrize("method", ["density_matrix", "statevector"])
def test_noise_with_mid_circuit_measurements_matches_aer(method):
    model = _strong_noise_model()
    circuit = QuantumCircuit(2, 3)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure(0, 0)
    circuit.sx(0)
    circuit.cx(0, 1)
    circuit.measure([0, 1], [1, 2])
    shots = 8000
    aer = AerSimulator(noise_model=model).run(circuit, shots=shots, seed_simulator=1)
    mine = MaestroBackend(noise_model=model, method=method, seed_simulator=2)
    assert_same_distribution(
        mine.run(circuit, shots=shots).result().get_counts(), aer.result().get_counts()
    )


def test_circuits_transpiled_for_a_large_device_drop_idle_qubits():
    from qiskit_aer.primitives import EstimatorV2 as AerEstimator

    device = GenericBackendV2(30, seed=3)
    model = NoiseModel.from_backend(device)
    circuit = QuantumCircuit(3)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.cx(1, 2)
    compiled = transpile(circuit, device, seed_transpiler=4)
    assert compiled.num_qubits == 30
    # Terms with X or Z on qubits the circuit never touches, and one on its own qubits.
    observable = SparsePauliOp(["XII", "ZZI", "IZZ"]).apply_layout(compiled.layout)
    idle = next(q for q in range(30) if q not in compiled.layout.final_index_layout())
    labels = ["I" * 30, "I" * 30]
    labels = [
        label[: 29 - idle] + letter + label[30 - idle :] for label, letter in zip(labels, "XZ")
    ]
    observable = observable + SparsePauliOp(labels, [0.5, 0.25])
    aer = AerEstimator(options={"backend_options": {"noise_model": model}})
    expected = aer.run([(compiled, observable)]).result()[0].data.evs
    mine = MaestroEstimator(noise_model=model, method="density_matrix")
    assert mine.run([(compiled, observable)]).result()[0].data.evs == pytest.approx(
        expected, abs=1e-12
    )
    measured = compiled.copy()
    measured.measure_active()
    shots = 8000
    reference = AerSimulator(noise_model=model).run(measured, shots=shots, seed_simulator=1)
    counts = MaestroBackend(noise_model=model, seed_simulator=2).run(measured, shots=shots)
    assert_same_distribution(counts.result().get_counts(), reference.result().get_counts())


# ---- Errors that can't be reproduced --------------------------------------------------


def test_coherent_errors_raise_naming_the_instruction():
    from qiskit.circuit.library import RXGate

    model = NoiseModel()
    model.add_quantum_error(coherent_unitary_error(RXGate(0.1).to_matrix()), ["sx"], [1])
    circuit = QuantumCircuit(2)
    circuit.sx(1)
    circuit.measure_all()
    with pytest.raises(UnsupportedNoiseError, match=r"'sx'.*qubits \[1\].*coherent") as error:
        MaestroBackend(noise_model=model).run(circuit)
    assert error.value.instruction == "sx"
    assert error.value.qubits == (1,)
    assert isinstance(error.value, UnsupportedInstructionError)


def test_three_qubit_errors_raise():
    model = NoiseModel()
    model.add_all_qubit_quantum_error(depolarizing_error(0.1, 3), ["ccx"])
    circuit = QuantumCircuit(3)
    circuit.ccx(0, 1, 2)
    with pytest.raises(UnsupportedNoiseError, match="3 qubits"):
        MaestroEstimator(noise_model=model).run([(circuit, "ZZZ")])


def test_correlated_readout_errors_raise():
    model = NoiseModel()
    model.add_readout_error(
        ReadoutError([[0.9, 0.1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]), [0, 1]
    )
    circuit = QuantumCircuit(2)
    circuit.measure_all()
    with pytest.raises(UnsupportedNoiseError, match="correlated readout"):
        MaestroSampler(noise_model=model).run([circuit])


def test_quantum_noise_with_classical_control_raises():
    circuit = QuantumCircuit(1, 1)
    circuit.measure(0, 0)
    with circuit.if_test((circuit.clbits[0], 0)):
        circuit.x(0)
    with pytest.raises(UnsupportedNoiseError, match="classical control"):
        MaestroBackend(noise_model=_flip("x")).run(circuit)


def test_relaxation_on_the_stabilizer_method_raises():
    model = NoiseModel()
    model.add_all_qubit_quantum_error(amplitude_damping_error(0.1), ["h"])
    circuit = QuantumCircuit(1)
    circuit.h(0)
    circuit.measure_all()
    with pytest.raises(UnsupportedNoiseError, match="stabilizer"):
        MaestroBackend(noise_model=model, method="stabilizer").run(circuit)
    # Pauli noise is fine there.
    counts = _counts(circuit, _flip("h"), "stabilizer")
    assert set(counts) <= {"0", "1"}


def test_run_evolution_rejects_noise():
    step = QuantumCircuit(1)
    step.x(0)
    with pytest.raises(UnsupportedNoiseError, match="run_evolution"):
        MaestroEstimator(noise_model=_flip("x")).run_evolution(QuantumCircuit(1), step, "Z", [1])


def test_noise_model_type_is_checked():
    with pytest.raises(ValueError, match="qiskit_aer.noise.NoiseModel"):
        MaestroBackend(noise_model="depolarizing")


# ---- Transpiling for a noise model ----------------------------------------------------


def test_target_follows_the_noise_model_basis_like_aer():
    device = GenericBackendV2(3, seed=2)
    model = NoiseModel.from_backend(device)
    backend = MaestroBackend(noise_model=model)
    gates = set(backend.target.operation_names) - {
        "measure",
        "reset",
        "delay",
        "if_else",
        "for_loop",
        "switch_case",
    }
    assert gates == {"cx", "id", "rz", "sx", "x"}
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure_all()
    compiled = transpile(circuit, backend)
    assert set(compiled.count_ops()) <= {"cx", "rz", "sx", "x", "measure", "barrier"}
    backend.set_options(noise_model=None)
    assert "h" in backend.target.operation_names


def test_a_run_option_noise_model_applies_to_that_run_only():
    backend = MaestroBackend()
    circuit = QuantumCircuit(1)
    circuit.x(0)
    circuit.measure_all()
    assert backend.run(circuit, shots=20, noise_model=_flip("x")).result().get_counts() == {"0": 20}
    assert backend.run(circuit, shots=20).result().get_counts() == {"1": 20}


# ---- An IBM device model ------------------------------------------------------------


@pytest.mark.skipif(not HAS_RUNTIME, reason="qiskit-ibm-runtime isn't installed")
@pytest.mark.parametrize("method", ["automatic", "density_matrix", "statevector"])
def test_fake_sherbrooke_matches_aer(method):
    from qiskit_aer.primitives import EstimatorV2 as AerEstimator

    device = FakeSherbrooke()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = NoiseModel.from_backend(device)
    circuit = QuantumCircuit(3)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.cx(1, 2)
    # A 127-qubit circuit with ecr, sx, x and noiseless rz on three of them.
    compiled = transpile(circuit, device, seed_transpiler=1)
    observable = SparsePauliOp(["ZZI", "XXX", "IYY"]).apply_layout(compiled.layout)
    # Aer's automatic method drops the idle qubits and uses the density matrix.
    aer = AerEstimator(options={"backend_options": {"noise_model": model}})
    expected = aer.run([(compiled, observable)]).result()[0].data.evs
    estimator = MaestroEstimator(noise_model=model, method=method, seed=1)
    evs = estimator.run([(compiled, observable)]).result()[0].data.evs
    if method == "statevector":  # 1024 trajectories.
        assert evs == pytest.approx(expected, abs=0.1)
    else:
        assert evs == pytest.approx(expected, abs=1e-12)

    measured = compiled.copy()
    measured.measure_active()
    shots = 8000
    reference = AerSimulator(noise_model=model).run(measured, shots=shots, seed_simulator=2)
    counts = MaestroBackend(noise_model=model, method=method, seed_simulator=1).run(
        measured, shots=shots
    )
    assert_same_distribution(counts.result().get_counts(), reference.result().get_counts())
