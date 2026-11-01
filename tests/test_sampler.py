# SPDX-FileCopyrightText: 2026 Qoro Quantum
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import numpy as np
import pytest
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter, ParameterVector
from qiskit.primitives import StatevectorSampler

from qiskit_maestro import MaestroBackend, MaestroSampler, UnsupportedInstructionError

from .conftest import assert_same_distribution, requires_aer, seeded_random_circuits


def _parametrised() -> QuantumCircuit:
    a, b = Parameter("a"), Parameter("b")
    circuit = QuantumCircuit(2)
    circuit.ry(a, 0)
    circuit.cx(0, 1)
    circuit.rz(b, 1)
    circuit.h(1)
    circuit.measure_all()
    return circuit


@pytest.mark.parametrize(
    "values",
    [
        np.array([0.3, 1.2]),  # One parameter set: shape ().
        np.linspace(0, 1, 10).reshape(5, 2),  # Shape (5,).
        np.linspace(0, 1, 24).reshape(3, 4, 2),  # Shape (3, 4).
        np.linspace(0, 1, 2).reshape(1, 1, 1, 2),  # Shape (1, 1, 1).
    ],
    ids=["scalar", "1d", "2d", "3d-ones"],
)
def test_shapes_match_statevector_sampler(values):
    pub = (_parametrised(), values)
    mine = MaestroSampler().run([pub], shots=13).result()[0]
    reference = StatevectorSampler().run([pub], shots=13).result()[0]
    assert mine.data.shape == reference.data.shape
    assert mine.data.meas.shape == reference.data.meas.shape
    assert mine.data.meas.array.shape == reference.data.meas.array.shape
    assert mine.data.meas.num_shots == 13
    assert mine.metadata["shots"] == reference.metadata["shots"] == 13


def test_parameters_given_as_a_dict():
    circuit = _parametrised()
    a, b = circuit.parameters
    pub = (circuit, {a: [0.1, 0.2, 0.3], b: [1.0, 2.0, 3.0]})
    mine = MaestroSampler().run([pub], shots=5).result()[0]
    assert mine.data.meas.shape == (3,)


def test_parameter_vector_order():
    theta = ParameterVector("θ", 3)
    circuit = QuantumCircuit(3)
    for q in range(3):
        circuit.rx(theta[q], q)
    circuit.measure_all()
    # Only theta[1] = pi: qubit 1 flips, so Qiskit shows "010".
    data = MaestroSampler().run([(circuit, [0, np.pi, 0])], shots=8).result()[0].data
    assert data.meas.get_counts() == {"010": 8}


def test_each_parameter_set_gets_its_own_results():
    circuit = QuantumCircuit(1)
    circuit.rx(Parameter("t"), 0)
    circuit.measure_all()
    data = MaestroSampler().run([(circuit, [[0.0], [np.pi]])], shots=6).result()[0].data
    assert data.meas[0].get_counts() == {"0": 6}
    assert data.meas[1].get_counts() == {"1": 6}


def test_per_pub_shots():
    circuit = QuantumCircuit(1)
    circuit.h(0)
    circuit.measure_all()
    results = MaestroSampler(default_shots=11).run([circuit, (circuit, None, 29)]).result()
    assert results[0].data.meas.num_shots == 11
    assert results[1].data.meas.num_shots == 29
    assert MaestroSampler().run([circuit], shots=17).result()[0].data.meas.num_shots == 17


def test_several_registers_and_unmeasured_bits():
    qr = QuantumRegister(3)
    alpha = ClassicalRegister(2, "alpha")
    beta = ClassicalRegister(3, "beta")
    circuit = QuantumCircuit(qr, alpha, beta)
    circuit.x([0, 2])
    circuit.measure(0, alpha[0])
    circuit.measure(2, beta[2])
    mine = MaestroSampler().run([circuit], shots=4).result()[0].data
    reference = StatevectorSampler().run([circuit], shots=4).result()[0].data
    assert mine.alpha.get_counts() == reference.alpha.get_counts() == {"01": 4}
    assert mine.beta.get_counts() == reference.beta.get_counts() == {"100": 4}


def test_circuit_without_measurements():
    circuit = QuantumCircuit(2, 2)
    circuit.h(0)
    mine = MaestroSampler().run([circuit], shots=3).result()[0].data
    reference = StatevectorSampler().run([circuit], shots=3).result()[0].data
    assert mine.c.get_counts() == reference.c.get_counts() == {"00": 3}


def test_circuit_without_classical_registers():
    circuit = QuantumCircuit(2)
    circuit.h(0)
    result = MaestroSampler().run([circuit], shots=3).result()[0]
    assert list(result.data) == []


def test_random_circuits_match_statevector_sampler():
    circuits = seeded_random_circuits(6, 5, 6, measure=True, seed=400)
    mine = MaestroSampler(seed=1).run(circuits, shots=20_000).result()
    reference = StatevectorSampler(seed=1).run(circuits, shots=20_000).result()
    for m, r in zip(mine, reference):
        assert_same_distribution(m.data.c.get_counts(), r.data.c.get_counts())


def test_parameter_sweep_matches_statevector_sampler():
    values = np.random.default_rng(5).uniform(0, np.pi, size=(4, 2))
    mine = MaestroSampler(seed=2).run([(_parametrised(), values)], shots=20_000).result()[0]
    reference = (
        StatevectorSampler(seed=2).run([(_parametrised(), values)], shots=20_000).result()[0]
    )
    for i in range(4):
        assert_same_distribution(
            mine.data.meas[i].get_counts(), reference.data.meas[i].get_counts()
        )


@requires_aer
def test_random_circuits_match_aer_sampler():
    from qiskit import transpile
    from qiskit_aer import AerSimulator
    from qiskit_aer.primitives import SamplerV2 as AerSampler

    circuits = seeded_random_circuits(4, 4, 5, measure=True, seed=500)
    mine = MaestroSampler(seed=1).run(circuits, shots=20_000).result()
    # Aer's sampler needs circuits in Aer's gate set; Maestro's expands them itself.
    aer_circuits = transpile(circuits, AerSimulator())
    aer = AerSampler(seed=1).run(aer_circuits, shots=20_000).result()
    for m, r in zip(mine, aer):
        assert_same_distribution(m.data.c.get_counts(), r.data.c.get_counts())


def test_seed_makes_runs_reproducible():
    values = np.linspace(0, 3, 8).reshape(4, 2)
    run = lambda seed: (
        MaestroSampler(seed=seed).run([(_parametrised(), values)], shots=100).result()[0]
    )
    np.testing.assert_array_equal(run(3).data.meas.array, run(3).data.meas.array)
    assert not np.array_equal(run(3).data.meas.array, run(4).data.meas.array)


def test_parameter_sets_are_not_correlated():
    # Two identical parameter sets with a seed must still be independent samples.
    circuit = QuantumCircuit(4)
    circuit.rx(Parameter("t"), range(4))
    circuit.measure_all()
    data = MaestroSampler(seed=1).run([(circuit, [[1.0], [1.0]])], shots=200).result()[0].data
    assert not np.array_equal(data.meas[0].array, data.meas[1].array)


def test_shots_are_shuffled():
    # Expanded counts must not come out sorted by outcome.
    circuit = QuantumCircuit(1)
    circuit.h(0)
    circuit.measure_all()
    bits = MaestroSampler(seed=3).run([circuit], shots=1000).result()[0].data.meas
    first_half = bits.array[:500].ravel()
    assert 150 < first_half.sum() < 350


def test_mid_circuit_measurement_and_classical_control():
    circuit = QuantumCircuit(2, 2)
    circuit.h(0)
    circuit.measure(0, 0)
    with circuit.if_test((circuit.clbits[0], 1)):
        circuit.x(1)
    circuit.measure(1, 1)
    counts = MaestroSampler(seed=6).run([circuit], shots=4000).result()[0].data.c.get_counts()
    assert set(counts) == {"00", "11"}


def test_unsupported_instruction_raises_at_run():
    circuit = QuantumCircuit(1, 1)
    with circuit.while_loop((circuit.clbits[0], 0)):
        circuit.measure(0, 0)
    with pytest.raises(UnsupportedInstructionError, match="while_loop"):
        MaestroSampler().run([circuit])


def test_aer_options_form():
    sampler = MaestroSampler(
        options={"backend_options": {"method": "matrix_product_state"}, "run_options": {"shots": 5}}
    )
    assert sampler.options["method"] == "matrix_product_state"
    with pytest.raises(ValueError, match="Unknown options"):
        MaestroSampler(options={"transpilation": {}})


def test_from_backend():
    backend = MaestroBackend(method="mps", max_bond_dimension=8, seed_simulator=4)
    sampler = MaestroSampler.from_backend(backend)
    assert sampler.options["method"] == "matrix_product_state"
    assert sampler.options["max_bond_dimension"] == 8
    assert sampler.seed == 4


def test_metadata_records_what_ran():
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.t(1)
    circuit.measure_all()
    circuit.metadata = {"tag": "bell"}
    result = MaestroSampler(method="mps").run([circuit], shots=10).result()
    assert result.metadata == {"version": 2}
    assert result[0].metadata["circuit_metadata"] == {"tag": "bell"}
    assert result[0].metadata["maestro"]["methods"] == ["matrix_product_state"]
