# SPDX-FileCopyrightText: 2026 Qoro Quantum
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import numpy as np
import pytest
from qiskit import QuantumCircuit, transpile
from qiskit.primitives import BackendSamplerV2, StatevectorSampler
from qiskit.quantum_info import Statevector
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from qiskit_maestro import MaestroBackend, MaestroSampler, UnsupportedInstructionError
from qiskit_maestro.converter import NATIVE_GATES

from .conftest import (
    assert_matches_probabilities,
    assert_same_distribution,
    qft_gate,
    requires_aer,
    seeded_random_circuits,
)

# ---- Target and transpilation -------------------------------------------------------


def test_target_lists_native_operations_without_a_coupling_map():
    backend = MaestroBackend()
    names = set(backend.target.operation_names)
    assert {"measure", "reset", "delay", "if_else", "cx", "u", "rz", "ccx", "cswap"} <= names
    assert {"for_loop", "switch_case"} <= names
    assert len(names) == len(NATIVE_GATES) + 6
    assert backend.coupling_map is None
    assert backend.max_circuits is None


@pytest.mark.parametrize("level", [0, 1, 2, 3])
def test_preset_pass_manager_output_runs(level):
    circuit = QuantumCircuit(5)
    circuit.append(qft_gate(5), range(5))
    circuit.rzz(0.3, 0, 4)
    circuit.ecr(1, 3)
    circuit.measure_all()
    backend = MaestroBackend(seed_simulator=3)
    compiled = generate_preset_pass_manager(optimization_level=level, backend=backend).run(circuit)
    assert set(compiled.count_ops()) <= set(backend.target.operation_names) | {"barrier"}
    counts = backend.run(compiled, shots=20_000).result().get_counts()
    probabilities = Statevector(
        circuit.remove_final_measurements(inplace=False)
    ).probabilities_dict()
    assert_matches_probabilities(counts, probabilities)


def test_transpile_with_backend():
    circuit = QuantumCircuit(3)
    circuit.append(qft_gate(3), range(3))
    circuit.measure_all()
    compiled = transpile(circuit, MaestroBackend())
    assert set(compiled.count_ops()) <= set(MaestroBackend().target.operation_names) | {"barrier"}


# ---- Options ------------------------------------------------------------------------


def test_aer_option_names_are_accepted():
    backend = MaestroBackend(
        method="mps",
        matrix_product_state_max_bond_dimension=16,
        matrix_product_state_truncation_threshold=1e-8,
    )
    assert backend.options.method == "matrix_product_state"
    assert backend.options.max_bond_dimension == 16
    assert backend.options.truncation_threshold == 1e-8


@pytest.mark.parametrize(
    "options,match",
    [
        ({"method": "unitary"}, "method must be one of"),
        ({"device": "TPU"}, "device must be"),
        ({"precision": "half"}, "precision must be"),
        ({"max_bond_dimension": 0}, "max_bond_dimension"),
        ({"truncation_threshold": -1}, "truncation_threshold"),
        ({"seed_simulator": -3}, "seed_simulator"),
        ({"maestro_options": {"seed": 1}}, "through the qiskit-maestro option"),
        ({"maestro_options": {"not_a_field": 1}}, "isn't a maestro.SimulatorConfig field"),
        ({"bogus": 1}, "Unknown Maestro option"),
    ],
)
def test_invalid_options_are_rejected(options, match):
    with pytest.raises(ValueError, match=match):
        MaestroBackend(**options)


def test_maestro_options_pass_through():
    backend = MaestroBackend(method="mps", maestro_options={"mps_sampling": "probabilities"})
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure_all()
    assert set(backend.run(circuit, shots=100).result().get_counts()) <= {"00", "11"}


def test_set_options_and_run_options():
    backend = MaestroBackend()
    backend.set_options(shots=7, seed_simulator=5)
    circuit = QuantumCircuit(1, 1)
    circuit.h(0)
    circuit.measure(0, 0)
    result = backend.run(circuit).result()
    assert sum(result.get_counts().values()) == 7
    assert sum(backend.run(circuit, shots=11).result().get_counts().values()) == 11
    assert backend.options.shots == 7  # A run option doesn't stick.


def test_unknown_run_option_warns():
    circuit = QuantumCircuit(1, 1)
    circuit.measure(0, 0)
    with pytest.warns(UserWarning, match="isn't used"):
        MaestroBackend().run(circuit, shots=1, meas_return="single")


@pytest.mark.parametrize("shots", [0, -1, 2.5])
def test_invalid_shots(shots):
    circuit = QuantumCircuit(1, 1)
    circuit.measure(0, 0)
    with pytest.raises(ValueError, match="shots"):
        MaestroBackend().run(circuit, shots=shots)


# ---- Running -----------------------------------------------------------------------


def test_seed_makes_runs_reproducible():
    circuit = QuantumCircuit(6)
    circuit.h(range(6))
    circuit.measure_all()
    run = lambda seed: (
        MaestroBackend(seed_simulator=seed).run(circuit, shots=200, memory=True).result()
    )
    assert run(11).get_memory() == run(11).get_memory()
    assert run(11).get_memory() != run(12).get_memory()


def test_memory_matches_counts():
    circuit = QuantumCircuit(3)
    circuit.h(range(3))
    circuit.measure_all()
    result = MaestroBackend(seed_simulator=2).run(circuit, shots=500, memory=True).result()
    memory = result.get_memory()
    assert len(memory) == 500
    tally: dict[str, int] = {}
    for bits in memory:
        tally[bits] = tally.get(bits, 0) + 1
    assert tally == result.get_counts()


def test_several_circuits_in_one_run():
    circuits = []
    for k in range(4):
        circuit = QuantumCircuit(3, 3, name=f"basis{k}")
        for q in range(3):
            if k >> q & 1:
                circuit.x(q)
        circuit.measure(range(3), range(3))
        circuits.append(circuit)
    result = MaestroBackend().run(circuits, shots=3).result()
    for k, circuit in enumerate(circuits):
        assert result.get_counts(circuit) == {format(k, "03b"): 3}


def test_circuit_without_measurements_has_no_counts():
    circuit = QuantumCircuit(2)
    circuit.h(0)
    result = MaestroBackend().run(circuit, shots=5).result()
    assert result.results[0].data.to_dict() == {}


def test_unmeasured_clbits_read_zero():
    circuit = QuantumCircuit(2, 5)
    circuit.x(1)
    circuit.measure(1, 3)
    assert MaestroBackend().run(circuit, shots=4).result().get_counts() == {"01000": 4}


def test_result_metadata_records_what_ran():
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.t(1)
    circuit.measure_all()
    result = MaestroBackend(method="mps").run(circuit, shots=10).result()
    metadata = result.results[0].metadata
    assert metadata["method"] == "matrix_product_state"
    assert metadata["maestro"]["methods"] == ["matrix_product_state"]
    assert metadata["maestro"]["simulators"] == ["QCSim"]


def test_unbound_parameters_raise():
    from qiskit.circuit import Parameter

    circuit = QuantumCircuit(1, 1)
    circuit.rx(Parameter("a"), 0)
    circuit.measure(0, 0)
    with pytest.raises(ValueError, match="unbound parameters"):
        MaestroBackend().run(circuit)


def test_unsupported_instruction_raises_at_run():
    circuit = QuantumCircuit(1, 1)
    with circuit.while_loop((circuit.clbits[0], 0)):
        circuit.measure(0, 0)
    with pytest.raises(UnsupportedInstructionError, match="while_loop"):
        MaestroBackend().run(circuit)


# ---- Mid-circuit measurement, reset and classical control ----------------------------


def test_mid_circuit_measurement():
    circuit = QuantumCircuit(1, 2)
    circuit.h(0)
    circuit.measure(0, 0)
    circuit.x(0)
    circuit.measure(0, 1)
    counts = MaestroBackend(seed_simulator=1).run(circuit, shots=4000).result().get_counts()
    # The second result is always the flip of the first.
    assert set(counts) == {"01", "10"}
    assert_matches_probabilities(counts, {"01": 0.5, "10": 0.5})


def test_reset_after_entangling():
    circuit = QuantumCircuit(2, 2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.reset(0)
    circuit.measure([0, 1], [0, 1])
    counts = MaestroBackend(seed_simulator=2).run(circuit, shots=4000).result().get_counts()
    assert_matches_probabilities(counts, {"00": 0.5, "10": 0.5})


def _teleportation() -> QuantumCircuit:
    """Teleport RY(1.1)|0> from qubit 0 to qubit 2 with feed-forward corrections."""
    circuit = QuantumCircuit(3, 3)
    circuit.ry(1.1, 0)
    circuit.h(1)
    circuit.cx(1, 2)
    circuit.cx(0, 1)
    circuit.h(0)
    circuit.measure(0, 0)
    circuit.measure(1, 1)
    with circuit.if_test((circuit.clbits[1], 1)):
        circuit.x(2)
    with circuit.if_test((circuit.clbits[0], 1)):
        circuit.z(2)
    circuit.measure(2, 2)
    return circuit


def test_if_test_feed_forward():
    counts = (
        MaestroBackend(seed_simulator=3).run(_teleportation(), shots=20_000).result().get_counts()
    )
    p_one = sum(n for bits, n in counts.items() if bits[0] == "1") / 20_000
    assert abs(p_one - np.sin(0.55) ** 2) < 0.015


def test_if_else_branches():
    circuit = QuantumCircuit(2, 2)
    circuit.h(0)
    circuit.measure(0, 0)
    with circuit.if_test((circuit.clbits[0], 1)) as else_:
        circuit.x(1)
    with else_:
        circuit.h(1)
    circuit.measure(1, 1)
    counts = MaestroBackend(seed_simulator=4).run(circuit, shots=8000).result().get_counts()
    assert_matches_probabilities(counts, {"11": 0.5, "00": 0.25, "10": 0.25})


@requires_aer
def test_classical_control_matches_aer():
    from qiskit_aer import AerSimulator

    circuit = _teleportation()
    mine = MaestroBackend(seed_simulator=5).run(circuit, shots=20_000).result().get_counts()
    aer = AerSimulator(seed_simulator=5).run(circuit, shots=20_000).result().get_counts()
    assert_same_distribution(mine, aer)


# ---- Against Qiskit's references ----------------------------------------------------


def test_random_circuits_match_statevector_sampler():
    for circuit in seeded_random_circuits(5, 4, 6, measure=True, seed=100):
        mine = MaestroBackend(seed_simulator=1).run(circuit, shots=20_000).result().get_counts()
        reference = StatevectorSampler(seed=1).run([circuit], shots=20_000).result()[0].data.c
        assert_same_distribution(mine, reference.get_counts())


@requires_aer
def test_random_circuits_match_aer():
    from qiskit_aer import AerSimulator

    for circuit in seeded_random_circuits(5, 4, 6, measure=True, seed=200):
        mine = MaestroBackend(seed_simulator=1).run(circuit, shots=20_000).result().get_counts()
        aer_backend = AerSimulator(seed_simulator=1)
        aer = aer_backend.run(transpile(circuit, aer_backend), shots=20_000).result().get_counts()
        assert_same_distribution(mine, aer)


def test_backend_sampler_v2_matches_native_sampler():
    circuit = seeded_random_circuits(1, 3, 5, measure=True, seed=300)[0]
    backend = MaestroBackend(seed_simulator=9)
    via_backend = BackendSamplerV2(backend=backend).run([circuit], shots=20_000).result()[0].data.c
    native = MaestroSampler(seed=9).run([circuit], shots=20_000).result()[0].data.c
    assert via_backend.num_shots == native.num_shots == 20_000
    assert_same_distribution(via_backend.get_counts(), native.get_counts())
