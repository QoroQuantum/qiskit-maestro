# SPDX-FileCopyrightText: 2026 Qoro Quantum
# SPDX-License-Identifier: GPL-3.0-or-later

"""The OpenQASM 2.0 the Qoro cloud is sent: Maestro must read it as Qiskit defines the circuit."""

from __future__ import annotations

import maestro
import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import Statevector

from qiskit_maestro import UnsupportedInstructionError
from qiskit_maestro.converter import convert

from .conftest import seeded_random_circuits


def _maestro_statevector(qasm: str, num_qubits: int) -> np.ndarray:
    circuit = maestro.QasmToCirc().parse_and_translate(qasm)
    vector = np.asarray(circuit.get_statevector(config=maestro.SimulatorConfig()), dtype=complex)
    # Maestro's width is the highest qubit an operation touches, plus one.
    padded = np.zeros(2**num_qubits, dtype=complex)
    padded[: len(vector)] = vector[: 2**num_qubits]
    return padded


def _assert_same_state(circuit: QuantumCircuit, values=None, *, exact: bool = True) -> None:
    """The QASM gives the circuit's state; ``exact=False`` allows a global phase.

    The converter drops global phase unless a circuit saves its statevector, so
    circuits whose gates are expanded through definitions may differ by one.
    """
    converted = convert(circuit)
    rows = converted.angle_rows(None if values is None else _bindings(circuit, values))
    bound = circuit if values is None else circuit.assign_parameters(values)
    mine = _maestro_statevector(converted.to_qasm2(rows[0]), circuit.num_qubits)
    expected = Statevector(bound).data
    if not exact:
        mine = mine * np.exp(-1j * np.angle(np.vdot(expected, mine)))
    np.testing.assert_allclose(mine, expected, atol=1e-9)


def _bindings(circuit, values):
    from qiskit.primitives.containers import BindingsArray

    return BindingsArray({tuple(circuit.parameters): np.asarray(values, dtype=float)})


def test_gates_renamed_or_rewritten_for_openqasm2_keep_their_phases():
    circuit = QuantumCircuit(3)
    circuit.h([0, 1, 2])
    circuit.p(0.4, 0)
    circuit.cp(0.8, 0, 1)
    circuit.csx(1, 2)
    circuit.cu(0.3, 0.5, 0.9, 0.6, 2, 0)
    circuit.u(0.2, 0.7, 1.1, 1)
    circuit.h([0, 1, 2])
    _assert_same_state(circuit)


@pytest.mark.parametrize("seed", range(10))
def test_random_circuits_match_qiskit(seed):
    (circuit,) = seeded_random_circuits(1, 4, 6, measure=False, seed=100 + 7 * seed)
    _assert_same_state(circuit, exact=False)


def test_parameter_values_are_written_in():
    a, b = Parameter("a"), Parameter("b")
    circuit = QuantumCircuit(2)
    circuit.ry(a, 0)
    circuit.cx(0, 1)
    circuit.rz(2 * b - a, 1)
    circuit.h(1)
    _assert_same_state(circuit, [0.3, -1.7])


def test_measure_and_reset():
    circuit = QuantumCircuit(2, 3)
    circuit.x(0)
    circuit.measure(0, 2)
    circuit.reset(0)
    circuit.measure(1, 0)
    qasm = convert(circuit).to_qasm2(np.zeros(0))
    assert qasm.startswith('OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\ncreg c[3];\n')
    assert "measure q[0] -> c[2];" in qasm
    assert "reset q[0];" in qasm


def test_classical_control_raises():
    circuit = QuantumCircuit(2, 1)
    circuit.h(0)
    circuit.measure(0, 0)
    with circuit.if_test((circuit.clbits[0], 1)):
        circuit.x(1)
    converted = convert(circuit)
    with pytest.raises(UnsupportedInstructionError, match="Qoro cloud"):
        converted.to_qasm2(np.zeros(0))
