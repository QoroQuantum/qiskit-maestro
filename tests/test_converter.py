# SPDX-FileCopyrightText: 2026 Qoro Quantum
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import maestro
import numpy as np
import pytest
import qiskit
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit import Instruction, Parameter
from qiskit.circuit import library as lib
from qiskit.primitives.containers.bindings_array import BindingsArray
from qiskit.quantum_info import SparsePauliOp, Statevector, random_unitary

from qiskit_maestro import UnsupportedInstructionError
from qiskit_maestro.converter import NATIVE_GATES, convert

from .conftest import qft_gate

QISKIT_1 = qiskit.__version__.startswith("1.")


def _prepared(num_qubits: int, seed: int = 7) -> QuantumCircuit:
    """A random product state touching every qubit, so Maestro keeps them all."""
    rng = np.random.default_rng(seed)
    circuit = QuantumCircuit(num_qubits)
    for q in range(num_qubits):
        circuit.ry(rng.uniform(0, np.pi), q)
        circuit.rz(rng.uniform(0, 2 * np.pi), q)
    return circuit


def _maestro_state(circuit: QuantumCircuit, *, via_qasm: bool = False) -> np.ndarray:
    converted = convert(circuit)
    row = converted.angle_rows()[0]
    if via_qasm:
        built = maestro.QasmToCirc().parse_and_translate(converted.to_qasm(row))
    else:
        built = converted.to_maestro(row)
    return np.asarray(maestro.get_statevector(built))


def _overlap(a: np.ndarray, b: np.ndarray) -> float:
    return abs(np.vdot(a, b))


def _gate_instance(gate_class):
    angles = {
        lib.PhaseGate: 1,
        lib.RXGate: 1,
        lib.RYGate: 1,
        lib.RZGate: 1,
        lib.UGate: 3,
        lib.CPhaseGate: 1,
        lib.CRXGate: 1,
        lib.CRYGate: 1,
        lib.CRZGate: 1,
        lib.CUGate: 4,
    }.get(gate_class, 0)
    return gate_class(*[0.3 + 0.41 * i for i in range(angles)])


@pytest.mark.parametrize("gate_class", list(NATIVE_GATES), ids=lambda c: c.__name__)
@pytest.mark.parametrize("via_qasm", [False, True], ids=["direct", "qasm"])
def test_native_gate_matches_qiskit_exactly(gate_class, via_qasm):
    gate = _gate_instance(gate_class)
    # Put the gate on non-adjacent, reversed qubits to catch argument-order slips.
    qubits = [3, 0, 2][: gate.num_qubits]
    circuit = _prepared(4)
    circuit.append(gate, qubits)
    expected = Statevector(circuit).data
    # Exact, global phase included.
    np.testing.assert_allclose(_maestro_state(circuit, via_qasm=via_qasm), expected, atol=1e-12)


def _decomposed_cases():
    theta = 0.37
    cases = {
        "rzz": (lib.RZZGate(theta), 2),
        "rxx": (lib.RXXGate(theta), 2),
        "ryy": (lib.RYYGate(theta), 2),
        "rzx": (lib.RZXGate(theta), 2),
        "ecr": (lib.ECRGate(), 2),
        "iswap": (lib.iSwapGate(), 2),
        "dcx": (lib.DCXGate(), 2),
        "xx_plus_yy": (lib.XXPlusYYGate(theta, 0.2), 2),
        "u1": (lib.U1Gate(theta), 1),
        "u2": (lib.U2Gate(theta, 0.5), 1),
        "u3": (lib.U3Gate(theta, 0.5, 0.9), 1),
        "r": (lib.RGate(theta, 0.4), 1),
        "cu1": (lib.CU1Gate(theta), 2),
        "cu3": (lib.CU3Gate(theta, 0.2, 0.1), 2),
        "csdg": (lib.CSdgGate(), 2),
        "open_cx": (lib.CXGate(ctrl_state=0), 2),
        "open_crz": (lib.CRZGate(theta, ctrl_state=0), 2),
        "csxdg": (lib.SXdgGate().control(1), 2),
        "mcx": (lib.MCXGate(3), 4),
        "rccx": (lib.RCCXGate(), 3),
        "unitary": (lib.UnitaryGate(random_unitary(4, seed=3)), 2),
        "qft": (qft_gate(3), 3),
        "pauli_evolution": (lib.PauliEvolutionGate(SparsePauliOp(["XZ", "YY"]), time=0.4), 2),
        "state_preparation": (lib.StatePreparation([0, 0.6, 0, 0.8]), 2),
    }
    return [pytest.param(gate, n, id=name) for name, (gate, n) in cases.items()]


@pytest.mark.parametrize("gate,num_qubits", _decomposed_cases())
def test_non_native_gate_is_expanded_through_its_definition(gate, num_qubits):
    circuit = _prepared(4)
    circuit.append(gate, [3, 1, 0, 2][:num_qubits])
    # Definitions may differ in global phase, which Maestro doesn't track.
    assert _overlap(_maestro_state(circuit), Statevector(circuit).data) == pytest.approx(
        1, abs=1e-10
    )


def test_skipped_instructions():
    circuit = _prepared(2)
    circuit.barrier()
    circuit.delay(100, 0)
    circuit.id(1)
    circuit.append(lib.GlobalPhaseGate(0.3), [])
    circuit.cx(0, 1)
    assert _overlap(_maestro_state(circuit), Statevector(circuit).data) == pytest.approx(1)


def test_parameter_expressions_are_evaluated_per_row():
    a, b = Parameter("a"), Parameter("b")
    circuit = _prepared(2)
    circuit.rx(2 * a + b, 0)
    circuit.crz(a * b - 1, 0, 1)
    circuit.u(a, b, a - b, 1)
    converted = convert(circuit)
    values = np.array([[0.1, 0.2], [1.3, -0.4], [2.0, 0.5]])
    rows = converted.angle_rows(BindingsArray.coerce({(a, b): values}))
    assert rows.shape[0] == 3
    for row, (va, vb) in zip(rows, values):
        bound = circuit.assign_parameters({a: va, b: vb})
        state = np.asarray(maestro.get_statevector(converted.to_maestro(row)))
        np.testing.assert_allclose(state, Statevector(bound).data, atol=1e-12)


def test_shared_expressions_are_evaluated_once():
    a = Parameter("a")
    circuit = QuantumCircuit(2)
    circuit.rx(a, 0)
    circuit.rx(a, 1)
    circuit.ry(2 * a, 0)
    circuit.ry(2 * a, 1)
    assert len(convert(circuit).expressions) == 2


def test_unbound_parameters_are_reported():
    circuit = QuantumCircuit(1)
    circuit.rx(Parameter("theta"), 0)
    with pytest.raises(ValueError, match="unbound parameters: theta"):
        convert(circuit).angle_rows()


def test_bound_parameter_expressions_become_constants():
    a = Parameter("a")
    circuit = QuantumCircuit(1)
    circuit.rx(a, 0)
    bound = circuit.assign_parameters({a: 0.25})
    converted = convert(bound)
    assert converted.expressions == []
    assert converted.ops[0].angles == (0.25,)


# ---- Measurement, reset and classical control ------------------------------------


def test_measure_and_reset_are_recorded():
    circuit = QuantumCircuit(2, 2)
    circuit.h(0)
    circuit.measure(0, 1)
    circuit.reset(0)
    converted = convert(circuit)
    assert converted.has_measurements
    assert [op.kind for op in converted.ops] == ["h", "measure", "reset"]
    assert converted.ops[1].clbits == (1,)


def test_if_test_on_a_clbit_with_else_becomes_qasm():
    circuit = QuantumCircuit(2, 2)
    circuit.h(0)
    circuit.measure(0, 0)
    with circuit.if_test((circuit.clbits[0], 1)) as else_:
        circuit.x(1)
    with else_:
        circuit.h(1)
    converted = convert(circuit)
    assert converted.has_control_flow
    qasm = converted.to_qasm(converted.angle_rows()[0])
    assert "if (c[0]) { x q[1]; } else { h q[1]; }" in qasm


def test_if_test_on_a_register_becomes_a_bit_test_chain():
    qr = QuantumRegister(2)
    cr = ClassicalRegister(3, "flags")
    circuit = QuantumCircuit(qr, cr)
    circuit.measure(0, 0)
    with circuit.if_test((cr, 0b101)):
        circuit.x(1)
    converted = convert(circuit)
    assert "if (c[0] && !c[1] && c[2]) { x q[1]; }" in converted.to_qasm(converted.angle_rows()[0])


def test_if_test_on_a_negated_clbit_expression():
    from qiskit.circuit.classical import expr

    circuit = QuantumCircuit(2, 1)
    circuit.measure(0, 0)
    with circuit.if_test(expr.logic_not(circuit.clbits[0])):
        circuit.x(1)
    converted = convert(circuit)
    assert "if (!c[0]) { x q[1]; }" in converted.to_qasm(converted.angle_rows()[0])


def test_measurement_inside_if_test_is_allowed():
    circuit = QuantumCircuit(2, 2)
    circuit.measure(0, 0)
    with circuit.if_test((circuit.clbits[0], 1)):
        circuit.measure(1, 1)
    converted = convert(circuit)
    assert "if (c[0]) { c[1] = measure q[1]; }" in converted.to_qasm(converted.angle_rows()[0])


def test_box_is_inlined():
    if not hasattr(QuantumCircuit, "box"):
        pytest.skip("BoxOp needs Qiskit 2.0")
    circuit = _prepared(2)
    with circuit.box():
        circuit.cx(0, 1)
    # Statevector can't evaluate a box, so compare with the same gates unboxed.
    reference = _prepared(2)
    reference.cx(0, 1)
    assert _overlap(_maestro_state(circuit), Statevector(reference).data) == pytest.approx(1)


@pytest.mark.skipif(not QISKIT_1, reason="c_if was removed in Qiskit 2.0")
def test_qiskit_1_c_if_becomes_a_branch():
    import warnings

    circuit = QuantumCircuit(2, 1)
    circuit.measure(0, 0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        circuit.x(1).c_if(circuit.clbits[0], 1)
    converted = convert(circuit)
    assert converted.has_control_flow
    assert "if (c[0]) { x q[1]; }" in converted.to_qasm(converted.angle_rows()[0])


def _loop_circuit():
    circuit = QuantumCircuit(1, 1)
    with circuit.for_loop(range(2)):
        circuit.x(0)
    return circuit


def _while_circuit():
    circuit = QuantumCircuit(1, 1)
    with circuit.while_loop((circuit.clbits[0], 0)):
        circuit.measure(0, 0)
    return circuit


def _switch_circuit():
    circuit = QuantumCircuit(1, 1)
    circuit.measure(0, 0)
    with circuit.switch(circuit.clbits[0]) as case, case(0):
        circuit.x(0)
    return circuit


def _nested_if_circuit():
    circuit = QuantumCircuit(2, 2)
    circuit.measure([0, 1], [0, 1])
    with circuit.if_test((circuit.clbits[0], 1)), circuit.if_test((circuit.clbits[1], 1)):
        circuit.x(0)
    return circuit


def _reset_in_if_circuit():
    circuit = QuantumCircuit(1, 1)
    circuit.measure(0, 0)
    with circuit.if_test((circuit.clbits[0], 1)):
        circuit.reset(0)
    return circuit


def _register_else_circuit():
    cr = ClassicalRegister(2, "c")
    circuit = QuantumCircuit(QuantumRegister(2), cr)
    circuit.measure([0, 1], [0, 1])
    with circuit.if_test((cr, 3)) as else_:
        circuit.x(0)
    with else_:
        circuit.h(0)
    return circuit


def _opaque_circuit():
    circuit = QuantumCircuit(1)
    circuit.append(Instruction("mystery", 1, 0, []), [0])
    return circuit


@pytest.mark.parametrize(
    "build,instruction,reason",
    [
        (_loop_circuit, "for_loop", "loops"),
        (_while_circuit, "while_loop", "loops"),
        (_switch_circuit, "switch_case", "switch"),
        (_nested_if_circuit, "if_else", "nested"),
        (_reset_in_if_circuit, "reset", "conditional reset"),
        (_register_else_circuit, "if_else", "else branch only"),
        (_opaque_circuit, "mystery", "no Qiskit definition"),
    ],
)
def test_unsupported_instructions_are_named(build, instruction, reason):
    with pytest.raises(UnsupportedInstructionError, match=reason) as error:
        convert(build())
    assert error.value.instruction == instruction
    assert f"'{instruction}'" in str(error.value)


def test_estimator_circuits_cannot_measure():
    circuit = QuantumCircuit(1, 1)
    circuit.h(0)
    circuit.measure(0, 0)
    with pytest.raises(UnsupportedInstructionError, match="remove_final_measurements"):
        convert(circuit, estimator=True)


def test_estimator_circuits_may_reset():
    circuit = QuantumCircuit(1)
    circuit.reset(0)
    circuit.h(0)
    assert [op.kind for op in convert(circuit, estimator=True).ops] == ["reset", "h"]
