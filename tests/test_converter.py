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
from qiskit.circuit.classical import expr, types
from qiskit.primitives.containers.bindings_array import BindingsArray
from qiskit.quantum_info import SparsePauliOp, Statevector, random_unitary

from qiskit_maestro import MaestroSampler, UnsupportedInstructionError
from qiskit_maestro.converter import NATIVE_GATES, convert

from .conftest import qft_gate, requires_aer

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


# ---- Classical conditions: casts and out-of-range values -----------------------------


def _register_circuit(value: int, condition) -> QuantumCircuit:
    """Set a two-bit register to ``value``, run ``if_test(condition(register))``, read a flag.

    The flag is 1 exactly when the condition held, because the qubit starts at 0 and the
    branch flips it.
    """
    qubits = QuantumRegister(1)
    register = ClassicalRegister(2, "reg")
    flag = ClassicalRegister(1, "flag")
    circuit = QuantumCircuit(qubits, register, flag)
    for bit in range(2):
        if value >> bit & 1:
            circuit.x(0)
            circuit.measure(0, register[bit])
            circuit.x(0)
    with circuit.if_test(condition(register)):
        circuit.x(0)
    circuit.measure(0, flag[0])
    return circuit


def _maestro_flag(circuit: QuantumCircuit) -> set[str]:
    data = MaestroSampler(seed=1).run([circuit], shots=50).result()[0].data
    return set(data.flag.get_counts())


def _aer_flag(circuit: QuantumCircuit) -> set[str]:
    from qiskit_aer import AerSimulator

    counts = AerSimulator(seed_simulator=1).run(circuit, shots=50).result().get_counts()
    return {key.split()[0] for key in counts}


_uint = types.Uint


_CONDITIONS = {
    "narrowed==1": lambda r: expr.equal(expr.cast(r, _uint(1)), 1),
    "narrowed==0": lambda r: expr.equal(expr.cast(r, _uint(1)), 0),
    "1==narrowed": lambda r: expr.equal(1, expr.cast(r, _uint(1))),
    "widened==3": lambda r: expr.equal(expr.cast(r, _uint(3)), 3),
    "widened==2": lambda r: expr.equal(expr.cast(r, _uint(3)), 2),
    "uncast==3": lambda r: expr.equal(r, 3),
}

# Aer 0.18.2 evaluates a cast to Bool upside down (a register holding 3 fails
# cast(register, Bool)), so these are checked against Qiskit's semantics instead.
_BOOL_CAST_CONDITIONS = {
    "narrowed bit": lambda r: expr.cast(expr.cast(r, _uint(1)), types.Bool()),
    "not narrowed bit": lambda r: expr.logic_not(expr.cast(expr.cast(r, _uint(1)), types.Bool())),
}


@requires_aer
@pytest.mark.parametrize("value", [0, 1, 2, 3])
@pytest.mark.parametrize("name", list(_CONDITIONS))
def test_cast_conditions_match_aer(name, value):
    circuit = _register_circuit(value, _CONDITIONS[name])
    assert _maestro_flag(circuit) == _aer_flag(circuit)


@pytest.mark.parametrize(
    "name,value,expected",
    [
        # A register holding 3 has low bit 1, and one holding 2 has low bit 0.
        ("narrowed==1", 3, {"1"}),
        ("narrowed==1", 2, {"0"}),
        ("narrowed==0", 2, {"1"}),
        ("widened==2", 2, {"1"}),
    ],
)
def test_narrowing_cast_keeps_the_low_bits(name, value, expected):
    # The expected flags come from Qiskit's semantics, so this runs without Aer.
    assert _maestro_flag(_register_circuit(value, _CONDITIONS[name])) == expected


@pytest.mark.parametrize(
    "name,value,expected",
    [
        ("narrowed bit", 3, {"1"}),
        ("narrowed bit", 2, {"0"}),
        ("narrowed bit", 1, {"1"}),
        ("not narrowed bit", 3, {"0"}),
        ("not narrowed bit", 2, {"1"}),
    ],
)
def test_bool_cast_of_a_narrowed_register(name, value, expected):
    assert _maestro_flag(_register_circuit(value, _BOOL_CAST_CONDITIONS[name])) == expected


def test_narrowing_cast_converts_to_a_test_of_the_low_bits():
    circuit = _register_circuit(3, _CONDITIONS["narrowed==1"])
    qasm = convert(circuit).to_qasm(np.zeros(0))
    assert "if (c[0]) {" in qasm
    assert "&&" not in qasm


@pytest.mark.parametrize(
    "condition",
    [
        lambda r: expr.cast(r, types.Bool()),
        lambda r: expr.logic_not(r),
        lambda r: expr.equal(expr.cast(r, types.Bool()), True),
    ],
    ids=["register to bool", "logic_not register", "bool(register)==True"],
)
def test_casting_a_register_to_bool_is_rejected(condition):
    # A non-zero test can't be a conjunction of bits. It must not turn into reg == 1.
    circuit = _register_circuit(3, condition)
    with pytest.raises(UnsupportedInstructionError, match="if_else"):
        convert(circuit)


@requires_aer
def test_clbit_compared_with_a_value_above_one_is_never_true():
    from qiskit_aer import AerSimulator

    circuit = QuantumCircuit(1, 2)
    circuit.x(0)
    circuit.measure(0, 0)
    with circuit.if_test((circuit.clbits[0], 2)):
        circuit.x(0)  # Would reset the qubit to 0 if the condition held.
    circuit.measure(0, 1)

    assert not convert(circuit).has_control_flow
    counts = MaestroSampler(seed=1).run([circuit], shots=50).result()[0].data.c.get_counts()
    assert counts == {"11": 50}
    assert AerSimulator(seed_simulator=1).run(circuit, shots=50).result().get_counts() == counts


def _clbit_condition_circuit(value: int, measured: int) -> QuantumCircuit:
    """``if_test((clbit, value))`` with an else branch, then read the qubit."""
    circuit = QuantumCircuit(1, 2)
    if measured:
        circuit.x(0)
        circuit.measure(0, 0)
        circuit.x(0)
    with circuit.if_test((circuit.clbits[0], value)) as else_:
        circuit.x(0)  # The qubit reads 1 if the condition held.
    with else_:
        circuit.z(0)  # A no-op on |0>, so the qubit reads 0 otherwise.
    circuit.measure(0, 1)
    return circuit


def _clbit_condition_counts(circuit: QuantumCircuit) -> dict[str, int]:
    return MaestroSampler(seed=1).run([circuit], shots=50).result()[0].data.c.get_counts()


@requires_aer
@pytest.mark.parametrize("value", [0, 1])
@pytest.mark.parametrize("measured", [0, 1])
def test_clbit_tuple_conditions_match_aer(value, measured):
    from qiskit_aer import AerSimulator

    circuit = _clbit_condition_circuit(value, measured)
    aer = AerSimulator(seed_simulator=1).run(circuit, shots=50).result().get_counts()
    assert _clbit_condition_counts(circuit) == aer


@pytest.mark.parametrize("value", [2, 3])
@pytest.mark.parametrize("measured", [0, 1])
def test_clbit_compared_with_a_value_above_one_takes_the_else_branch(value, measured):
    # No clbit equals 2 or 3, so the else branch must run whatever the clbit holds.
    # (Aer's result for these depends on the other clbits, so it isn't the reference.)
    counts = _clbit_condition_counts(_clbit_condition_circuit(value, measured))
    assert counts == {f"0{measured}": 50}


def test_never_true_condition_runs_only_the_else_branch():
    circuit = QuantumCircuit(2, 2)
    with circuit.if_test((circuit.clbits[0], 2)) as else_:
        circuit.x(0)
    with else_:
        circuit.x(1)
    converted = convert(circuit)
    assert not converted.has_control_flow
    assert [(op.kind, op.qubits) for op in converted.ops] == [("x", (1,))]


@pytest.mark.parametrize("value", [4, 7])
def test_register_value_that_does_not_fit_is_never_true(value):
    # Aer treats it the same way: the register can't hold the value, so the test fails.
    circuit = _register_circuit(3, lambda r: (r, value))
    assert _maestro_flag(circuit) == {"0"}
    assert not convert(circuit).has_control_flow
