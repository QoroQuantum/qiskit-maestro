# SPDX-FileCopyrightText: 2026 Qoro Quantum
# SPDX-License-Identifier: GPL-3.0-or-later

"""Convert Qiskit circuits into Maestro circuits.

A circuit is converted once into a list of operations whose angles may still be
Qiskit parameter expressions. Binding a parameter set then only evaluates those
expressions and replays the list onto a fresh :class:`maestro.circuits.QuantumCircuit`,
so a PUB with many parameter sets pays for the conversion once.

Gates outside Maestro's native set are expanded through their Qiskit
definitions. Circuits with ``if_test`` are written out as OpenQASM 3 instead,
because Maestro accepts classical control only through its QASM parser.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import maestro
import numpy as np
from maestro.circuits import QuantumCircuit as MaestroCircuit
from qiskit.circuit import (
    ClassicalRegister,
    Clbit,
    ControlFlowOp,
    ControlledGate,
    Parameter,
    ParameterExpression,
    QuantumCircuit,
)
from qiskit.circuit import library as lib
from qiskit.circuit.classical import expr

from .exceptions import UnsupportedInstructionError

__all__ = ["NATIVE_GATES", "ConvertedCircuit", "convert"]

# Qiskit gate class -> the maestro.circuits.QuantumCircuit method that applies it.
# Every pair was checked against Qiskit's matrix, global phase included. The
# methods take qubits first, then angles in Qiskit's order.
NATIVE_GATES = {
    lib.XGate: "x",
    lib.YGate: "y",
    lib.ZGate: "z",
    lib.HGate: "h",
    lib.SGate: "s",
    lib.SdgGate: "sdg",
    lib.TGate: "t",
    lib.TdgGate: "tdg",
    lib.SXGate: "sx",
    lib.SXdgGate: "sxdg",
    lib.PhaseGate: "p",
    lib.RXGate: "rx",
    lib.RYGate: "ry",
    lib.RZGate: "rz",
    lib.UGate: "u",
    lib.CXGate: "cx",
    lib.CYGate: "cy",
    lib.CZGate: "cz",
    lib.CHGate: "ch",
    lib.CSXGate: "csx",
    lib.SwapGate: "swap",
    lib.CPhaseGate: "cp",
    lib.CRXGate: "crx",
    lib.CRYGate: "cry",
    lib.CRZGate: "crz",
    lib.CUGate: "cu",
    lib.CCXGate: "ccx",
    lib.CSwapGate: "cswap",
}

# Instructions that don't change the state of a noiseless simulation. Delays
# are dropped rather than passed on because Maestro takes them in seconds and a
# simulator has no `dt` to convert Qiskit's default unit.
_IDENTITIES = frozenset({"barrier", "delay", "id", "global_phase"})

_UNSUPPORTED_REASONS = {
    "for_loop": "loops aren't supported; Maestro runs classical control only as if/else.",
    "while_loop": "loops aren't supported; Maestro runs classical control only as if/else.",
    "switch_case": "switch statements aren't supported; use if_test.",
    "break_loop": "loops aren't supported.",
    "continue_loop": "loops aren't supported.",
    "store": "classical variables aren't supported.",
}


@dataclass(frozen=True)
class _Slot:
    """An angle taken from column ``index`` of the bound-angle row."""

    index: int


_Angle = float | _Slot
_Condition = tuple[tuple[int, bool], ...]


@dataclass(frozen=True)
class _Op:
    kind: str  # A NATIVE_GATES method name, "measure", "reset" or "if".
    qubits: tuple[int, ...] = ()
    angles: tuple[_Angle, ...] = ()
    clbits: tuple[int, ...] = ()
    condition: _Condition = ()
    true_body: tuple[_Op, ...] = ()
    false_body: tuple[_Op, ...] | None = None


@dataclass
class ConvertedCircuit:
    """A Qiskit circuit in a form that can be replayed onto Maestro."""

    num_qubits: int
    num_clbits: int
    parameters: tuple[Parameter, ...]
    ops: list[_Op] = field(default_factory=list)
    expressions: list[ParameterExpression] = field(default_factory=list)
    has_measurements: bool = False
    has_control_flow: bool = False

    def angle_rows(self, parameter_values=None) -> np.ndarray:
        """Evaluate every symbolic angle for each parameter set.

        Args:
            parameter_values: A :class:`~qiskit.primitives.containers.BindingsArray`,
                or ``None`` for a circuit without parameters.

        Returns:
            Array of shape ``(num_sets, len(self.expressions))``, one row per
            parameter set in C order of the bindings' shape.
        """
        if parameter_values is None:
            if self.parameters:
                names = ", ".join(sorted(p.name for p in self.parameters))
                raise ValueError(f"The circuit has unbound parameters: {names}.")
            return np.zeros((1, 0))
        num_sets = int(np.prod(parameter_values.shape, dtype=int))
        if not self.expressions:
            return np.zeros((num_sets, 0))
        values = np.asarray(parameter_values.as_array(self.parameters), dtype=float)
        values = values.reshape(num_sets, len(self.parameters))
        column = {param: i for i, param in enumerate(self.parameters)}
        rows = np.empty((num_sets, len(self.expressions)))
        for j, expression in enumerate(self.expressions):
            if isinstance(expression, Parameter):
                rows[:, j] = values[:, column[expression]]
                continue
            free = list(expression.parameters)
            for i in range(num_sets):
                bound = expression.bind({p: values[i, column[p]] for p in free})
                rows[i, j] = _real(bound, expression)
        return rows

    def to_maestro(self, angles: np.ndarray) -> MaestroCircuit:
        """Build the Maestro circuit for one row of :meth:`angle_rows`."""
        if self.has_control_flow:
            raise RuntimeError("Circuits with classical control run through to_qasm().")
        return build_maestro(self.ops, angles)

    def to_qasm(self, angles: np.ndarray) -> str:
        """Write one row of :meth:`angle_rows` as OpenQASM 3 for Maestro's parser."""
        lines = [
            "OPENQASM 3.0;",
            'include "stdgates.inc";',
            f"qubit[{max(self.num_qubits, 1)}] q;",
        ]
        if self.num_clbits:
            lines.append(f"bit[{self.num_clbits}] c;")
        lines.extend(_qasm(op, angles) for op in self.ops)
        return "\n".join(lines) + "\n"

    def to_maestro_qasm(self, angles: np.ndarray) -> MaestroCircuit:
        """Parse :meth:`to_qasm` output, naming the construct Maestro rejected."""
        parser = maestro.QasmToCirc()
        try:
            return parser.parse_and_translate(self.to_qasm(angles))
        except ValueError:
            raise UnsupportedInstructionError("if_else", parser.get_error_message()) from None


def build_maestro(ops: list[_Op], angles: np.ndarray | None = None) -> MaestroCircuit:
    """Replay ops without classical control onto a new Maestro circuit.

    ``angles`` is a row of :meth:`ConvertedCircuit.angle_rows`; ops with no
    symbolic angles need none.
    """
    if angles is None:
        angles = np.zeros(0)
    circuit = MaestroCircuit()
    for op in ops:
        if op.kind == "measure":
            circuit.measure([(op.qubits[0], op.clbits[0])])
        elif op.kind == "reset":
            circuit.reset(op.qubits[0])
        else:
            getattr(circuit, op.kind)(*op.qubits, *_angles(op, angles))
    return circuit


def _real(value, expression) -> float:
    number = complex(value)
    if abs(number.imag) > 1e-12:
        raise ValueError(f"Gate angle {expression} evaluated to the complex number {number}.")
    return number.real


def _angles(op: _Op, row: np.ndarray) -> list[float]:
    return [float(row[a.index]) if isinstance(a, _Slot) else a for a in op.angles]


def _qasm(op: _Op, row: np.ndarray) -> str:
    if op.kind == "measure":
        return f"c[{op.clbits[0]}] = measure q[{op.qubits[0]}];"
    if op.kind == "reset":
        return f"reset q[{op.qubits[0]}];"
    if op.kind == "if":
        test = " && ".join(f"{'' if value else '!'}c[{bit}]" for bit, value in op.condition)
        text = f"if ({test}) {{ {' '.join(_qasm(o, row) for o in op.true_body)} }}"
        if op.false_body is not None:
            text += f" else {{ {' '.join(_qasm(o, row) for o in op.false_body)} }}"
        return text
    qubits = ", ".join(f"q[{q}]" for q in op.qubits)
    if not op.angles:
        return f"{op.kind} {qubits};"
    angles = ", ".join(repr(a) for a in _angles(op, row))
    return f"{op.kind}({angles}) {qubits};"


class _Converter:
    def __init__(self, circuit: QuantumCircuit, *, estimator: bool):
        self.circuit = circuit
        self.estimator = estimator
        self.result = ConvertedCircuit(
            num_qubits=circuit.num_qubits,
            num_clbits=circuit.num_clbits,
            parameters=tuple(circuit.parameters),
        )
        self._slots: dict[ParameterExpression, int] = {}

    def run(self) -> ConvertedCircuit:
        qubits = [self.circuit.find_bit(q).index for q in self.circuit.qubits]
        clbits = [self.circuit.find_bit(c).index for c in self.circuit.clbits]
        self._circuit(self.circuit, qubits, clbits, self.result.ops, in_branch=False)
        return self.result

    def _angle(self, value, name: str) -> _Angle:
        if isinstance(value, ParameterExpression):
            if not value.parameters:
                return _real(value, value)
            if value not in self._slots:
                self._slots[value] = len(self.result.expressions)
                self.result.expressions.append(value)
            return _Slot(self._slots[value])
        try:
            return float(value)
        except (TypeError, ValueError):
            raise UnsupportedInstructionError(
                name, f"its parameter {value!r} isn't a real angle."
            ) from None

    def _circuit(self, circuit, qubits, clbits, out: list[_Op], *, in_branch: bool) -> None:
        def resolve(bit: Clbit) -> int:
            # Control-flow blocks share the outer circuit's bits, so a condition
            # inside one resolves through the block's mapping.
            return clbits[circuit.find_bit(bit).index]

        for instruction in circuit.data:
            self._instruction(
                instruction.operation,
                [qubits[circuit.find_bit(q).index] for q in instruction.qubits],
                [clbits[circuit.find_bit(c).index] for c in instruction.clbits],
                out,
                resolve,
                in_branch=in_branch,
            )

    def _instruction(self, op, qubits, clbits, out: list[_Op], resolve, *, in_branch: bool) -> None:
        name = op.name
        if name in _IDENTITIES:
            return
        # Qiskit 1.x stores c_if conditions on the instruction itself.
        legacy_condition = (
            None if isinstance(op, ControlFlowOp) else getattr(op, "_condition", None)
        )
        if legacy_condition is not None:
            op = op.to_mutable()
            op._condition = None
            self._branch(
                legacy_condition, [(op, qubits, clbits)], None, out, resolve, in_branch=in_branch
            )
            return
        method = NATIVE_GATES.get(getattr(op, "base_class", type(op)))
        if method is not None and not (
            isinstance(op, ControlledGate) and op.ctrl_state != 2**op.num_ctrl_qubits - 1
        ):
            angles = tuple(self._angle(p, name) for p in op.params)
            out.append(_Op(method, tuple(qubits), angles))
            return
        if name == "measure":
            if self.estimator:
                raise UnsupportedInstructionError(
                    "measure",
                    "the Estimator computes expectation values of the state, so its circuits "
                    "can't measure. Remove the measurements, for example with "
                    "circuit.remove_final_measurements().",
                )
            self.result.has_measurements = True
            out.append(_Op("measure", (qubits[0],), clbits=(clbits[0],)))
            return
        if name == "reset":
            if in_branch:
                raise UnsupportedInstructionError(
                    "reset", "Maestro has no conditional reset, so a reset can't go inside if_test."
                )
            out.append(_Op("reset", (qubits[0],)))
            return
        if isinstance(op, ControlFlowOp):
            self._control_flow(op, qubits, clbits, out, resolve, in_branch=in_branch)
            return
        if name in _UNSUPPORTED_REASONS:
            raise UnsupportedInstructionError(name, _UNSUPPORTED_REASONS[name])
        definition = op.definition
        if definition is None:
            raise UnsupportedInstructionError(
                name,
                "it isn't one of Maestro's native operations and has no Qiskit definition to expand.",
            )
        self._circuit(definition, qubits, clbits, out, in_branch=in_branch)

    def _control_flow(
        self, op, qubits, clbits, out: list[_Op], resolve, *, in_branch: bool
    ) -> None:
        if op.name == "box":
            self._circuit(op.blocks[0], qubits, clbits, out, in_branch=in_branch)
            return
        if op.name != "if_else":
            raise UnsupportedInstructionError(
                op.name,
                _UNSUPPORTED_REASONS.get(op.name, "only if_test control flow is supported."),
            )
        blocks = [(block, qubits, clbits) for block in op.blocks]
        self._branch(
            op.condition,
            blocks[:1],
            blocks[1] if len(blocks) > 1 else None,
            out,
            resolve,
            in_branch=in_branch,
        )

    def _branch(
        self, condition, true_body, false_block, out: list[_Op], resolve, *, in_branch: bool
    ) -> None:
        """Append an ``if`` op.

        ``true_body`` is a list of ``(item, qubits, clbits)`` and ``false_block`` is
        one such tuple or ``None``. An item is an instruction or a block circuit;
        ``qubits`` and ``clbits`` give the circuit index of each of its bits.
        """
        if self.estimator:
            raise UnsupportedInstructionError(
                "if_else",
                "the Estimator's circuits can't measure, so they can't branch on results.",
            )
        if in_branch:
            raise UnsupportedInstructionError(
                "if_else", "Maestro doesn't support nested if_test blocks."
            )
        bits = self._condition(condition, resolve)
        if false_block is not None and len(bits) > 1:
            raise UnsupportedInstructionError(
                "if_else",
                "Maestro supports an else branch only when the condition is a single clbit, "
                "not a register or a combination of bits.",
            )

        def body(items) -> tuple[_Op, ...]:
            ops: list[_Op] = []
            for item, item_qubits, item_clbits in items:
                if isinstance(item, QuantumCircuit):
                    self._circuit(item, item_qubits, item_clbits, ops, in_branch=True)
                else:
                    self._instruction(item, item_qubits, item_clbits, ops, resolve, in_branch=True)
            return tuple(ops)

        self.result.has_control_flow = True
        out.append(
            _Op(
                "if",
                condition=bits,
                true_body=body(true_body),
                false_body=None if false_block is None else body([false_block]),
            )
        )

    def _condition(self, condition, resolve) -> _Condition:
        if isinstance(condition, tuple):
            target, value = condition
        else:
            target, value = self._expr_condition(condition)
        if isinstance(target, Clbit):
            return ((resolve(target), bool(value)),)
        if isinstance(target, ClassicalRegister):
            value = int(value)
            if not 0 <= value < 2 ** len(target):
                raise ValueError(f"Condition value {value} doesn't fit register {target.name!r}.")
            return tuple((resolve(bit), bool(value >> j & 1)) for j, bit in enumerate(target))
        raise UnsupportedInstructionError("if_else", f"unsupported condition {condition!r}.")

    def _expr_condition(self, condition):
        unsupported = UnsupportedInstructionError(
            "if_else",
            "Maestro supports conditions on one clbit or on a register equal to a value, "
            f"not the classical expression {condition}.",
        )
        node = condition
        while isinstance(node, expr.Cast):
            node = node.operand
        if isinstance(node, expr.Var) and isinstance(node.var, Clbit):
            return node.var, True
        if (
            isinstance(node, expr.Unary)
            and node.op in (expr.Unary.Op.LOGIC_NOT, expr.Unary.Op.BIT_NOT)
            and isinstance(node.operand, expr.Var)
            and isinstance(node.operand.var, Clbit)
        ):
            return node.operand.var, False
        if isinstance(node, expr.Binary) and node.op == expr.Binary.Op.EQUAL:
            left, right = node.left, node.right
            while isinstance(left, expr.Cast):
                left = left.operand
            while isinstance(right, expr.Cast):
                right = right.operand
            if isinstance(right, expr.Var):
                left, right = right, left
            if (
                isinstance(left, expr.Var)
                and isinstance(left.var, (Clbit, ClassicalRegister))
                and isinstance(right, expr.Value)
            ):
                return left.var, right.value
        raise unsupported


def convert(circuit: QuantumCircuit, *, estimator: bool = False) -> ConvertedCircuit:
    """Convert a Qiskit circuit for Maestro.

    Args:
        circuit: The circuit, possibly with unbound parameters.
        estimator: Reject measurements and classical control, which an
            Estimator circuit can't have.

    Raises:
        UnsupportedInstructionError: naming the first instruction Maestro can't run.
    """
    return _Converter(circuit, estimator=estimator).run()
