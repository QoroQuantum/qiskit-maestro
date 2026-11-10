# SPDX-FileCopyrightText: 2026 Qoro Quantum
# SPDX-License-Identifier: GPL-3.0-or-later

"""Convert Qiskit circuits into Maestro circuits.

A circuit is converted once into a list of operations whose angles may still be
Qiskit parameter expressions. Binding a parameter set then only evaluates those
expressions and replays the list onto a fresh :class:`maestro.circuits.QuantumCircuit`,
so a PUB with many parameter sets pays for the conversion once.

Gates outside Maestro's native set are expanded through their Qiskit
definitions. ``for_loop`` is unrolled and ``switch`` becomes a chain of ``if``
blocks. Circuits with classical control are written out as OpenQASM 3,
because Maestro accepts classical control only through its QASM parser.
Circuits sent to the Qoro cloud are written as OpenQASM 2.0, which it takes.

With an Aer noise model, each instruction's errors become placeholder ``noise``
operations where Aer applies them. :meth:`ConvertedCircuit.with_noise` replaces
them with Maestro operations once the simulation method is known.
"""

from __future__ import annotations

import dataclasses
import math
import numbers
from dataclasses import dataclass, field

import maestro
import numpy as np
from maestro.circuits import QuantumCircuit as MaestroCircuit
from qiskit.circuit import (
    CASE_DEFAULT,
    ClassicalRegister,
    Clbit,
    ControlFlowOp,
    ControlledGate,
    Parameter,
    ParameterExpression,
    QuantumCircuit,
)
from qiskit.circuit import library as lib
from qiskit.circuit.classical import expr, types

from .exceptions import UnsupportedInstructionError, UnsupportedNoiseError
from .noise import maestro_noise_model

__all__ = ["NATIVE_GATES", "SAVE_INSTRUCTIONS", "ConvertedCircuit", "NoiseSpec", "convert"]

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
# simulator has no `dt` to convert Qiskit's default unit. A noise model can
# still attach errors to them.
_IDENTITIES = frozenset({"barrier", "delay", "id", "global_phase"})

# Aer's save instructions that MaestroBackend.run returns data for, as Aer does.
SAVE_INSTRUCTIONS = frozenset(
    {
        "save_statevector",
        "save_statevector_dict",
        "save_probabilities",
        "save_probabilities_dict",
        "save_expval",
        "save_expval_var",
    }
)

_UNSUPPORTED_REASONS = {
    "while_loop": "while loops aren't supported; Maestro runs classical control only as if/else.",
    "break_loop": "a loop with break can't be unrolled; only for loops without break or continue are.",
    "continue_loop": "a loop with continue can't be unrolled; only for loops without break or "
    "continue are.",
    "store": "classical variables aren't supported.",
}

# Most if blocks a switch's default case may become.
_MAX_DEFAULT_BRANCHES = 64


@dataclass(frozen=True)
class _Slot:
    """An angle taken from column ``index`` of the bound-angle row."""

    index: int


_Angle = float | _Slot
_Condition = tuple[tuple[int, bool], ...]


@dataclass(frozen=True)
class _Op:
    # A NATIVE_GATES method name, "measure", "reset", "if", "noise" (an Aer error to
    # translate), "pauli_channel" or "delay" (a Maestro idle-noise delay).
    kind: str
    qubits: tuple[int, ...] = ()
    angles: tuple[_Angle, ...] = ()
    clbits: tuple[int, ...] = ()
    condition: _Condition = ()
    true_body: tuple[_Op, ...] = ()
    false_body: tuple[_Op, ...] | None = None
    probabilities: tuple[float, ...] = ()
    noise: int = -1


@dataclass(frozen=True)
class Save:
    """An Aer save instruction: what to save, and where in the circuit."""

    kind: str
    label: str
    qubits: tuple[int, ...]
    # save_expval terms: (Pauli label on ``qubits``, coefficient, coefficient in the square).
    terms: tuple[tuple[str, float, float], ...]
    position: int  # Number of top-level ops before it.
    phases: int  # Number of global-phase terms before it.
    after_reset: bool = False
    after_noise: bool = False


@dataclass(frozen=True)
class NoiseSpec:
    """How a circuit with translated noise runs.

    ``stochastic`` circuits hold operations a pure-state simulation samples, so
    they run one noise realization per shot. ``idle`` maps each qubit with
    damping ``delay`` operations to its excited-state population.
    """

    family: str
    stochastic: bool
    active: bool = False  # Whether the circuit holds any noise operation.
    idle: tuple[tuple[int, float], ...] = ()
    readout: tuple[tuple[int, float, float], ...] = ()

    def maestro_model(self, *, readout: bool = True) -> maestro.NoiseModel:
        return maestro_noise_model(
            dict(self.idle), {q: (a, b) for q, a, b in self.readout} if readout else {}
        )


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
    saves: list[Save] = field(default_factory=list)
    # Global-phase terms of the top-level circuit and the definitions expanded in
    # it, kept only for circuits that save their statevector.
    phases: list[_Angle] = field(default_factory=list)
    noise: NoiseSpec | None = None
    # The qubit pure-state noise uses as an ancilla, or None.
    ancilla: int | None = None
    # After with_noise drops idle qubits: the original index of each qubit left.
    qubit_map: tuple[int, ...] | None = None

    @property
    def active_qubits(self) -> list[int]:
        """The qubits an operation or a save acts on, as Aer counts them for truncation."""
        if any(save.kind.startswith("save_statevector") for save in self.saves):
            return list(range(self.num_qubits))
        used = {q for op in _walk(self.ops) for q in op.qubits}
        used.update(q for save in self.saves for q in save.qubits)
        return sorted(used)

    @property
    def has_noise(self) -> bool:
        """Whether the circuit holds Aer errors not yet translated by :meth:`with_noise`."""
        return any(op.kind == "noise" for op in _walk(self.ops))

    @property
    def can_sample(self) -> bool:
        """Whether every measurement comes at the end, as Aer's automatic method requires.

        Then a density-matrix simulation can sample all shots from one run.
        """
        measured = False
        for op in self.ops:
            if op.kind == "if":
                return False
            if op.kind == "measure":
                measured = True
            elif measured and op.kind != "noise":
                return False
        return True

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

    def phase(self, angles: np.ndarray, count: int) -> float:
        """The global phase of the first ``count`` phase terms, for one row of angles."""
        return sum(
            float(angles[a.index]) if isinstance(a, _Slot) else a for a in self.phases[:count]
        )

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

    def to_qasm2(self, angles: np.ndarray) -> str:
        """Write one row of :meth:`angle_rows` as OpenQASM 2.0, the form the Qoro cloud takes.

        Raises:
            UnsupportedInstructionError: for classical control, which OpenQASM 2
                can express only as a test of a whole register.
        """
        return self._qasm2(lambda slot: repr(float(angles[slot.index])))

    def qasm2_template(self) -> tuple[str, list[int]]:
        """OpenQASM 2.0 with each symbolic angle left as a named parameter.

        The parameter for column ``j`` of :meth:`angle_rows` is ``angle_j``.
        Returns the template and the columns it uses, in the order its
        parameters are listed.

        Raises:
            UnsupportedInstructionError: as :meth:`to_qasm2` does.
        """
        columns = sorted(
            {a.index for op in _walk(self.ops) for a in op.angles if isinstance(a, _Slot)}
        )
        return self._qasm2(lambda slot: f"{QASM2_PARAMETER_PREFIX}{slot.index}"), columns

    def _qasm2(self, slot_text) -> str:
        lines = ["OPENQASM 2.0;", 'include "qelib1.inc";', f"qreg q[{max(self.num_qubits, 1)}];"]
        if self.num_clbits:
            lines.append(f"creg c[{self.num_clbits}];")
        for op in self.ops:
            lines.extend(_qasm2(op, slot_text))
        return "\n".join(lines) + "\n"

    def to_maestro_qasm(self, angles: np.ndarray) -> MaestroCircuit:
        """Parse :meth:`to_qasm` output, naming the construct Maestro rejected."""
        parser = maestro.QasmToCirc()
        try:
            return parser.parse_and_translate(self.to_qasm(angles))
        except ValueError:
            raise UnsupportedInstructionError("if_else", parser.get_error_message()) from None

    def build(self, angles: np.ndarray) -> MaestroCircuit:
        """The Maestro circuit for one row of angles, through QASM if it has classical control."""
        return self.to_maestro_qasm(angles) if self.has_control_flow else self.to_maestro(angles)

    def with_noise(self, translator, method: str) -> ConvertedCircuit:
        """Translate the circuit's Aer errors for ``method`` and attach the readout errors.

        Returns a copy whose ``noise`` says how to run it. A circuit converted
        without a noise model is returned unchanged.

        Raises:
            UnsupportedNoiseError: for an error Maestro can't reproduce exactly
                with this method, or noise on a circuit with classical control.
        """
        if translator is None:
            return self
        family = "exact" if method == "density_matrix" else "sampled"
        ancilla = self.num_qubits
        idle: dict[int, float] = {}
        stochastic = False

        def realize(op: _Op) -> list[_Op]:
            nonlocal stochastic
            if op.kind != "noise":
                return [op]
            name, qubits = translator.describe(op.noise)
            channel = translator.channel(op.noise, family)
            out = []
            if channel.paulis is not None:
                out.append(_Op("pauli_channel", op.qubits, probabilities=channel.paulis))
                stochastic = stochastic or family == "sampled"
            for k, a in channel.decays:
                qubit = op.qubits[k]
                if family == "exact":
                    idle[qubit] = translator.excited_population(qubit)
                    out.append(_Op("delay", (qubit,), angles=(-math.log(a),)))
                    continue
                if method == "stabilizer":
                    raise UnsupportedNoiseError(
                        name,
                        f"its noise on qubits {list(qubits)} includes relaxation, which the "
                        "stabilizer method can't apply. Use method='statevector' or "
                        "'density_matrix'.",
                        qubits,
                    )
                # Amplitude damping, exactly: rotate the excited population into a fresh
                # ancilla, return the decayed part to |0>, and discard the ancilla.
                stochastic = True
                theta = 2 * math.asin(math.sqrt(1 - a))
                out.extend(
                    [
                        _Op("cry", (qubit, ancilla), (theta,)),
                        _Op("cx", (ancilla, qubit)),
                        _Op("reset", (ancilla,)),
                    ]
                )
            return out

        if self.has_noise and self.has_control_flow:
            raise UnsupportedNoiseError(
                "if_else",
                "noise on circuits with classical control isn't supported: Maestro runs "
                "classical control only through OpenQASM, which can't hold noise channels. "
                "Readout errors alone are supported.",
            )
        positions = []
        ops: list[_Op] = []
        for op in self.ops:
            positions.append(len(ops))
            ops.extend(realize(op))
        positions.append(len(ops))
        noisy_before = np.cumsum([0] + [op.kind == "noise" for op in self.ops])
        saves = [
            dataclasses.replace(
                save,
                position=positions[save.position],
                after_noise=bool(noisy_before[save.position]),
            )
            for save in self.saves
        ]
        readout = []
        if self.has_measurements:
            translator.check_readout()
            for qubit in range(self.num_qubits):
                rates = translator.readout(qubit)
                if rates is not None:
                    readout.append((qubit, *rates))
        uses_ancilla = any(ancilla in op.qubits for op in ops)
        noise = NoiseSpec(
            family=family,
            stochastic=stochastic,
            active=uses_ancilla or any(op.kind in ("pauli_channel", "delay") for op in _walk(ops)),
            idle=tuple(sorted(idle.items())),
            readout=tuple(readout),
        )
        # Drop the qubits nothing acts on, as Aer does. A transpiled circuit is as wide
        # as its device, and Maestro would otherwise simulate every qubit of a noisy
        # circuit for an expectation value.
        keep = self.active_qubits
        if len(keep) == self.num_qubits:
            return dataclasses.replace(
                self, ops=ops, saves=saves, noise=noise, ancilla=ancilla if uses_ancilla else None
            )
        index = {q: i for i, q in enumerate(keep)}
        index[ancilla] = len(keep)
        noise = dataclasses.replace(
            noise,
            idle=tuple((index[q], p) for q, p in noise.idle),
            readout=tuple((index[q], a, b) for q, a, b in noise.readout if q in index),
        )
        return dataclasses.replace(
            self,
            num_qubits=len(keep),
            ops=[_remap(op, index) for op in ops],
            saves=[
                dataclasses.replace(save, qubits=tuple(index[q] for q in save.qubits))
                for save in saves
            ],
            noise=noise,
            ancilla=len(keep) if uses_ancilla else None,
            qubit_map=tuple(keep),
        )


def _remap(op: _Op, index: dict[int, int]) -> _Op:
    if op.kind == "if":
        return dataclasses.replace(
            op,
            true_body=tuple(_remap(o, index) for o in op.true_body),
            false_body=None
            if op.false_body is None
            else tuple(_remap(o, index) for o in op.false_body),
        )
    # The constructor, as dataclasses.replace is several times slower on large circuits.
    return _Op(
        kind=op.kind,
        qubits=tuple(index[q] for q in op.qubits),
        angles=op.angles,
        clbits=op.clbits,
        condition=op.condition,
        true_body=op.true_body,
        false_body=op.false_body,
        probabilities=op.probabilities,
        noise=op.noise,
    )


def _walk(ops):
    for op in ops:
        yield op
        if op.kind == "if":
            yield from _walk(op.true_body)
            yield from _walk(op.false_body or ())


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
        elif op.kind == "pauli_channel":
            circuit.pauli_channel(list(op.qubits), list(op.probabilities))
        elif op.kind == "delay":
            circuit.delay(op.qubits[0], float(op.angles[0]))
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
    if op.kind in ("noise", "pauli_channel", "delay"):
        raise RuntimeError("Noise operations can't be written as OpenQASM.")
    qubits = ", ".join(f"q[{q}]" for q in op.qubits)
    if not op.angles:
        return f"{op.kind} {qubits};"
    angles = ", ".join(repr(a) for a in _angles(op, row))
    return f"{op.kind}({angles}) {qubits};"


# Native gates whose OpenQASM 2 (qelib1) names differ from Maestro's method names.
_QASM2_NAMES = {"p": "u1", "cp": "cu1", "u": "u3"}

# The name of the template parameter for column j of the angle rows is this plus j.
QASM2_PARAMETER_PREFIX = "angle_"


def _qasm2(op: _Op, slot_text) -> list[str]:
    """OpenQASM 2.0 statements for one op, using only the gates the Qoro cloud accepts.

    ``slot_text`` writes a symbolic angle: its value, or a template parameter.
    """
    if op.kind == "measure":
        return [f"measure q[{op.qubits[0]}] -> c[{op.clbits[0]}];"]
    if op.kind == "reset":
        return [f"reset q[{op.qubits[0]}];"]
    if op.kind == "if":
        raise UnsupportedInstructionError(
            "if_else",
            "classical control can't be sent to the Qoro cloud yet. It takes OpenQASM 2, "
            "which can only test a whole register. Run the circuit locally instead.",
        )
    if op.kind in ("noise", "pauli_channel", "delay"):
        raise RuntimeError("Noise operations can't be written as OpenQASM.")
    angles = [slot_text(a) if isinstance(a, _Slot) else repr(float(a)) for a in op.angles]
    qubits = [f"q[{q}]" for q in op.qubits]
    if op.kind == "csx":
        # Controlled H S H, and H S H is SX exactly.
        control, target = qubits
        return [f"h {target};", f"cu1({math.pi / 2!r}) {control},{target};", f"h {target};"]
    if op.kind == "cu":
        # CU's fourth angle is a phase on the control's |1>; cu3 is CU without it.
        control, target = qubits
        theta, phi, lam, gamma = angles
        return [f"u1({gamma}) {control};", f"cu3({theta},{phi},{lam}) {control},{target};"]
    name = _QASM2_NAMES.get(op.kind, op.kind)
    arguments = f"({','.join(angles)})" if angles else ""
    return [f"{name}{arguments} {','.join(qubits)};"]


class _Converter:
    def __init__(self, circuit: QuantumCircuit, *, estimator: bool, noise, saves: bool):
        self.circuit = circuit
        self.estimator = estimator
        self.noise = noise
        self.allow_saves = saves
        # Unrolling binds loop variables, which Qiskit 1.x lists among the parameters.
        loop_variables = _loop_variables(circuit) if circuit.parameters else set()
        self.result = ConvertedCircuit(
            num_qubits=circuit.num_qubits,
            num_clbits=circuit.num_clbits,
            parameters=tuple(p for p in circuit.parameters if p not in loop_variables),
        )
        self._slots: dict[ParameterExpression, int] = {}
        self._has_reset = False
        # Global phase matters only for a saved statevector.
        self._track_phase = saves and any(
            instruction.operation.name in ("save_statevector", "save_statevector_dict")
            for instruction in circuit.data
        )

    def run(self) -> ConvertedCircuit:
        qubits = [self.circuit.find_bit(q).index for q in self.circuit.qubits]
        clbits = [self.circuit.find_bit(c).index for c in self.circuit.clbits]
        self._phase(self.circuit.global_phase, in_branch=False)
        self._circuit(
            self.circuit, qubits, clbits, self.result.ops, in_branch=False, top_level=True
        )
        labels = [save.label for save in self.result.saves]
        for label in labels:
            if labels.count(label) > 1:
                raise ValueError(f"Two save instructions use the label {label!r}.")
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

    def _phase(self, value, *, in_branch: bool) -> None:
        if self._track_phase and not in_branch and value != 0:
            self.result.phases.append(self._angle(value, "global_phase"))

    def _circuit(
        self,
        circuit,
        qubits,
        clbits,
        out: list[_Op],
        *,
        in_branch: bool,
        top_level: bool = False,
        noisy: bool = True,
    ) -> None:
        """Convert ``circuit``'s instructions into ``out``.

        ``top_level`` is true only for the circuit itself, where Aer runs its
        custom noise passes. ``noisy`` is false inside the definition of an
        expanded gate: Aer attaches noise to the gate as a whole.
        """

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
                top_level=top_level,
                noisy=noisy,
            )

    def _noise_ops(self, keys: list[int], qubits) -> list[_Op]:
        return [_Op("noise", tuple(qubits), noise=key) for key in keys]

    def _instruction(
        self,
        op,
        qubits,
        clbits,
        out: list[_Op],
        resolve,
        *,
        in_branch: bool,
        top_level: bool = False,
        noisy: bool = True,
    ) -> None:
        name = op.name
        if name in SAVE_INSTRUCTIONS or (
            name.startswith("save_") and type(op).__module__.startswith("qiskit_aer")
        ):
            self._save(op, qubits, out, in_branch=in_branch, top_level=top_level)
            return
        before: list[int] = []
        after: list[int] = []
        if self.noise is not None and noisy and not isinstance(op, ControlFlowOp):
            before, after = self.noise.errors(op, qubits, top_level=top_level)
        if name in _IDENTITIES:
            if name == "global_phase":
                self._phase(op.params[0], in_branch=in_branch)
            out.extend(self._noise_ops(before + after, qubits))
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
            out.extend(self._noise_ops(before, qubits))
            out.append(_Op(method, tuple(qubits), angles))
            out.extend(self._noise_ops(after, qubits))
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
            out.extend(self._noise_ops(before, qubits))
            out.append(_Op("measure", (qubits[0],), clbits=(clbits[0],)))
            out.extend(self._noise_ops(after, qubits))
            return
        if name == "reset":
            if in_branch:
                raise UnsupportedInstructionError(
                    "reset", "Maestro has no conditional reset, so a reset can't go inside if_test."
                )
            self._has_reset = True
            out.extend(self._noise_ops(before, qubits))
            out.append(_Op("reset", (qubits[0],)))
            out.extend(self._noise_ops(after, qubits))
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
        out.extend(self._noise_ops(before, qubits))
        self._phase(definition.global_phase, in_branch=in_branch)
        self._circuit(definition, qubits, clbits, out, in_branch=in_branch, noisy=False)
        out.extend(self._noise_ops(after, qubits))

    def _save(self, op, qubits, out: list[_Op], *, in_branch: bool, top_level: bool) -> None:
        name = op.name
        if not self.allow_saves:
            raise UnsupportedInstructionError(
                name,
                "save instructions return data through MaestroBackend.run only; the Sampler "
                "and Estimator don't take them.",
            )
        if name not in SAVE_INSTRUCTIONS:
            raise UnsupportedInstructionError(
                name,
                f"the supported save instructions are {', '.join(sorted(SAVE_INSTRUCTIONS))}.",
            )
        if in_branch or not top_level:
            raise UnsupportedInstructionError(
                name, "save instructions inside control flow aren't supported."
            )
        subtype = getattr(op, "_subtype", "single")
        if subtype not in ("single", "average"):
            raise UnsupportedInstructionError(
                name,
                "only the default save is supported, not pershot, conditional or "
                "unnormalized ones.",
            )
        if self.result.has_measurements or self.result.has_control_flow:
            raise UnsupportedInstructionError(
                name,
                "a save after a measurement depends on its random outcome. Move the save "
                "before the measurements.",
            )
        terms = ()
        if name in ("save_expval", "save_expval_var"):
            terms = tuple(
                (str(label), float(np.real(coeffs[0])), float(np.real(coeffs[1])))
                for label, coeffs in op.params
            )
        self.result.saves.append(
            Save(
                kind=name,
                label=op.label or name,
                qubits=tuple(qubits),
                terms=terms,
                position=len(out),
                phases=len(self.result.phases),
                after_reset=self._has_reset,
            )
        )

    def _control_flow(
        self, op, qubits, clbits, out: list[_Op], resolve, *, in_branch: bool
    ) -> None:
        if op.name == "box":
            self._phase(op.blocks[0].global_phase, in_branch=in_branch)
            self._circuit(op.blocks[0], qubits, clbits, out, in_branch=in_branch)
            return
        if op.name == "for_loop":
            self._for_loop(op, qubits, clbits, out, in_branch=in_branch)
            return
        if op.name == "switch_case":
            self._switch(op, qubits, clbits, out, resolve, in_branch=in_branch)
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

    def _for_loop(self, op, qubits, clbits, out: list[_Op], *, in_branch: bool) -> None:
        """Unroll a loop: Qiskit fixes its index set when the circuit is built."""
        indexset, loop_parameter, body = op.params
        values = list(indexset)
        for value in values:
            if isinstance(value, bool) or not isinstance(value, numbers.Integral):
                raise UnsupportedInstructionError(
                    "for_loop", f"its index set holds {value!r}, which isn't an integer."
                )
        for value in values:
            block = (
                body
                if loop_parameter is None
                else body.assign_parameters(
                    {loop_parameter: int(value)}, inplace=False, strict=False
                )
            )
            self._phase(block.global_phase, in_branch=in_branch)
            self._circuit(block, qubits, clbits, out, in_branch=in_branch)

    def _switch(self, op, qubits, clbits, out: list[_Op], resolve, *, in_branch: bool) -> None:
        """Rewrite a switch as one ``if`` block per case value.

        The values are distinct, so at most one block runs, as in the switch. A
        default case becomes a block for each value no case names.
        """
        unsupported = UnsupportedInstructionError(
            "switch_case",
            "Maestro supports a switch on one clbit or a register, not on the classical "
            f"expression {op.target}.",
        )
        target = op.target
        if isinstance(target, Clbit):
            bits = [target]
        elif isinstance(target, ClassicalRegister):
            bits = list(target)
        else:
            bits = _operand_bits(target, unsupported)
        target_bits = {resolve(bit) for bit in bits}
        covered: set[int] = set()
        cases: list[tuple[int, QuantumCircuit]] = []
        default = None
        for values, block in op.cases_specifier():
            for value in values:
                if value is CASE_DEFAULT:
                    default = block
                else:
                    covered.add(int(value))
                    cases.append((int(value), block))
        if default is not None:
            missing = [v for v in range(2 ** len(bits)) if v not in covered]
            if len(missing) > _MAX_DEFAULT_BRANCHES:
                raise UnsupportedInstructionError(
                    "switch_case",
                    f"its default case covers {len(missing)} values of a {len(bits)}-bit "
                    f"target, which would need more than {_MAX_DEFAULT_BRANCHES} if blocks.",
                )
            cases.extend((value, default) for value in missing)
        for value, block in cases:
            ops: list[_Op] = []
            self._resolved_branch(
                _equals(bits, value, resolve),
                [(block, qubits, clbits)],
                None,
                ops,
                resolve,
                in_branch=in_branch,
                name="switch_case",
            )
            for if_op in ops:
                written = {c for o in _walk(if_op.true_body) for c in o.clbits}
                if written & target_bits:
                    raise UnsupportedInstructionError(
                        "switch_case",
                        "a case measures into the switch's own target, which a chain of if "
                        "blocks would read again.",
                    )
            out.extend(ops)

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
        self._resolved_branch(
            bits, true_body, false_block, out, resolve, in_branch=in_branch, name="if_else"
        )

    def _resolved_branch(
        self,
        bits: _Condition | None,
        true_body,
        false_block,
        out: list[_Op],
        resolve,
        *,
        in_branch: bool,
        name: str,
    ) -> None:
        """Append an ``if`` op on a condition already resolved to bits."""
        if self.estimator:
            raise UnsupportedInstructionError(
                name, "the Estimator's circuits can't measure, so they can't branch on results."
            )
        if in_branch:
            raise UnsupportedInstructionError(
                name, "Maestro doesn't support nested if_test blocks."
            )
        if bits is not None and false_block is not None and len(bits) > 1:
            raise UnsupportedInstructionError(
                "if_else",
                "Maestro supports an else branch only when the condition is a single clbit, "
                "not a register or a combination of bits.",
            )

        def body(items, *, nested: bool) -> tuple[_Op, ...]:
            ops: list[_Op] = []
            for item, item_qubits, item_clbits in items:
                if isinstance(item, QuantumCircuit):
                    self._circuit(item, item_qubits, item_clbits, ops, in_branch=nested)
                else:
                    self._instruction(
                        item, item_qubits, item_clbits, ops, resolve, in_branch=nested
                    )
            return tuple(ops)

        if bits is None:
            # The condition can never hold, as for a clbit compared with 2, so only the
            # else branch runs, and it runs unconditionally.
            if false_block is not None:
                out.extend(body([false_block], nested=in_branch))
            return

        self.result.has_control_flow = True
        out.append(
            _Op(
                "if",
                condition=bits,
                true_body=body(true_body, nested=True),
                false_body=None if false_block is None else body([false_block], nested=True),
            )
        )

    def _condition(self, condition, resolve) -> _Condition | None:
        """The bits that must hold for ``condition`` to be true, or ``None`` if it never is."""
        if isinstance(condition, tuple):
            target, value = condition
            if isinstance(target, Clbit):
                bits = [target]
            elif isinstance(target, ClassicalRegister):
                bits = list(target)
            else:
                raise UnsupportedInstructionError(
                    "if_else", f"unsupported condition {condition!r}."
                )
            return _equals(bits, int(value), resolve)
        return self._expr_condition(condition, resolve)

    def _expr_condition(self, condition, resolve) -> _Condition | None:
        unsupported = UnsupportedInstructionError(
            "if_else",
            "Maestro supports conditions on one clbit or on a register equal to a value, "
            f"not the classical expression {condition}.",
        )
        node = condition
        if isinstance(node, expr.Binary) and node.op == expr.Binary.Op.EQUAL:
            left, right = node.left, node.right
            if _is_constant(left):
                left, right = right, left
            bits = _operand_bits(left, unsupported)
            return _equals(bits, _constant(right, unsupported), resolve)
        if isinstance(node, expr.Unary) and node.op in (
            expr.Unary.Op.LOGIC_NOT,
            expr.Unary.Op.BIT_NOT,
        ):
            bits = _operand_bits(node.operand, unsupported)
            wanted = False
        else:
            bits = _operand_bits(node, unsupported)
            wanted = True
        if len(bits) != 1:
            raise unsupported
        return ((resolve(bits[0]), wanted),)


def _loop_variables(circuit: QuantumCircuit) -> set[Parameter]:
    """The loop variables of every for loop in ``circuit``, at any depth."""
    found = set()
    for instruction in circuit.data:
        operation = instruction.operation
        if not isinstance(operation, ControlFlowOp):
            continue
        if operation.name == "for_loop" and operation.params[1] is not None:
            found.add(operation.params[1])
        for block in operation.blocks:
            found |= _loop_variables(block)
    return found


def _equals(bits: list[Clbit], value: int, resolve) -> _Condition | None:
    """``bits`` (least significant first) read as an integer equal ``value``.

    A value that doesn't fit in the bits is never equal, so there is no condition.
    """
    if not 0 <= value < 2 ** len(bits):
        return None
    return tuple((resolve(bit), bool(value >> j & 1)) for j, bit in enumerate(bits))


def _operand_bits(node, unsupported: UnsupportedInstructionError) -> list[Clbit]:
    """The clbits, least significant first, that make up the unsigned integer ``node``.

    A cast to a narrower ``Uint`` keeps the low bits, as Qiskit and Aer evaluate it. A
    cast to a wider type adds zero high bits, which the list leaves implicit: a value
    needing them is out of range for :func:`_equals`. A cast of a register to ``Bool``
    is a non-zero test, which a conjunction of bits can't express.
    """
    if isinstance(node, expr.Var):
        if isinstance(node.var, Clbit):
            return [node.var]
        if isinstance(node.var, ClassicalRegister):
            return list(node.var)
        raise unsupported
    if isinstance(node, expr.Cast):
        bits = _operand_bits(node.operand, unsupported)
        if isinstance(node.type, types.Uint):
            return bits[: node.type.width]
        if isinstance(node.type, types.Bool) and len(bits) == 1:
            return bits
        raise UnsupportedInstructionError(
            "if_else",
            f"Maestro can't evaluate the cast to {node.type} of {len(bits)} bits in the "
            f"classical expression {node}.",
        )
    raise unsupported


def _is_constant(node) -> bool:
    while isinstance(node, expr.Cast):
        node = node.operand
    return isinstance(node, expr.Value)


def _constant(node, unsupported: UnsupportedInstructionError) -> int:
    """The integer value of a literal, through any casts around it."""
    if isinstance(node, expr.Value) and isinstance(node.type, (types.Bool, types.Uint)):
        return int(node.value)
    if isinstance(node, expr.Cast):
        value = _constant(node.operand, unsupported)
        if isinstance(node.type, types.Uint):
            return value & (2**node.type.width - 1)
        if isinstance(node.type, types.Bool):
            return int(value != 0)
    raise unsupported


def convert(
    circuit: QuantumCircuit, *, estimator: bool = False, noise=None, saves: bool = False
) -> ConvertedCircuit:
    """Convert a Qiskit circuit for Maestro.

    Args:
        circuit: The circuit, possibly with unbound parameters.
        estimator: Reject measurements and classical control, which an
            Estimator circuit can't have.
        noise: A :class:`~qiskit_maestro.noise.NoiseTranslator` for an Aer noise
            model. Its errors become ``noise`` placeholders, which
            :meth:`ConvertedCircuit.with_noise` translates.
        saves: Accept Aer's save instructions, as :class:`MaestroBackend` does.

    Raises:
        UnsupportedInstructionError: naming the first instruction Maestro can't run.
    """
    return _Converter(circuit, estimator=estimator, noise=noise, saves=saves).run()
