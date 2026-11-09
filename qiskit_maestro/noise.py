# SPDX-FileCopyrightText: 2026 Qoro Quantum
# SPDX-License-Identifier: GPL-3.0-or-later

"""Translate Qiskit Aer noise models for Maestro.

Aer attaches an error to an instruction, chosen by its name and qubits, so an
``rz`` can be noiseless while an ``sx`` on the same qubit is not. Maestro's
:class:`maestro.NoiseModel` attaches noise to a qubit after every gate on it,
and the plugin expands gates such as ``ecr`` into several Maestro gates. A
direct mapping would therefore put noise where Aer has none, and repeat it on
expanded gates. Instead, each Aer error becomes operations placed exactly where
Aer applies it: after the instruction, or before it for ``measure``.

An error is written as a Pauli channel followed by amplitude damping on each of
its qubits. That covers depolarizing, Pauli, bit-flip and phase-flip errors,
phase damping, amplitude damping, thermal relaxation (with any T2 <= 2 T1),
reset errors, and their compositions, such as the errors of
``NoiseModel.from_backend``. The Pauli part becomes a ``pauli_channel``. The
damping becomes, on density-matrix runs, a ``delay`` under Maestro's idle noise,
which Maestro applies as an exact thermal-relaxation channel; on pure-state
runs, it becomes an exact amplitude damping on an ancilla qubit, sampled per
shot like Aer's trajectories.

Every translation is checked: the operations must reproduce Aer's channel to
1e-9 in every entry of its Pauli transfer matrix. Errors that fail, such as
coherent over-rotations, raise :class:`UnsupportedNoiseError` naming the
instruction and qubits. No noise is dropped.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cache, reduce

import maestro
import numpy as np

from .exceptions import UnsupportedNoiseError

__all__ = ["Channel", "NoiseTranslator", "is_aer_noise_model"]

# Absolute tolerance on Pauli transfer matrix entries for a translation to count as exact.
_TOLERANCE = 1e-9

# chi[P, Q] = +1 if the one-qubit Paulis P and Q (I, X, Y, Z) commute, else -1. A Pauli
# channel with probabilities p has Pauli transfer matrix diag(chi @ p), and chi @ chi = 4 I.
_CHI = np.array([[1, 1, 1, 1], [1, 1, -1, -1], [1, -1, 1, -1], [1, -1, -1, 1]], dtype=float)

# Instructions Aer never attaches noise to, whatever the model says.
_NOISELESS = frozenset({"barrier"})

_PAULI_MATRICES = (
    np.eye(2, dtype=complex),
    np.array([[0, 1], [1, 0]], dtype=complex),
    np.array([[0, -1j], [1j, 0]], dtype=complex),
    np.array([[1, 0], [0, -1]], dtype=complex),
)
_PAULI_INDEX = {"I": 0, "X": 1, "Y": 2, "Z": 3}
_PAULI_GATES = {"id": 0, "x": 1, "y": 2, "z": 3}

# Kraus operators of a reset to |0>.
_RESET = (
    np.array([[1, 0], [0, 0]], dtype=complex),
    np.array([[0, 1], [0, 0]], dtype=complex),
)


@cache
def _pauli_basis(n: int) -> np.ndarray:
    """The n-qubit Paulis, shape ``(4**n, 2**n, 2**n)``; base-4 digit k of the index is qubit k."""
    basis = np.array(
        [
            reduce(np.kron, [_PAULI_MATRICES[i // 4**k % 4] for k in reversed(range(n))])
            for i in range(4**n)
        ]
    )
    basis.flags.writeable = False
    return basis


@cache
def _pauli_ptm(index: int, n: int) -> np.ndarray:
    """PTM of the n-qubit Pauli ``index``: +1 on the Paulis it commutes with, -1 on the rest."""
    signs = reduce(np.kron, [_CHI[index // 4**k % 4] for k in reversed(range(n))])
    ptm = np.diag(signs)
    ptm.flags.writeable = False
    return ptm


def _kraus_ptm(kraus) -> np.ndarray:
    """PTM of the channel with these Kraus operators: ``R[i, j] = Tr(P_i E(P_j)) / d``."""
    kraus = np.asarray(kraus, dtype=complex)
    d = kraus.shape[-1]
    paulis = _pauli_basis(d.bit_length() - 1)
    # images[j] = sum over m of K_m P_j K_m^dagger.
    images = (kraus[:, None] @ paulis[None] @ kraus.conj().transpose(0, 2, 1)[:, None]).sum(axis=0)
    # Tr(P_i M) = sum over a, b of P_i[a, b] M[b, a].
    return paulis.reshape(d * d, -1) @ images.transpose(0, 2, 1).reshape(d * d, -1).T / d


_IDENTITY_PTM = np.eye(4)
_IDENTITY_PTM.flags.writeable = False
# Swaps the two base-4 digits of a two-qubit PTM index.
_SWAP_DIGITS = np.array([i // 4 + 4 * (i % 4) for i in range(16)])


def _embed(ptm: np.ndarray, qubits: tuple[int, ...], n: int) -> np.ndarray | None:
    """An instruction's PTM on ``qubits`` of an error on ``n <= 2`` qubits, or ``None``."""
    if qubits == tuple(range(n)):
        return ptm
    if n == 2 and qubits == (0,):
        # kron(identity, ptm): qubit 0 is the less significant digit.
        return (_IDENTITY_PTM[:, None, :, None] * ptm[None, :, None, :]).reshape(16, 16)
    if n == 2 and qubits == (1,):
        return (ptm[:, None, :, None] * _IDENTITY_PTM[None, :, None, :]).reshape(16, 16)
    if n == 2 and qubits == (1, 0):
        return ptm[np.ix_(_SWAP_DIGITS, _SWAP_DIGITS)]
    return None


def _instruction_ptm(operation) -> np.ndarray | None:
    """PTM of one instruction of an Aer error, on its own qubits, or ``None`` if unknown."""
    name = operation.name
    if name in _PAULI_GATES:
        return _pauli_ptm(_PAULI_GATES[name], 1)
    if name == "pauli":
        # The label's last letter acts on the instruction's first qubit. A sign or
        # phase prefix doesn't change the channel.
        label = str(operation.params[0]).lstrip("+-i")
        index = sum(_PAULI_INDEX[letter] * 4**k for k, letter in enumerate(reversed(label)))
        return _pauli_ptm(index, len(label))
    if name == "reset" and operation.num_qubits == 1:
        return _kraus_ptm(_RESET)
    if name == "kraus":
        return _kraus_ptm(operation.params)
    if getattr(operation, "num_clbits", 0) == 0 and hasattr(operation, "to_matrix"):
        from qiskit.circuit.exceptions import CircuitError

        try:
            matrix = operation.to_matrix()
        except (CircuitError, TypeError, ValueError):
            # No matrix, or unbound parameters: Qiskit's conversion decides.
            return None
        return _kraus_ptm([matrix])
    return None


def _content_key(operation, qubits: tuple[int, ...]):
    """A key equal for instructions with the same channel, or ``None`` if unsure."""
    try:
        params = tuple(
            np.asarray(p, dtype=complex).tobytes() if isinstance(p, np.ndarray) else p
            for p in operation.params
        )
        hash(params)
    except (TypeError, ValueError):
        return None
    return operation.name, qubits, params


def error_ptm(error) -> np.ndarray:
    """The Pauli transfer matrix of an Aer ``QuantumError`` on up to two qubits.

    Computed from each circuit's instructions, which is much faster than going
    through :class:`~qiskit.quantum_info.SuperOp`. Qiskit's conversion is the
    fallback for instructions this doesn't know. Both index the PTM with qubit 0
    as the least significant base-4 digit.
    """
    n = error.num_qubits
    total = np.zeros((4**n, 4**n))
    # Aer's composed errors repeat the same Kraus instruction in many circuits.
    seen: dict = {}
    for circuit, probability in zip(error.circuits, error.probabilities):
        ptm = None
        for instruction in circuit.data:
            operation = instruction.operation
            qubits = tuple(circuit.find_bit(q).index for q in instruction.qubits)
            key = None if instruction.clbits else _content_key(operation, qubits)
            part = seen.get(key) if key is not None else None
            if part is None:
                part = None if instruction.clbits else _instruction_ptm(operation)
                if part is not None:
                    part = _embed(part, qubits, n)
                if part is None or np.max(np.abs(part.imag)) > _TOLERANCE:
                    from qiskit.quantum_info import PTM

                    return np.asarray(PTM(error.to_quantumchannel()).data, dtype=complex)
                part = part.real
                if key is not None:
                    seen[key] = part
            ptm = part if ptm is None else part @ ptm
        total += probability * (np.eye(4**n) if ptm is None else ptm)
    return total


def is_aer_noise_model(value) -> bool:
    try:
        from qiskit_aer.noise import NoiseModel
    except ImportError:
        return False
    return isinstance(value, NoiseModel)


@dataclass(frozen=True)
class Channel:
    """An Aer error as a Pauli channel followed by amplitude damping.

    ``paulis`` holds 4**n probabilities in ``pauli_channel`` order (base-4 digit
    k picks I, X, Y or Z on qubit k), or is ``None`` for the identity.
    ``decays`` holds ``(k, a)`` for each qubit k whose damping keeps a fraction
    ``a < 1`` of the excited population.
    """

    paulis: tuple[float, ...] | None
    decays: tuple[tuple[int, float], ...]


def _damping_ptm(a: float, excited: float) -> np.ndarray:
    """Pauli transfer matrix of generalized amplitude damping keeping ``a`` of the population.

    Coherences shrink by sqrt(a), as for T2 = 2 T1, and the state relaxes towards
    excited-state population ``excited``.
    """
    c = math.sqrt(a)
    return np.array(
        [[1, 0, 0, 0], [0, c, 0, 0], [0, 0, c, 0], [(1 - a) * (1 - 2 * excited), 0, 0, a]]
    )


def decompose(ptm: np.ndarray, excited: tuple[float, ...]) -> Channel | str:
    """Write a channel as a Pauli channel followed by damping on each qubit.

    Args:
        ptm: The channel's Pauli transfer matrix, with qubit 0 the least
            significant base-4 digit of the index, as Qiskit orders it.
        excited: The excited-state population the damping relaxes towards, per qubit.

    Returns:
        The :class:`Channel`, or a string saying why there is none.
    """
    n = len(excited)
    decays = []
    for k, population in enumerate(excited):
        # The only non-unital entry the decomposition allows: <Z_k> of the output for
        # the maximally mixed input. A Pauli channel leaves that at zero.
        z0 = ptm[3 * 4**k, 0]
        if abs(z0) <= _TOLERANCE:
            decays.append(1.0)
            continue
        if abs(1 - 2 * population) < 1e-12:
            return "it relaxes towards a state other than its qubits' thermal state"
        a = 1 - z0 / (1 - 2 * population)
        if not 0 < a <= 1 + _TOLERANCE:
            return "it relaxes towards a state other than its qubits' thermal state"
        decays.append(min(a, 1.0))
    damping = reduce(np.kron, [_damping_ptm(decays[k], excited[k]) for k in reversed(range(n))])
    pauli_part = np.linalg.solve(damping, ptm)
    if np.max(np.abs(pauli_part - np.diag(np.diag(pauli_part)))) > _TOLERANCE:
        return (
            "it isn't a Pauli channel combined with amplitude damping or thermal relaxation, "
            "as a coherent (unitary) error or a general Kraus channel isn't"
        )
    chi = reduce(np.kron, [_CHI] * n)
    probabilities = chi @ np.diag(pauli_part) / 4**n
    if probabilities.min() < -_TOLERANCE:
        return "it isn't a Pauli channel combined with amplitude damping or thermal relaxation"
    probabilities = np.clip(probabilities, 0.0, None)
    probabilities /= probabilities.sum()
    if np.max(np.abs(damping @ np.diag(chi @ probabilities) - ptm)) > _TOLERANCE:
        return "it isn't a Pauli channel combined with amplitude damping"
    identity = abs(probabilities[0] - 1.0) <= 1e-15
    return Channel(
        paulis=None if identity else tuple(float(p) for p in probabilities),
        decays=tuple((k, a) for k, a in enumerate(decays) if a < 1.0),
    )


class NoiseTranslator:
    """Looks up an Aer noise model's errors per instruction and translates them.

    Lookups follow Aer: a local error for the instruction's name and qubits wins
    over an all-qubit error for the name; errors on ``measure`` act before the
    measurement and all others after the instruction; custom noise passes such
    as ``from_backend``'s delay relaxation act on the circuit's top-level
    instructions only. Translations are computed on first use and cached.
    """

    def __init__(self, model):
        self.model = model
        self._local = dict(getattr(model, "_local_quantum_errors", {}) or {})
        self._default = dict(getattr(model, "_default_quantum_errors", {}) or {})
        if getattr(model, "_nonlocal_quantum_errors", None):
            raise UnsupportedNoiseError(
                "noise_model",
                "Aer's nonlocal errors (on qubits other than the instruction's) aren't supported.",
            )
        self._passes = []
        for noise_pass in getattr(model, "_custom_noise_passes", []) or []:
            func = getattr(noise_pass, "_func", None)
            method = getattr(noise_pass, "_method", None)
            if func is None or method not in ("append", "prepend"):
                raise UnsupportedNoiseError(
                    "noise_model",
                    f"its custom noise pass {type(noise_pass).__name__} isn't supported. Only "
                    "passes that add errors before or after instructions, such as "
                    "RelaxationNoisePass, are.",
                )
            self._passes.append((getattr(noise_pass, "_ops", None), func, method))
        self._p1 = {}
        for noise_pass in getattr(model, "_custom_noise_passes", []) or []:
            p1s = getattr(noise_pass, "_p1s", None)
            if p1s is not None:
                self._p1.update({q: float(p) for q, p in enumerate(p1s)})
        self._entries: list[tuple[object, str, tuple[int, ...]]] = []
        self._keys: dict[tuple[int, str, tuple[int, ...]], int] = {}
        self._channels: dict[tuple[int, str], Channel] = {}

    @property
    def has_quantum_errors(self) -> bool:
        """Whether the model has any quantum errors, as Aer's automatic method checks."""
        return bool(self._local or self._default or self._passes)

    def _key(self, error, name: str, qubits: tuple[int, ...]) -> int:
        lookup = (id(error), name, qubits)
        if lookup not in self._keys:
            self._keys[lookup] = len(self._entries)
            self._entries.append((error, name, qubits))
        return self._keys[lookup]

    def errors(self, op, qubits, *, top_level: bool) -> tuple[list[int], list[int]]:
        """Keys of the errors before and after ``op`` on ``qubits``."""
        name = op.name
        before: list[int] = []
        after: list[int] = []
        if name in _NOISELESS:
            return before, after
        qubits = tuple(qubits)
        error = self._local.get(name, {}).get(qubits)
        if error is None:
            error = self._default.get(name)
        if error is not None:
            if error.num_qubits != len(qubits):
                raise UnsupportedNoiseError(
                    name,
                    f"its {error.num_qubits}-qubit noise doesn't fit an instruction on "
                    f"{len(qubits)} qubits.",
                )
            (before if name == "measure" else after).append(self._key(error, name, qubits))
        if top_level:
            for op_types, func, method in self._passes:
                if op_types and not isinstance(op, op_types):
                    continue
                error = func(op, list(qubits))
                if error is None:
                    continue
                if not hasattr(error, "to_quantumchannel"):
                    from qiskit_aer.noise import QuantumError

                    error = QuantumError(error)
                (before if method == "prepend" else after).append(self._key(error, name, qubits))
        return before, after

    def describe(self, key: int) -> tuple[str, tuple[int, ...]]:
        _, name, qubits = self._entries[key]
        return name, qubits

    def excited_population(self, qubit: int) -> float:
        """The equilibrium excited-state population of a qubit, from the model's relaxation pass."""
        return self._p1.get(qubit, 0.0)

    def channel(self, key: int, family: str) -> Channel:
        """Translate the error ``key`` for ``family``: ``"exact"`` or ``"sampled"``.

        ``"exact"`` is for density-matrix runs, where damping may relax towards a
        thermal state. ``"sampled"`` is for pure-state runs, where it must relax
        towards ``|0>``.

        Raises:
            UnsupportedNoiseError: if the error can't be reproduced exactly.
        """
        cached = self._channels.get((key, family))
        if cached is not None:
            return cached
        error, name, qubits = self._entries[key]
        if error.num_qubits > 2:
            raise UnsupportedNoiseError(
                name,
                f"its noise on qubits {list(qubits)} acts on {error.num_qubits} qubits; Maestro "
                "places noise channels on one or two.",
                qubits,
            )
        ptm = np.asarray(error_ptm(error), dtype=complex)
        if np.max(np.abs(ptm.imag)) > _TOLERANCE:
            raise UnsupportedNoiseError(name, "its noise isn't a valid channel.", qubits)
        excited = tuple(self.excited_population(q) if family == "exact" else 0.0 for q in qubits)
        result = decompose(ptm.real, excited)
        if isinstance(result, str):
            hint = ""
            thermal = tuple(self.excited_population(q) for q in qubits)
            if family == "sampled" and not isinstance(decompose(ptm.real, thermal), str):
                hint = (
                    " It relaxes towards an excited-state population, which Maestro applies "
                    "exactly only with method='density_matrix'."
                )
            raise UnsupportedNoiseError(
                name,
                f"its noise on qubits {list(qubits)} can't be reproduced exactly: {result}.{hint}",
                qubits,
            )
        self._channels[(key, family)] = result
        return result

    def readout(self, qubit: int) -> tuple[float, float] | None:
        """``(P(1|0), P(0|1))`` for measurements of ``qubit``, or ``None``."""
        local = getattr(self.model, "_local_readout_errors", {}) or {}
        error = local.get((qubit,))
        if error is None:
            error = getattr(self.model, "_default_readout_error", None)
        if error is None:
            return None
        probabilities = np.asarray(error.probabilities, dtype=float)
        return float(probabilities[0][1]), float(probabilities[1][0])

    def check_readout(self) -> None:
        """Reject readout errors Maestro can't apply: correlated ones on several qubits."""
        local = getattr(self.model, "_local_readout_errors", {}) or {}
        default = getattr(self.model, "_default_readout_error", None)
        wide = [qubits for qubits in local if len(qubits) != 1]
        if wide or (default is not None and default.number_of_qubits != 1):
            raise UnsupportedNoiseError(
                "measure",
                "correlated readout errors on several qubits aren't supported; Maestro "
                "applies readout error to each measured qubit independently.",
            )


def maestro_noise_model(
    idle: dict[int, float], readout: dict[int, tuple[float, float]]
) -> maestro.NoiseModel:
    """The Maestro noise model a translated circuit runs under.

    ``idle`` maps each qubit that has damping ``delay`` operations to its
    excited-state population. Delays are in units of T1, with T2 = 2 T1, so a
    delay of ``-ln(a)`` keeps a fraction ``a`` of the excited population.
    """
    model = maestro.NoiseModel()
    for qubit, population in idle.items():
        model.set_idle_noise(qubit, 1.0, 2.0, population)
    for qubit, (p10, p01) in readout.items():
        model.set_readout_error(qubit, p10, p01)
    return model
