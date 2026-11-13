# SPDX-FileCopyrightText: 2026 Qoro Quantum
# SPDX-License-Identifier: GPL-3.0-or-later

"""Running converted circuits on Maestro, shared by the backend and the primitives."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, TypeVar

import maestro
import numpy as np

from .bit_order import maestro_counts_to_outcomes, maestro_samples_to_outcomes
from .converter import ConvertedCircuit, NoiseSpec, Save, build_maestro
from .exceptions import UnsupportedInstructionError
from .options import check_device, parallel_workers, simulator_config

T = TypeVar("T")

_MAX_SEED = 2**31 - 1


@dataclass
class Samples:
    """Sampled outcomes of one parameter set: unique rows and how often each occurred."""

    outcomes: np.ndarray  # (num_unique, num_clbits), clbit j in column j.
    frequencies: np.ndarray  # (num_unique,)
    info: Mapping[str, Any]  # Maestro's result dict, minus the counts.

    def shots(self, rng: np.random.Generator) -> np.ndarray:
        """Expand to one row per shot, in random order.

        Maestro returns counts only. Shots are independent, so shuffling the
        expanded counts gives the same distribution as recording them in the
        order they ran.
        """
        rows = np.repeat(self.outcomes, self.frequencies, axis=0)
        rng.shuffle(rows)
        return rows


def run_parallel(fn: Callable[[int], T], count: int, workers: int) -> list[T]:
    """``[fn(i) for i in range(count)]`` on up to ``workers`` threads.

    Maestro releases the GIL while it simulates, so the threads run in parallel.
    """
    if workers <= 1 or count <= 1:
        return [fn(i) for i in range(count)]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(fn, range(count)))


def maestro_seeds(rng: np.random.Generator, seeded: bool, count: int) -> list[int | None]:
    """A Maestro seed per task, drawn from ``rng`` so a seeded run is reproducible."""
    if not seeded:
        return [None] * count
    return [int(s) for s in rng.integers(0, _MAX_SEED, size=count)]


def _strip_counts(result: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in result.items() if k not in ("counts", "samples")}


def runs_own_workers(circuit: ConvertedCircuit) -> bool:
    """Whether Maestro spreads each of the circuit's calls over its own workers.

    Sampled noise draws its noise anew for every shot, and Maestro runs the shots
    of a call in parallel. Running such calls on threads too would oversubscribe
    the CPU, so their callers run them one at a time.
    """
    return circuit.noise is not None and circuit.noise.stochastic


def sample(
    circuit: ConvertedCircuit,
    angle_rows: np.ndarray,
    shots: int,
    options: Mapping[str, Any],
    seeds: Sequence[int | None],
) -> list[Samples]:
    """Sample every row of ``angle_rows``, in parallel where that's safe."""
    noise = circuit.noise

    def one(i: int) -> Samples:
        config = simulator_config(options, seeds[i])
        maestro_circuit = circuit.build(angle_rows[i])
        if noise is None or not (noise.active or noise.readout):
            result = maestro_circuit.execute(config=config, shots=shots)
        elif noise.stochastic:
            # A fresh noise draw for every shot, as in Aer's trajectories: one noise
            # realization per shot. Where the simulator samples the circuit's channels
            # from its state, every realization is the same circuit, and Maestro runs it
            # once per shot without building the others. With return_samples, Maestro
            # samples each realization directly even when seeded; with counts, seeded
            # calls take a slower path.
            result = maestro_circuit.noisy_execute(
                noise.maestro_model(),
                config=config,
                shots=shots,
                noise_realizations=shots,
                return_samples=True,
            )
        else:
            # Exact (density-matrix) noise is the same for every shot.
            result = maestro_circuit.noisy_execute(
                noise.maestro_model(), config=config, shots=shots, noise_realizations=1
            )
        check_device(result, options)
        if "samples" in result:
            outcomes, frequencies = maestro_samples_to_outcomes(
                result["samples"], circuit.num_clbits
            )
        else:
            outcomes, frequencies = maestro_counts_to_outcomes(result["counts"], circuit.num_clbits)
        return Samples(outcomes, frequencies, _strip_counts(result))

    workers = (
        1
        if runs_own_workers(circuit)
        else parallel_workers(options, circuit.num_qubits, len(angle_rows))
    )
    return run_parallel(one, len(angle_rows), workers)


def _estimate_options(options: Mapping[str, Any]) -> Mapping[str, Any]:
    """The options to estimate with, working around a Maestro 0.3.10 stabilizer bug.

    With Maestro's default ``auto_reduce=True``, some Clifford estimates fail with "The
    stabilizer simulator does not support the Rx gate", for example ``X`` on qubit 0 of a
    two-qubit circuit and the observable ``ZZ``. In the cases found, the circuit has a
    qubit that no gate touches, and the Pauli string has Z on that qubit and an operator
    that anticommutes with X on a qubit with an X gate. The failure is inside
    Maestro's reduction, which hands the stabilizer simulator an Rx. Every estimate with
    ``auto_reduce=False`` matched the exact value, so the stabilizer method turns it off
    unless the user sets it in ``maestro_options``. Drop this once a fixed Maestro is
    the minimum version.
    """
    maestro_options = options["maestro_options"] or {}
    if options["method"] != "stabilizer" or "auto_reduce" in maestro_options:
        return options
    return {**options, "maestro_options": {**maestro_options, "auto_reduce": False}}


def _expectations(
    maestro_circuit,
    labels: list[str],
    options: Mapping[str, Any],
    seed: int | None,
    noise: NoiseSpec | None,
    trajectories: int,
    *,
    stochastic: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Expectation values of Maestro-order Pauli strings on one Maestro circuit.

    A circuit with sampled noise, or with resets that a pure-state method applies
    to one random outcome per run (``stochastic``), is averaged over
    ``trajectories`` independent runs, as Aer averages over shots. Exact noise
    needs one run.
    """
    config = simulator_config(options, seed)
    stochastic = stochastic or (noise is not None and noise.stochastic)
    if (noise is None or not noise.active) and not stochastic:
        result = maestro_circuit.estimate(labels, config=config)
    else:
        model = noise.maestro_model(readout=False) if noise is not None else maestro.NoiseModel()
        result = maestro_circuit.noisy_estimate_montecarlo(
            labels, model, noise_realizations=trajectories if stochastic else 1, config=config
        )
    check_device(result, options)
    info = {k: v for k, v in result.items() if k != "expectation_values"}
    return np.asarray(result["expectation_values"], dtype=float), info


# Noise trajectories averaged for an expectation value on a pure-state method. Aer's
# EstimatorV2 averages over its backend's default 1024 shots.
NOISE_TRAJECTORIES = 1024


def estimate(
    circuit: ConvertedCircuit,
    angle_rows: np.ndarray,
    labels: Sequence[list[str]],
    options: Mapping[str, Any],
    seeds: Sequence[int | None],
    trajectories: int = NOISE_TRAJECTORIES,
) -> list[tuple[np.ndarray, dict[str, Any] | None]]:
    """Expectation values of Maestro-order Pauli ``labels[i]`` for row ``i``.

    Returns ``(values, info)`` per row; ``info`` is ``None`` when the row needed
    no simulation. With sampled noise, or resets on a pure-state method, values
    are averaged over ``trajectories`` runs.
    """

    options = _estimate_options(options)
    resets = options["method"] != "density_matrix" and any(op.kind == "reset" for op in circuit.ops)

    def one(i: int) -> tuple[np.ndarray, dict[str, Any] | None]:
        if not labels[i]:
            return np.zeros(0), None
        if circuit.qubit_map is None:
            reduced = list(labels[i])
        else:
            reduced = [_drop_idle(label, circuit.qubit_map) for label in labels[i]]
        needed = sorted({label for label in reduced if label and label.strip("I")})
        values = {"": 0.0}  # A string with X or Y on an idle qubit, in |0>.
        values.update({label: 1.0 for label in reduced if label and not label.strip("I")})
        info = None
        if needed:
            result, info = _expectations(
                circuit.to_maestro(angle_rows[i]),
                needed,
                options,
                seeds[i],
                circuit.noise,
                trajectories,
                stochastic=resets,
            )
            values.update(zip(needed, result.tolist()))
        return np.array([values[label] for label in reduced], dtype=float), info

    # Averaged runs go to Maestro's own workers, one row at a time.
    workers = (
        1
        if runs_own_workers(circuit) or resets
        else parallel_workers(options, circuit.num_qubits, len(angle_rows))
    )
    return run_parallel(one, len(angle_rows), workers)


def _drop_idle(label: str, keep: tuple[int, ...]) -> str:
    """A Maestro-order string on the qubits ``keep``, or ``""`` if its value is zero.

    The dropped qubits are never acted on, so they stay in |0>: Z and I there
    contribute a factor of one, and X or Y make the expectation value zero.
    """
    kept = set(keep)
    if any(letter in "XYxy" for q, letter in enumerate(label) if q not in kept):
        return ""
    reduced = "".join(label[q] if q < len(label) else "I" for q in keep)
    return reduced or "I"


# Pure-state methods whose state Maestro can return as a statevector.
_STATEVECTOR_METHODS = frozenset({"statevector", "matrix_product_state"})

# Amplitudes and probabilities smaller than this are left out of Aer's dict saves.
_CHOP = 1e-10

# Widest save_probabilities computed from Z expectation values (2**k of them).
_MAX_Z_PROBABILITY_QUBITS = 12


def _one_trajectory(ops: list, rng: np.random.Generator) -> list:
    """The ops with each Pauli channel replaced by one Pauli drawn from it.

    Channel ops stay: the statevector and the matrix product state draw one
    branch of each from their state as they run.
    """
    out = []
    for op in ops:
        if op.kind != "pauli_channel":
            out.append(op)
            continue
        index = int(rng.choice(len(op.probabilities), p=op.probabilities))
        for qubit in op.qubits:
            # Base-4 digit k of the index picks I, X, Y or Z on qubit k.
            pauli = "ixyz"[index % 4]
            index //= 4
            if pauli != "i":
                out.append(dataclasses.replace(op, kind=pauli, qubits=(qubit,), probabilities=()))
    return out


def _statevector(circuit: ConvertedCircuit, save: Save, row, options, seed) -> np.ndarray:
    """The statevector at the save, padded to the circuit's width and with its global phase.

    After noise or a reset the state is random. Aer then saves the state of one
    random trajectory, so this does too: Maestro draws one branch of each
    channel and one outcome for each reset. On the GPU, each Pauli channel is
    drawn here, and the resets include those of the ancilla that applies
    amplitude damping.
    """
    method = options["method"]
    if method not in _STATEVECTOR_METHODS:
        raise UnsupportedInstructionError(
            save.kind,
            f"the {method} method has no statevector to save. Use method='statevector' or "
            "'matrix_product_state'.",
        )
    prefix = circuit.ops[: save.position]
    if save.after_noise:
        prefix = _one_trajectory(prefix, np.random.default_rng(seed))
    n = circuit.num_qubits
    vector = np.asarray(
        build_maestro(prefix, row).get_statevector(config=simulator_config(options, seed)),
        dtype=complex,
    )
    # Maestro's width is the highest qubit an operation touches, plus one; the
    # untouched qubits above it are |0>, the high bits of the index. A noise ancilla,
    # where the method needs one, is above the circuit's qubits and reset to |0> after
    # every use.
    if len(vector) < 2**n:
        vector = np.concatenate([vector, np.zeros(2**n - len(vector), dtype=complex)])
    vector = vector[: 2**n]
    return vector * np.exp(1j * circuit.phase(row, save.phases))


def _pauli_on(label: str, qubits: tuple[int, ...], num_qubits: int) -> str:
    """A Qiskit-order label on ``qubits`` as a Maestro-order string on the whole circuit."""
    chars = ["I"] * num_qubits
    for j, letter in enumerate(reversed(label)):
        chars[qubits[j]] = letter
    return "".join(chars)


def _save_expectations(
    circuit: ConvertedCircuit, save: Save, strings: list[str], row, options, seed, shots
) -> tuple[dict[str, float], dict[str, Any] | None]:
    """Values of Maestro-order strings at the save's position, averaged over shots if random."""
    values = {s: 1.0 for s in strings if not s.strip("I")}
    needed = sorted({s for s in strings if s.strip("I")})
    if not needed:
        return values, None
    noise = circuit.noise if save.after_noise else None
    stochastic = save.after_reset and options["method"] != "density_matrix"
    result, info = _expectations(
        build_maestro(circuit.ops[: save.position], row),
        needed,
        _estimate_options(options),
        seed,
        noise,
        shots,
        stochastic=stochastic,
    )
    values.update(zip(needed, result.tolist()))
    return values, info


def _probabilities(
    circuit: ConvertedCircuit, save: Save, row, options, seed, shots
) -> tuple[np.ndarray, dict[str, Any] | None]:
    """Outcome probabilities of the save's qubits, the first being the least significant bit."""
    k = len(save.qubits)
    n = circuit.num_qubits
    if k > _MAX_Z_PROBABILITY_QUBITS:
        if save.after_noise or save.after_reset:
            raise UnsupportedInstructionError(
                save.kind,
                "after noise or a reset, probabilities are averaged over shots from Z "
                f"expectation values, which supports up to {_MAX_Z_PROBABILITY_QUBITS} qubits.",
            )
        vector = _statevector(circuit, save, row, options, seed)
        marginal = (np.abs(vector) ** 2).reshape([2] * n)
        # Axis j of the reshaped array is qubit n - 1 - j.
        keep = [n - 1 - q for q in reversed(save.qubits)]
        drop = tuple(a for a in range(n) if a not in keep)
        marginal = marginal.sum(axis=drop) if drop else marginal
        order = sorted(keep)
        marginal = np.transpose(marginal, [order.index(a) for a in keep])
        return marginal.reshape(-1), None
    # P(b) = 2**-k * sum over subsets m of (-1)**|b & m| <Z_m>.
    strings = []
    for m in range(2**k):
        chars = ["I"] * n
        for j, qubit in enumerate(save.qubits):
            if m >> j & 1:
                chars[qubit] = "Z"
        strings.append("".join(chars))
    values, info = _save_expectations(circuit, save, strings, row, options, seed, shots)
    z = np.array([values[s] for s in strings])
    h = 1
    while h < len(z):
        for start in range(0, len(z), 2 * h):
            a = z[start : start + h].copy()
            b = z[start + h : start + 2 * h].copy()
            z[start : start + h] = a + b
            z[start + h : start + 2 * h] = a - b
        h *= 2
    probabilities = z / 2**k
    probabilities[np.abs(probabilities) < 1e-15] = 0.0
    return np.clip(probabilities, 0.0, None), info


def saved_data(
    circuit: ConvertedCircuit,
    row: np.ndarray,
    options: Mapping[str, Any],
    seed: int | None,
    shots: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """The data of the circuit's Aer save instructions, in the types Aer returns.

    Returns the data by label, and Maestro's result info for each simulation.
    """
    from qiskit.quantum_info import Statevector
    from qiskit.result import ProbDistribution

    data: dict[str, Any] = {}
    infos = []
    for save in circuit.saves:
        info = None
        if save.kind in ("save_statevector", "save_statevector_dict"):
            vector = _statevector(circuit, save, row, options, seed)
            if save.kind == "save_statevector":
                data[save.label] = Statevector(vector)
            else:
                data[save.label] = {
                    hex(i): complex(a) for i, a in enumerate(vector) if abs(a) > _CHOP
                }
        elif save.kind in ("save_probabilities", "save_probabilities_dict"):
            probabilities, info = _probabilities(circuit, save, row, options, seed, shots)
            if save.kind == "save_probabilities":
                data[save.label] = probabilities
            else:
                data[save.label] = ProbDistribution(
                    {i: float(p) for i, p in enumerate(probabilities) if p > _CHOP}
                )
        else:
            strings = [
                _pauli_on(label, save.qubits, circuit.num_qubits) for label, *_ in save.terms
            ]
            values, info = _save_expectations(circuit, save, strings, row, options, seed, shots)
            mean = sum(c * values[s] for s, (_, c, _) in zip(strings, save.terms))
            if save.kind == "save_expval":
                data[save.label] = float(mean)
            else:
                square = sum(c2 * values[s] for s, (_, _, c2) in zip(strings, save.terms))
                data[save.label] = np.array([mean, square - mean**2])
        if info is not None:
            infos.append(info)
    return data, infos
