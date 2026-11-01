# SPDX-FileCopyrightText: 2026 Qoro Quantum
# SPDX-License-Identifier: GPL-3.0-or-later

"""Running converted circuits on Maestro, shared by the backend and the primitives."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, TypeVar

import numpy as np

from .bit_order import maestro_counts_to_outcomes
from .converter import ConvertedCircuit
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
    return {k: v for k, v in result.items() if k != "counts"}


def sample(
    circuit: ConvertedCircuit,
    angle_rows: np.ndarray,
    shots: int,
    options: Mapping[str, Any],
    seeds: Sequence[int | None],
) -> list[Samples]:
    """Sample every row of ``angle_rows``, in parallel where that's safe."""

    def one(i: int) -> Samples:
        config = simulator_config(options, seeds[i])
        if circuit.has_control_flow:
            maestro_circuit = circuit.to_maestro_qasm(angle_rows[i])
        else:
            maestro_circuit = circuit.to_maestro(angle_rows[i])
        result = maestro_circuit.execute(config=config, shots=shots)
        check_device(result, options)
        outcomes, frequencies = maestro_counts_to_outcomes(result["counts"], circuit.num_clbits)
        return Samples(outcomes, frequencies, _strip_counts(result))

    workers = parallel_workers(options, circuit.num_qubits, len(angle_rows))
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


def estimate(
    circuit: ConvertedCircuit,
    angle_rows: np.ndarray,
    labels: Sequence[list[str]],
    options: Mapping[str, Any],
    seeds: Sequence[int | None],
) -> list[tuple[np.ndarray, dict[str, Any] | None]]:
    """Expectation values of Maestro-order Pauli ``labels[i]`` for row ``i``.

    Returns ``(values, info)`` per row; ``info`` is ``None`` when the row needed
    no simulation.
    """

    options = _estimate_options(options)

    def one(i: int) -> tuple[np.ndarray, dict[str, Any] | None]:
        if not labels[i]:
            return np.zeros(0), None
        config = simulator_config(options, seeds[i])
        result = circuit.to_maestro(angle_rows[i]).estimate(list(labels[i]), config=config)
        check_device(result, options)
        info = {k: v for k, v in result.items() if k != "expectation_values"}
        return np.asarray(result["expectation_values"], dtype=float), info

    workers = parallel_workers(options, circuit.num_qubits, len(angle_rows))
    return run_parallel(one, len(angle_rows), workers)
