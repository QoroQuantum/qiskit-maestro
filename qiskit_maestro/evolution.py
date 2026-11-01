# SPDX-FileCopyrightText: 2026 Qoro Quantum
# SPDX-License-Identifier: GPL-3.0-or-later

"""Time evolution on one persistent Maestro simulator.

Maestro's ``incremental_evolve`` runs an initial circuit once, applies a step
circuit again and again, and computes expectation values on the live state at
chosen step counts. Measuring at n points of an N-step evolution then costs N
steps, where simulating each time point as its own circuit costs about n·N/2.

:func:`find_chain` recognises that structure in a batch of converted circuits,
so the Estimator can use it for Qiskit code that builds one circuit per time
point.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import gcd, isqrt
from typing import Any

import maestro
import numpy as np

from .converter import _Op, _Slot, build_maestro
from .options import check_device, simulator_config

__all__ = ["Chain", "evolve", "find_chain"]

# Angles closer than this count as equal when matching steps. LieTrotter(reps=k)
# computes each repetition's angle as (k·dt)/k, which differs from dt in the
# last bits.
ANGLE_TOLERANCE = 1e-12


@dataclass(frozen=True)
class Chain:
    """Circuits that are ``initial`` followed by ``step`` repeated ``steps[i]`` times."""

    initial: list[_Op]
    step: list[_Op]
    steps: list[int]


def _signature(ops: list[_Op]) -> tuple[list[tuple], np.ndarray]:
    """The ops without their angles, and all their angles as one array."""
    shape = [(op.kind, op.qubits, op.clbits, len(op.angles)) for op in ops]
    angles = np.asarray([a for op in ops for a in op.angles], dtype=float)
    return shape, angles


def _divisors(n: int) -> list[int]:
    small = [d for d in range(1, isqrt(n) + 1) if n % d == 0]
    return sorted(set(small + [n // d for d in small]))


def find_chain(op_lists: Sequence[list[_Op]]) -> Chain | None:
    """Find an initial circuit and a step that build every circuit in ``op_lists``.

    Each circuit must be the shortest one followed by whole repetitions of the
    same step. The circuits must have bound parameters and no classical
    control. Returns ``None`` when there's no such structure, or when all the
    circuits are the same length.
    """
    if len(op_lists) < 2:
        return None
    for ops in op_lists:
        for op in ops:
            if op.kind in ("if", "measure") or any(isinstance(a, _Slot) for a in op.angles):
                return None
    signatures = [_signature(ops) for ops in op_lists]
    longest = max(range(len(op_lists)), key=lambda i: len(op_lists[i]))
    long_shape, long_angles = signatures[longest]
    # Every circuit has to be a prefix of the longest.
    for shape, angles in signatures:
        if shape != long_shape[: len(shape)]:
            return None
        if not np.allclose(angles, long_angles[: len(angles)], rtol=0, atol=ANGLE_TOLERANCE):
            return None

    base = min(len(ops) for ops in op_lists)
    extra = [len(ops) - base for ops in op_lists]
    period = gcd(*extra)
    if period == 0:
        return None
    tail_shape = long_shape[base:]
    tail_angles = long_angles[sum(entry[3] for entry in long_shape[:base]) :]
    # The step is the shortest repeating unit of the tail that every circuit
    # ends on a whole number of.
    for length in _divisors(period):
        repeats = len(tail_shape) // length
        step_shape = tail_shape[:length]
        if tail_shape != step_shape * repeats:
            continue
        per_step = sum(entry[3] for entry in step_shape)
        if per_step and not np.allclose(
            tail_angles.reshape(repeats, per_step),
            tail_angles[:per_step],
            rtol=0,
            atol=ANGLE_TOLERANCE,
        ):
            continue
        ops = op_lists[longest]
        return Chain(
            list(ops[:base]), list(ops[base : base + length]), [e // length for e in extra]
        )
    return None


def evolve(
    initial: list[_Op],
    step: list[_Op],
    labels: Sequence[str],
    steps: Sequence[int],
    options: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any] | None]:
    """Expectation values of Maestro-order Pauli ``labels`` after each count in ``steps``.

    Returns ``(values, info)``: ``values`` has one row per entry of ``steps``, in
    the order given; ``info`` is ``None`` when no simulation was needed.
    """
    steps = [int(s) for s in steps]
    if not labels:
        return np.zeros((len(steps), 0)), None
    config = simulator_config(options, options["seed_simulator"])
    result = maestro.incremental_evolve(
        build_maestro(initial), build_maestro(step), sorted(set(steps)), list(labels), config
    )
    # incremental_evolve echoes the configured enums rather than the ints that
    # execute and estimate report.
    info = {
        key: getattr(value, "value", value)
        for key, value in result.items()
        if key not in ("expectation_values", "steps", "dynamic_bond_dims", "time_per_step")
    }
    check_device(info, options)
    rows = dict(zip(result["steps"], result["expectation_values"]))
    bond_dimensions = dict(zip(result["steps"], result["dynamic_bond_dims"]))
    info["bond_dimensions"] = [bond_dimensions[s] for s in steps]
    return np.asarray([rows[s] for s in steps], dtype=float), info
