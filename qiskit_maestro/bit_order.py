# SPDX-FileCopyrightText: 2026 Qoro Quantum
# SPDX-License-Identifier: GPL-3.0-or-later

"""Conversion between Maestro's and Qiskit's bit orders.

Maestro writes qubit 0 and clbit 0 as the *leftmost* character of bitstrings and
Pauli strings. Qiskit writes them as the *rightmost*. Every value that crosses
between the two goes through this module, so the rest of the package never
reverses a string itself.

Inside the package, sampled outcomes are held as ``uint8`` arrays with one
column per clbit, column ``j`` holding clbit ``j``. That layout has no
left-or-right, and every Qiskit format (counts keys, memory strings, packed
:class:`~qiskit.primitives.BitArray` data) is produced from it here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

__all__ = [
    "maestro_counts_to_outcomes",
    "outcomes_to_bitstrings",
    "outcomes_to_hex",
    "pack_register",
    "pauli_label_to_maestro",
]


def pauli_label_to_maestro(label: str) -> str:
    """Convert a Qiskit Pauli label to Maestro's order.

    Qiskit's ``"XIZ"`` is Z on qubit 0 and X on qubit 2. Maestro spells the same
    operator ``"ZIX"``.
    """
    return label[::-1]


def maestro_counts_to_outcomes(
    counts: Mapping[str, int], num_clbits: int
) -> tuple[np.ndarray, np.ndarray]:
    """Turn Maestro counts into outcome rows and their frequencies.

    Maestro keys counts by bitstrings with clbit 0 first, and each string is only
    as long as the highest clbit the circuit wrote, plus one. Missing clbits were
    never written and are zero, as in Qiskit.

    Returns:
        ``(outcomes, frequencies)``: ``outcomes`` has shape
        ``(len(counts), num_clbits)`` with clbit ``j`` in column ``j``.
    """
    outcomes = np.zeros((len(counts), num_clbits), dtype=np.uint8)
    frequencies = np.zeros(len(counts), dtype=np.int64)
    for row, (bits, count) in enumerate(counts.items()):
        if len(bits) > num_clbits and "1" in bits[num_clbits:]:
            raise ValueError(
                f"Maestro returned outcome {bits!r} for a circuit with {num_clbits} clbits."
            )
        width = min(len(bits), num_clbits)
        outcomes[row, :width] = np.frombuffer(bits[:width].encode(), dtype=np.uint8) - ord("0")
        frequencies[row] = count
    return outcomes, frequencies


def maestro_samples_to_outcomes(samples, num_clbits: int) -> tuple[np.ndarray, np.ndarray]:
    """Turn Maestro's per-shot rows (``return_samples=True``) into outcome rows and frequencies.

    Each row has clbit 0 first and is as wide as the highest clbit the circuit
    wrote, plus one; missing clbits are zero. The result has the form of
    :func:`maestro_counts_to_outcomes`'s.
    """
    rows = np.asarray(samples, dtype=np.uint8)
    rows = rows.reshape(len(rows), -1)
    if rows.shape[1] > num_clbits and rows[:, num_clbits:].any():
        raise ValueError(
            f"Maestro returned {rows.shape[1]}-bit outcomes for a circuit with {num_clbits} clbits."
        )
    padded = np.zeros((len(rows), num_clbits), dtype=np.uint8)
    width = min(rows.shape[1], num_clbits)
    padded[:, :width] = rows[:, :width]
    outcomes, frequencies = np.unique(padded, axis=0, return_counts=True)
    return outcomes, frequencies.astype(np.int64)


def _pack(bits: np.ndarray) -> np.ndarray:
    """Pack rows whose columns run from the most to the least significant bit."""
    pad = -bits.shape[-1] % 8
    if pad:
        bits = np.pad(bits, [(0, 0)] * (bits.ndim - 1) + [(pad, 0)])
    return np.packbits(bits, axis=-1)


def outcomes_to_hex(outcomes: np.ndarray) -> list[str]:
    """Qiskit memory strings (``"0x5"``) for outcome rows; bit ``j`` is clbit ``j``."""
    if outcomes.shape[-1] == 0:
        return ["0x0"] * len(outcomes)
    packed = _pack(outcomes[:, ::-1])
    return [hex(int.from_bytes(row.tobytes(), "big")) for row in packed]


def outcomes_to_bitstrings(outcomes: np.ndarray) -> list[str]:
    """Qiskit bitstrings for outcome rows, clbit 0 last (rightmost)."""
    chars = (outcomes[:, ::-1] + ord("0")).astype(np.uint8)
    return [row.tobytes().decode() for row in chars]


def pack_register(shots: np.ndarray, clbits: Sequence[int]) -> np.ndarray:
    """Pack one classical register into :class:`~qiskit.primitives.BitArray` layout.

    Args:
        shots: Outcome rows, shape ``(..., num_shots, num_clbits)``.
        clbits: The circuit clbit index of each register bit, register bit 0
            first.

    Returns:
        ``uint8`` array of shape ``(..., num_shots, ceil(len(clbits) / 8))``,
        big-endian, so register bit 0 is the lowest bit of the last byte.
    """
    columns = np.asarray(clbits, dtype=np.intp)[::-1]
    return _pack(shots[..., columns])
