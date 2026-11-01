# SPDX-FileCopyrightText: 2026 Qoro Quantum
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import maestro
import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit.random import random_circuit

try:
    import qiskit_aer  # noqa: F401

    HAS_AER = True
except ImportError:
    HAS_AER = False

requires_aer = pytest.mark.skipif(not HAS_AER, reason="qiskit-aer isn't installed")


def pytest_collection_modifyitems(config, items):
    if maestro.is_gpu_available():
        return
    skip = pytest.mark.skip(reason="Maestro's GPU backend isn't available")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)


def assert_same_distribution(counts_a: dict, counts_b: dict, *, sigmas: float = 5.0) -> None:
    """Two samples come from the same distribution, outcome by outcome.

    Each outcome's frequency difference must be within ``sigmas`` standard
    errors of the pooled estimate, plus a little slack for outcomes too rare to
    have a meaningful standard error.
    """
    shots_a, shots_b = sum(counts_a.values()), sum(counts_b.values())
    for outcome in set(counts_a) | set(counts_b):
        p_a = counts_a.get(outcome, 0) / shots_a
        p_b = counts_b.get(outcome, 0) / shots_b
        pooled = (counts_a.get(outcome, 0) + counts_b.get(outcome, 0)) / (shots_a + shots_b)
        error = np.sqrt(pooled * (1 - pooled) * (1 / shots_a + 1 / shots_b))
        assert abs(p_a - p_b) <= sigmas * error + 2e-3, (
            f"outcome {outcome}: {p_a:.4f} vs {p_b:.4f} (allowed {sigmas * error + 2e-3:.4f})"
        )


def assert_matches_probabilities(counts: dict, probabilities: dict, *, sigmas: float = 5.0) -> None:
    """A sample is consistent with exact outcome probabilities."""
    shots = sum(counts.values())
    for outcome in set(counts) | set(probabilities):
        p = probabilities.get(outcome, 0.0)
        p_hat = counts.get(outcome, 0) / shots
        error = np.sqrt(p * (1 - p) / shots)
        assert abs(p_hat - p) <= sigmas * error + 2e-3, f"outcome {outcome}: {p_hat:.4f} vs {p:.4f}"


def seeded_random_circuits(count: int, num_qubits: int, depth: int, *, measure: bool, seed: int):
    """Random circuits with gates of up to three qubits, from Qiskit's generator."""
    return [
        random_circuit(num_qubits, depth, max_operands=3, measure=measure, seed=seed + i)
        for i in range(count)
    ]


def qft_gate(num_qubits: int):
    """``QFTGate``, or the older ``QFT`` circuit as a gate before Qiskit 1.2."""
    try:
        from qiskit.circuit.library import QFTGate
    except ImportError:
        from qiskit.circuit.library import QFT

        return QFT(num_qubits).to_gate()
    return QFTGate(num_qubits)


def efficient_su2(num_qubits: int, reps: int, entanglement: str = "linear") -> QuantumCircuit:
    """``efficient_su2``, or the older ``EfficientSU2`` class before Qiskit 1.3."""
    try:
        from qiskit.circuit.library import efficient_su2 as build
    except ImportError:
        from qiskit.circuit.library import EfficientSU2

        return EfficientSU2(num_qubits, reps=reps, entanglement=entanglement).decompose()
    return build(num_qubits, reps=reps, entanglement=entanglement)


def ghz(num_qubits: int, *, measure: bool) -> QuantumCircuit:
    circuit = QuantumCircuit(num_qubits)
    circuit.h(0)
    for q in range(num_qubits - 1):
        circuit.cx(q, q + 1)
    if measure:
        circuit.measure_all()
    return circuit
