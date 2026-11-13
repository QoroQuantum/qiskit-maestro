# SPDX-FileCopyrightText: 2026 Qoro Quantum
# SPDX-License-Identifier: GPL-3.0-or-later

"""One GHZ circuit on three simulation methods, from 20 to 1,000 qubits.

A GHZ state puts every qubit into (|00…0⟩ + |11…1⟩)/√2, so its shots are all
zeros or all ones, about half each. Each method below runs it at a size the
others can't reach:

- ``"statevector"`` stores all 2ⁿ amplitudes: easy at 20 qubits, out of reach
  at 100.
- ``"matrix_product_state"`` needs only bond dimension 2 for a GHZ state,
  however many qubits it has.
- ``"stabilizer"`` runs Clifford circuits, like this one, in polynomial time.

The method is a run option, so one backend runs all three.

Run it with ``python examples/simulation_methods.py``.
"""

import time

from qiskit import QuantumCircuit

from qiskit_maestro import MaestroBackend


def ghz(num_qubits: int) -> QuantumCircuit:
    circuit = QuantumCircuit(num_qubits)
    circuit.h(0)
    for q in range(num_qubits - 1):
        circuit.cx(q, q + 1)
    circuit.measure_all()
    return circuit


def outcome(bitstring: str) -> str:
    return {"0": "all zeros", "1": "all ones"}.get("".join(set(bitstring)), bitstring)


sizes = {"statevector": 20, "matrix_product_state": 200, "stabilizer": 1000}
backend = MaestroBackend(seed_simulator=7)
results = {}
for method, num_qubits in sizes.items():
    start = time.perf_counter()
    result = backend.run(ghz(num_qubits), shots=1000, method=method).result()
    seconds = time.perf_counter() - start
    counts = {outcome(bits): shots for bits, shots in result.get_counts().items()}
    results[method] = num_qubits, counts, seconds

if __name__ == "__main__":
    for method, (num_qubits, counts, seconds) in results.items():
        print(f"{num_qubits:>5} qubits  {method:<21} {counts}  {seconds * 1000:.0f} ms")
