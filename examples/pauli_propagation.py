# SPDX-FileCopyrightText: 2026 Qoro Quantum
# SPDX-License-Identifier: GPL-3.0-or-later

"""Expectation values on IBM's 115-qubit heavy-hex lattice, by Pauli propagation.

Pauli propagation runs the circuit backwards on the observable, a sum of Pauli
strings, instead of forwards on the state. A Clifford gate maps each string to
one string, while a rotation can split it in two. ``pp_coefficient_threshold``
drops the strings whose coefficients fall below it. The cost depends mostly on
how many strings there are, not on how many qubits, so it suits wide circuits
that a statevector can't hold, on lattices that an MPS handles badly, such as
the heavy-hex layout of IBM's devices.

The circuit is the kicked Ising model of IBM's utility experiment: 5 steps of
rx(π/8) on every qubit, then the Clifford rzz(−π/2) on every edge. The
observable is the average magnetization. With no threshold, Pauli propagation
is exact, and at this depth that's still fast. Each threshold below is faster
and less accurate. For deeper circuits the exact run grows out of reach, and a
threshold is what keeps them feasible: lower it until the value stops changing.

Run it with ``python examples/pauli_propagation.py``.
"""

import time

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler import CouplingMap

from qiskit_maestro import MaestroEstimator

lattice = CouplingMap.from_heavy_hex(7, bidirectional=False)
n = lattice.size()  # 115 qubits.
circuit = QuantumCircuit(n)
for _ in range(5):
    circuit.rx(np.pi / 8, range(n))
    for a, b in lattice.get_edges():
        circuit.rzz(-np.pi / 2, a, b)
magnetization = SparsePauliOp.from_sparse_list([("Z", [q], 1 / n) for q in range(n)], n)

results = {}
for threshold in [None, 1e-2, 1e-3, 1e-4]:  # None keeps every string: exact.
    estimator = MaestroEstimator(
        method="pauli_propagation",
        maestro_options={"pp_coefficient_threshold": threshold},
    )
    start = time.perf_counter()
    value = float(estimator.run([(circuit, magnetization)]).result()[0].data.evs)
    results[threshold] = value, time.perf_counter() - start

if __name__ == "__main__":
    exact = results[None][0]
    for threshold, (value, seconds) in results.items():
        label = "none (exact)" if threshold is None else f"{threshold:g}"
        print(
            f"threshold {label:>12}: <M> = {value:.6f}, error {abs(value - exact):.1e}, "
            f"{seconds:.2f} s"
        )
