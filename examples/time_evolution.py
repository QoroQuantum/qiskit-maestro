# SPDX-FileCopyrightText: 2026 Qoro Quantum
# SPDX-License-Identifier: GPL-3.0-or-later

"""Trotter evolution of a 50-qubit Ising chain, measured at every time step.

``MaestroEstimator`` notices when the PUBs of one ``run`` call are one circuit
followed by 0, 1, 2, ... repetitions of the same step. It then simulates the
longest circuit once and reads every time point on the way, so measuring at N
steps costs N steps instead of about N²/2. The first half writes one circuit
per time point, as you would for any estimator; the second gives the step to
``run_evolution`` directly. Both give the same magnetisation curve.

Run it with ``python examples/time_evolution.py``.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from qiskit_maestro import MaestroEstimator

n = 50
steps = range(21)

step = QuantumCircuit(n)  # One Trotter step of a transverse-field Ising chain.
for start in (0, 1):
    for q in range(start, n - 1, 2):
        step.rzz(-0.2, q, q + 1)
step.rx(-0.14, range(n))
magnetisation = SparsePauliOp.from_sparse_list([("Z", [q], 1 / n) for q in range(n)], n)
estimator = MaestroEstimator(method="matrix_product_state")

# One circuit per time point.
circuits, circuit = [], QuantumCircuit(n)
for _ in steps:
    circuits.append(circuit.copy())
    circuit.compose(step, inplace=True)
results = estimator.run([(c, magnetisation) for c in circuits]).result()
per_circuit = np.array([float(r.data.evs) for r in results])

# The same evolution, given as an initial state and a step.
evolution = estimator.run_evolution(QuantumCircuit(n), step, magnetisation, steps=steps)
from_step = evolution.result()[0].data.evs  # One value per step.

if __name__ == "__main__":
    for k, value in zip(steps, from_step):
        print(f"step {k:2d}: <Z> = {value:+.4f}")
    print("both ways agree:", np.allclose(per_circuit, from_step))
