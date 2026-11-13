# SPDX-FileCopyrightText: 2026 Qoro Quantum
# SPDX-License-Identifier: GPL-3.0-or-later

"""A Bell state on the backend, the Sampler and the Estimator.

Each class is a drop-in for the Qiskit or Aer class in its "was:" comment, and
returns results in the same form.

Run it with ``python examples/getting_started.py``.
"""

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp

from qiskit_maestro import MaestroBackend, MaestroEstimator, MaestroSampler

bell = QuantumCircuit(2)
bell.h(0)
bell.cx(0, 1)
measured = bell.measure_all(inplace=False)

# Counts from the backend.
backend = MaestroBackend()  # was: AerSimulator()
job = backend.run(transpile(measured, backend), shots=1000, seed_simulator=7)
counts = job.result().get_counts()

# Shots from the Sampler, as a BitArray named after the circuit's register.
sampler = MaestroSampler(seed=7)  # was: StatevectorSampler(seed=7)
bits = sampler.run([measured], shots=1000).result()[0].data.meas
sampled = bits.get_counts()

# Exact expectation values from the Estimator, with no shots.
estimator = MaestroEstimator()  # was: StatevectorEstimator()
observables = [SparsePauliOp("ZZ"), SparsePauliOp("XX"), SparsePauliOp("IZ")]
evs = estimator.run([(bell, observables)]).result()[0].data.evs

if __name__ == "__main__":
    print("backend counts:", counts)
    print("sampler counts:", sampled)
    for observable, value in zip(observables, evs):
        print(f"<{observable.paulis[0]}> = {value:+.3f}")
