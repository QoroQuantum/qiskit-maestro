# SPDX-FileCopyrightText: 2026 Qoro Quantum
# SPDX-License-Identifier: GPL-3.0-or-later

"""Run the backend, the Sampler and the Estimator on Maestro in the Qoro cloud.

Get an API key at https://dash.qoroquantum.net, then:

    pip install "qiskit-maestro[cloud]"
    export QORO_API_KEY=...   # or put it in a .env file
    python examples/qoro_cloud.py

The only change from a local run is ``service=service``. Each ``run()`` sends
one cloud job and returns at once; ``result()`` waits for it. Results come back
in the same form and bit order as local runs. Add ``device="GPU"`` to run on the
Qoro GPU cluster.
"""

import numpy as np
from divi.backends import QoroService
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import efficient_su2
from qiskit.quantum_info import SparsePauliOp

from qiskit_maestro import MaestroBackend, MaestroEstimator, MaestroSampler

service = QoroService()  # Reads QORO_API_KEY from a .env file or the environment.

# A 20-qubit GHZ state on the backend, with MPS. The job has a cloud job ID
# you can check on, and result() waits for it.
ghz = QuantumCircuit(20)
ghz.h(0)
for q in range(19):
    ghz.cx(q, q + 1)
ghz.measure_all()

backend = MaestroBackend(service=service, method="mps")
job = backend.run(transpile(ghz, backend), shots=1000)
job_id, status = job.job_id(), job.status()
counts = job.result().get_counts()

# A sweep over 8 parameter sets on the Sampler. It goes as one circuit
# template, and the cloud binds the values.
ansatz = efficient_su2(4, reps=1)
values = np.random.default_rng(0).uniform(0, np.pi, (8, ansatz.num_parameters))
sampler = MaestroSampler(service=service)
samples = sampler.run([(ansatz.measure_all(inplace=False), values)], shots=500).result()
bits = samples[0].data.meas  # A BitArray of shape (8,).

# The same sweep on the Estimator. Expectation values are exact, as locally.
hamiltonian = SparsePauliOp(["IIZZ", "IZZI", "ZZII", "XXXX"], [1.0, 1.0, 1.0, 0.5])
estimator = MaestroEstimator(service=service)
evs = estimator.run([(ansatz, hamiltonian, values)]).result()[0].data.evs  # Shape (8,).

if __name__ == "__main__":
    print(f"backend job {job_id}: {status.name}")
    print("GHZ counts:", counts)
    print("sampler, first parameter set:", bits.get_counts(0))
    print("estimator:", np.round(evs, 4))
