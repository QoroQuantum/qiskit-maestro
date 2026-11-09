# SPDX-FileCopyrightText: 2026 Qoro Quantum
# SPDX-License-Identifier: GPL-3.0-or-later

"""An Aer script with a device noise model, moved to Maestro by changing two lines.

The two lines marked "was:" are the only changes from the Aer version. The
noise model comes from a fake device, as ``NoiseModel.from_backend`` builds it
for IBM devices: depolarizing and thermal-relaxation errors on ``sx``, ``x``,
``id`` and the two-qubit gate, noiseless ``rz``, relaxation during delays and
readout errors. With ``qiskit-ibm-runtime`` installed, ``FakeSherbrooke()``
works the same way.

Run it with ``python examples/aer_noise_model.py``.
"""

from qiskit import QuantumCircuit, transpile
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.quantum_info import SparsePauliOp, Statevector, state_fidelity
from qiskit_aer.noise import NoiseModel

from qiskit_maestro import MaestroBackend  # was: from qiskit_aer import AerSimulator

device = GenericBackendV2(5, seed=1234)
noise_model = NoiseModel.from_backend(device)
sim = MaestroBackend(noise_model=noise_model)  # was: sim = AerSimulator(noise_model=noise_model)

# A GHZ state, with a pause on the last qubit, transpiled to the noise model's gates.
ghz = QuantumCircuit(3)
ghz.h(0)
ghz.cx(0, 1)
ghz.cx(1, 2)
ghz.delay(2, 2, unit="us")
noisy = transpile(ghz, sim, seed_transpiler=7)

# Save the state before measuring it. With noise, Aer saves the statevector of
# one random noise trajectory, and Maestro does the same; the probabilities and
# the expectation value are averaged over all the shots.
noisy.save_statevector()
noisy.save_probabilities(label="probabilities")
noisy.save_expectation_value(SparsePauliOp("ZZZ"), [0, 1, 2], label="zzz")
noisy.measure_all()

result = sim.run(noisy, shots=4000, seed_simulator=11).result()
counts = result.get_counts()
state = result.get_statevector()
probabilities = result.data()["probabilities"]
zzz = result.data()["zzz"]

ideal = Statevector(ghz.remove_final_measurements(inplace=False))

if __name__ == "__main__":
    print("counts:", dict(sorted(counts.items(), key=lambda item: -item[1])))
    print("P(000), P(111):", round(probabilities[0], 4), round(probabilities[7], 4))
    print("<ZZZ>:", round(zzz, 4))
    print("fidelity of one noisy trajectory:", round(state_fidelity(state, ideal), 4))
