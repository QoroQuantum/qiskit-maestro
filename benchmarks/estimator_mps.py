# SPDX-FileCopyrightText: 2026 Qoro Quantum
# SPDX-License-Identifier: GPL-3.0-or-later

"""MaestroEstimator against Aer's EstimatorV2, both on matrix product states.

This is the comparison a Qiskit user makes: the same circuit and observables go
to both estimators, with the same bond-dimension cap and truncation threshold,
and each uses every CPU core, as it does by default. Each case runs once to
warm up and is then timed ``--repeats`` times; the median is reported. Timings
are printed only if the two estimators agree to 1e-4 of each value's size,
which catches wrong physics; with MPS truncation, both differ from the exact
values by about 1e-5 of that size, so a tighter check would test truncation.

``--common-basis`` first transpiles every circuit to rx, ry, rz and cx, so both
simulators apply the same gates. Without it, Aer applies ``rzz`` natively and
Maestro expands it into ``cx, rz, cx``.

The last case sends one PUB per time point, as dynamics code does. Maestro
recognises that the PUBs are the same Trotter step repeated and runs them as
one evolution; Aer simulates each from the start.

    pip install qiskit-maestro qiskit-aer
    python benchmarks/estimator_mps.py --qubits 120 200
"""

from __future__ import annotations

import argparse
import platform
import statistics
import time
from importlib import metadata

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import efficient_su2
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer.primitives import EstimatorV2 as AerEstimator

from qiskit_maestro import MaestroEstimator

BASIS = ["rx", "ry", "rz", "cx"]


def ising(num_qubits: int, j: float = 1.0, h: float = 0.7) -> SparsePauliOp:
    """Transverse-field Ising Hamiltonian on an open chain."""
    terms = [("ZZ", [q, q + 1], -j) for q in range(num_qubits - 1)]
    terms += [("X", [q], -h) for q in range(num_qubits)]
    return SparsePauliOp.from_sparse_list(terms, num_qubits)


def trotter(num_qubits: int, steps: int, dt: float = 0.1, j: float = 1.0, h: float = 0.7):
    """Ising time evolution from |0...0> by first-order Trotter steps."""
    circuit = QuantumCircuit(num_qubits)
    for _ in range(steps):
        for start in (0, 1):
            for q in range(start, num_qubits - 1, 2):
                circuit.rzz(-2 * j * dt, q, q + 1)
        for q in range(num_qubits):
            circuit.rx(-2 * h * dt, q)
    return circuit


def cases(num_qubits: int, seed: int = 7):
    rng = np.random.default_rng(seed)
    ansatz = efficient_su2(num_qubits, reps=3, entanglement="linear")
    values = rng.uniform(0, 2 * np.pi, ansatz.num_parameters)
    magnetisation = [
        SparsePauliOp.from_sparse_list([("Z", [q], 1.0)], num_qubits) for q in range(num_qubits)
    ]
    # Each case is a list of PUBs, sent to both estimators in one call.
    return {
        f"VQE energy: efficient_su2(reps=3), {num_qubits} qubits, one Ising Hamiltonian "
        f"({2 * num_qubits - 1} terms)": [(ansatz.assign_parameters(values), ising(num_qubits))],
        f"Trotter dynamics: 20 Ising steps, {num_qubits} qubits, <Z> on every qubit "
        f"({num_qubits} observables)": [(trotter(num_qubits, steps=20), magnetisation)],
        f"Trotter dynamics at every step: 0 to 20 Ising steps as 21 PUBs, {num_qubits} qubits, "
        f"energy and <Z> on every qubit": [
            (trotter(num_qubits, steps=k), [ising(num_qubits), *magnetisation]) for k in range(21)
        ],
    }


def timed(run, repeats: int) -> tuple[float, np.ndarray]:
    evs = run()  # Warm-up.
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        evs = run()
        times.append(time.perf_counter() - start)
    return statistics.median(times), evs


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--qubits", type=int, nargs="+", default=[120])
    parser.add_argument("--common-basis", action="store_true")
    parser.add_argument("--bond-dimension", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=1e-10)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    print(f"Platform: {platform.platform()}, {platform.processor() or platform.machine()}")
    for package in ("qiskit", "qiskit-aer", "qoro-maestro", "qiskit-maestro"):
        print(f"{package} {metadata.version(package)}")
    print(f"Bond dimension cap {args.bond_dimension}, truncation threshold {args.threshold}")
    print(f"Circuits {'transpiled to ' + ', '.join(BASIS) if args.common_basis else 'as built'}\n")

    maestro = MaestroEstimator(
        method="matrix_product_state",
        max_bond_dimension=args.bond_dimension,
        truncation_threshold=args.threshold,
    )
    aer = AerEstimator(
        options={
            "backend_options": {
                "method": "matrix_product_state",
                "matrix_product_state_max_bond_dimension": args.bond_dimension,
                "matrix_product_state_truncation_threshold": args.threshold,
            }
        }
    )

    print("| Case | Aer EstimatorV2 (s) | MaestroEstimator (s) | Speed-up | Max |Δ⟨O⟩| |")
    print("|---|---:|---:|---:|---:|")
    all_cases = {name: case for n in args.qubits for name, case in cases(n).items()}
    for name, pubs in all_cases.items():
        if args.common_basis:
            pubs = [
                (transpile(circuit, basis_gates=BASIS, optimization_level=1), observables)
                for circuit, observables in pubs
            ]

        def evs(estimator, pubs=pubs):
            return np.concatenate([r.data.evs.ravel() for r in estimator.run(pubs).result()])

        aer_time, aer_evs = timed(lambda: evs(aer), args.repeats)
        maestro_time, maestro_evs = timed(lambda: evs(maestro), args.repeats)
        gap = np.abs(np.asarray(aer_evs) - np.asarray(maestro_evs))
        difference = float(np.max(gap))
        if np.any(gap > 1e-4 * np.maximum(1.0, np.abs(aer_evs))):
            raise SystemExit(f"{name}: the estimators disagree by {difference:.2e}")
        print(
            f"| {name} | {aer_time:.3f} | {maestro_time:.3f} | "
            f"{aer_time / maestro_time:.1f}× | {difference:.1e} |"
        )


if __name__ == "__main__":
    main()
