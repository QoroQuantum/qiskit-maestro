# SPDX-FileCopyrightText: 2026 Qoro Quantum
# SPDX-License-Identifier: GPL-3.0-or-later

"""Pauli propagation on MaestroEstimator against IBM's pauli-prop: accuracy against time.

Pauli propagation evolves the observable instead of the state, and drops the
Pauli strings whose coefficients fall below a threshold, so a smaller
threshold is more accurate and slower. The two libraries apply it on different
schedules: pauli-prop after every gate, Maestro every few gates unless
``pp_gates_between_trims=1``. The same threshold therefore isn't the same work,
and each library gets its own range of thresholds. Compare the error each
reaches against the time it takes, one row per threshold.

The circuit is the kicked Ising model of IBM's utility experiment: each step
applies rx(θ) to every qubit, then rzz(−π/2) on every edge. The observable is
the average magnetization, (1/n) Σ Zᵢ. On a chain, the exact value comes from
Maestro's statevector. The 115-qubit heavy-hex lattice of IBM's devices is too
wide for a statevector, so there, watch where the values settle as the
threshold falls.

This is a short version of the Pauli propagation benchmark in Maestro's
documentation, https://qoroquantum.github.io/maestro/benchmarks.html, which ran
20 steps on a 28-qubit chain and on the heavy-hex lattice, with up to an hour
per point. The defaults here, 12 steps on a 20-qubit chain, finish in under a
minute. The second command below runs the documentation's heavy-hex case at
its faster thresholds, in about a minute.

    pip install qiskit-maestro pauli-prop
    python benchmarks/pauli_propagation.py
    python benchmarks/pauli_propagation.py --lattice heavy-hex --steps 20 \\
        --thresholds 1e-2 1e-3 --pauli-prop-thresholds 1e-3 1e-4 1e-5
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import platform
import statistics
import time
from importlib import metadata

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from qiskit_maestro import MaestroEstimator


def lattice(name: str, num_qubits: int) -> tuple[int, list[tuple[int, int]]]:
    """``(qubits, edges)`` of a chain, or of IBM's heavy-hex lattice at distance 7 (115 qubits)."""
    if name == "chain":
        return num_qubits, [(q, q + 1) for q in range(num_qubits - 1)]
    from qiskit.transpiler import CouplingMap

    coupling = CouplingMap.from_heavy_hex(7, bidirectional=False)
    return coupling.size(), sorted({(min(a, b), max(a, b)) for a, b in coupling.get_edges()})


def kicked_ising(num_qubits: int, edges, steps: int, theta: float) -> QuantumCircuit:
    circuit = QuantumCircuit(num_qubits)
    for _ in range(steps):
        circuit.rx(theta, range(num_qubits))
        for a, b in edges:
            circuit.rzz(-np.pi / 2, a, b)
    return circuit


def magnetization(num_qubits: int) -> SparsePauliOp:
    return SparsePauliOp.from_sparse_list(
        [("Z", [q], 1 / num_qubits) for q in range(num_qubits)], num_qubits
    )


def run_maestro(threshold: float, gates_between_trims: int | None = None):
    """Run on MaestroEstimator. Returns ``(value, whether a cap limited it)``."""
    options = {"pp_coefficient_threshold": threshold}
    if gates_between_trims is not None:
        options["pp_gates_between_trims"] = gates_between_trims
    estimator = MaestroEstimator(method="pauli_propagation", maestro_options=options)

    def run(circuit, observable):
        return float(estimator.run([(circuit, observable)]).result()[0].data.evs), False

    return run


def run_pauli_prop(threshold: float, max_terms: int):
    """Run on pauli-prop. Returns ``(value, whether the max_terms cap filled)``."""
    from pauli_prop import propagate_through_circuit

    def run(circuit, observable):
        # Heisenberg frame, U† O U. On |0...0>, only the strings of I and Z
        # contribute, each with value 1.
        evolved, _ = propagate_through_circuit(
            observable, circuit, max_terms=max_terms, atol=threshold, frame="h"
        )
        diagonal = ~evolved.paulis.x.any(axis=1)
        return float(np.real(evolved.coeffs[diagonal].sum())), len(evolved) >= max_terms

    return run


def timed(run, repeats: int) -> tuple[float, tuple[float, bool]]:
    """The median time of ``repeats`` runs; a run over 30 s is timed once."""
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = run()
        times.append(time.perf_counter() - start)
        if times[-1] > 30:
            break
    return statistics.median(times), result


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--lattice", choices=["chain", "heavy-hex"], default="chain")
    parser.add_argument("--qubits", type=int, default=20, help="chain length (heavy-hex has 115)")
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--theta", type=float, default=math.pi / 8, help="rx angle (default π/8)")
    parser.add_argument(
        "--thresholds", type=float, nargs="+", default=[1e-3, 1e-4, 1e-5], help="for Maestro"
    )
    parser.add_argument(
        "--pauli-prop-thresholds", type=float, nargs="+", default=[1e-4, 1e-5, 1e-6]
    )
    parser.add_argument(
        "--max-terms",
        type=int,
        default=10_000_000,
        help="pauli-prop's cap on Pauli strings, which it allocates up front",
    )
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    num_qubits, edges = lattice(args.lattice, args.qubits)
    circuit = kicked_ising(num_qubits, edges, args.steps, args.theta)
    observable = magnetization(num_qubits)

    has_pauli_prop = importlib.util.find_spec("pauli_prop") is not None

    print(f"Platform: {platform.platform()}, {platform.processor() or platform.machine()}")
    for package in ("qiskit", "qoro-maestro", "qiskit-maestro", "pauli-prop"):
        try:
            print(f"{package} {metadata.version(package)}")
        except metadata.PackageNotFoundError:
            print(f"{package} not installed")
    print(
        f"Kicked Ising on a {num_qubits}-qubit {args.lattice}, {args.steps} steps, "
        f"θ = {args.theta:.4f}, average magnetization\n"
    )

    exact = None
    if args.lattice == "chain":
        start = time.perf_counter()
        exact = float(MaestroEstimator().run([(circuit, observable)]).result()[0].data.evs)
        print(f"Exact value {exact:.6f} (statevector, {time.perf_counter() - start:.1f} s)\n")

    runners = {
        "Maestro": (run_maestro, args.thresholds),
        "Maestro, trim every gate": (lambda t: run_maestro(t, 1), args.thresholds),
    }
    if has_pauli_prop:
        runners["pauli-prop"] = (
            lambda t: run_pauli_prop(t, args.max_terms),
            args.pauli_prop_thresholds,
        )

    # Load each library before timing it.
    small = kicked_ising(4, [(0, 1), (1, 2), (2, 3)], 1, args.theta)
    for make, _ in runners.values():
        make(1e-2)(small, magnetization(4))

    print("| Simulator | Threshold | Value | Error | Time (s) |")
    print("|---|---:|---:|---:|---:|")
    cap_filled = False
    for name, (make, thresholds) in runners.items():
        for threshold in thresholds:
            run = make(threshold)
            seconds, (value, capped) = timed(lambda run=run: run(circuit, observable), args.repeats)
            error = "–" if exact is None else f"{abs(value - exact):.4f}"
            cap_filled |= capped
            print(
                f"| {name} | {threshold:g} | {value:.4f}{' †' if capped else ''} | {error} | "
                f"{seconds:.2f} |"
            )
    if cap_filled:
        print(
            f"\n† pauli-prop filled its cap of {args.max_terms:,} Pauli strings, so the cap, "
            "not the threshold, limited its accuracy. Raise --max-terms if memory allows."
        )
    if not has_pauli_prop:
        print("\npauli-prop isn't installed, so only Maestro ran: pip install pauli-prop")


if __name__ == "__main__":
    main()
