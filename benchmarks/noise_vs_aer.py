# SPDX-FileCopyrightText: 2026 Qoro Quantum
# SPDX-License-Identifier: GPL-3.0-or-later

"""MaestroBackend against AerSimulator under an IBM device's noise model.

This is the comparison a Qiskit user makes after changing one import: the same
transpiled circuit and the same ``NoiseModel.from_backend`` model go to both
backends, with the same method and shots, and each uses every CPU core, as it
does by default. Each case runs once to warm up and is then timed ``--repeats``
times; the median is reported.

Both simulators keep worker threads that spin for a moment after a run, and
those slow the other one down, which shows on runs of a few milliseconds. The
script therefore waits ``--settle`` seconds before timing each backend. For
the cleanest numbers, time them in separate processes with ``--skip-aer`` and
``--skip-maestro``.

A case is a random circuit on ``n`` qubits of the device, routed on those
qubits only and written in the device's basis gates, so the noise model's
errors apply to every ``sx``, ``x`` and ``cx`` and ``rz`` stays noiseless.
``--layout device`` takes a connected patch of the coupling map, and
``--layout chain`` a line through it.

Three noise models: ``ideal`` (none), ``pauli`` (the device's depolarizing gate
errors and readout errors, without relaxation) and ``full`` (with thermal
relaxation, as ``from_backend`` gives by default).

Before timing, each case checks the two backends against each other: the
density matrix must give Aer's outcome probabilities to 1e-9, and the sampled
methods' counts must agree with Aer's within sampling error in every qubit's
marginal, in the parity of every pair of consecutive qubits and in the parity
of all of them.

A matrix product state whose bond limit truncates the state is not compared
that way: the two simulators then keep different approximations of it. Each is
measured instead against the exact statistics of the case, from Maestro's
statevector under the same noise, up to 20 qubits: the largest difference in
those marginals and parities between the counts of its timed run and the
statevector's. That is a number to read beside the times, not a pass or a
fail. It holds the sampling error of both sets of counts, printed with it, and
being the largest of many differences it reaches two to three times that
without any truncation.

The first line printed, and every ``--json`` record, says what ran: the
versions of Qiskit, Aer, Maestro and this plugin (with the commit, for a
checkout), the QCSim revision Maestro was built against where Maestro reports
it, and the threads they could use. A record also holds the circuit's seed,
the bond limit and, where Maestro reports it, the workers it ran the noise
realizations on.

    pip install qiskit-maestro qiskit-aer qiskit-ibm-runtime
    python benchmarks/noise_vs_aer.py --methods density_matrix --qubits 12 --depth 20 --shots 1000
    python benchmarks/noise_vs_aer.py --methods statevector matrix_product_state --qubits 8 12 16 20
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import time
import warnings
from importlib import metadata
from pathlib import Path

import maestro
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.random import random_circuit
from qiskit.transpiler import CouplingMap
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

from qiskit_maestro import MaestroBackend

NOISE = ("ideal", "pauli", "full")


def placement(num_qubits: int, layout: str):
    """The smallest fake IBM device with a cx basis that holds the layout, and its qubits."""
    from qiskit_ibm_runtime import fake_provider

    for name in ("FakeGuadalupeV2", "FakeMontrealV2", "FakeWashingtonV2"):
        device = getattr(fake_provider, name)()
        if device.num_qubits < num_qubits:
            continue
        qubits = chain(device, num_qubits) if layout == "chain" else patch(device, num_qubits)
        if qubits:
            return device, qubits
    raise ValueError(f"No fake device holds a {layout} layout of {num_qubits} qubits.")


def _neighbours(device) -> dict[int, list[int]]:
    graph: dict[int, set[int]] = {q: set() for q in range(device.num_qubits)}
    for a, b in device.coupling_map.get_edges():
        graph[a].add(b)
        graph[b].add(a)
    return {q: sorted(others) for q, others in graph.items()}


def patch(device, num_qubits: int) -> list[int]:
    """A connected set of the device's qubits, grown breadth first from qubit 0."""
    graph = _neighbours(device)
    chosen, queue = [0], [0]
    while queue and len(chosen) < num_qubits:
        for other in graph[queue.pop(0)]:
            if other not in chosen and len(chosen) < num_qubits:
                chosen.append(other)
                queue.append(other)
    return chosen


def chain(device, num_qubits: int) -> list[int] | None:
    """A line of the device's qubits, each coupled to the next, or None if it has none that long."""
    graph = _neighbours(device)

    def extend(path: list[int]) -> list[int] | None:
        if len(path) == num_qubits:
            return path
        for other in graph[path[-1]]:
            if other not in path:
                found = extend(path + [other])
                if found:
                    return found
        return None

    for start in range(device.num_qubits):
        found = extend([start])
        if found:
            return found
    return None


def workload(device, qubits: list[int], depth: int, seed: int) -> QuantumCircuit:
    """A random circuit on ``qubits`` of the device, in its basis gates, as wide as the device."""
    n = len(qubits)
    logical = random_circuit(n, depth, max_operands=2, measure=False, seed=seed)
    edges = [
        (qubits.index(a), qubits.index(b))
        for a, b in device.coupling_map.get_edges()
        if a in qubits and b in qubits
    ]
    routed = transpile(
        logical,
        coupling_map=CouplingMap(edges),
        basis_gates=["rz", "sx", "x", "cx"],
        optimization_level=1,
        seed_transpiler=seed,
    )
    circuit = QuantumCircuit(device.num_qubits, n)
    circuit.compose(routed, qubits=qubits, inplace=True)
    return circuit


def noise_model(device, kind: str) -> NoiseModel | None:
    if kind == "ideal":
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return NoiseModel.from_backend(device, thermal_relaxation=kind == "full")


def _run(backend, measured, shots: int):
    """One run as a user waits for it, counts included; returns the result."""
    result = backend.run(measured, shots=shots).result()
    result.get_counts()
    return result


def _timed(run, repeats: int, settle: float, warm_up: bool = True):
    """The times of ``repeats`` runs, and what the last one returned."""
    time.sleep(settle)  # Let the other simulator's worker threads go idle.
    if warm_up:
        run()  # Noise translation, thread pools, allocation.
    times, last = [], None
    for _ in range(repeats):
        start = time.perf_counter()
        last = run()
        times.append(time.perf_counter() - start)
    return times, last


def truncates(options, n: int) -> bool:
    """Whether a matrix product state's bond limit is below what ``n`` qubits can need."""
    bond = options.get("matrix_product_state_max_bond_dimension")
    return bond is not None and bond < 2 ** (n // 2)


def _events(n: int) -> dict[str, tuple[int, ...]]:
    """Statistics that are well sampled at any width, each the probability of an event.

    Each qubit's marginal, the parity of each pair of consecutive qubits of the case,
    which a marginal does not see, and the parity of them all.
    """
    events = {f"the marginal of qubit {bit}": (bit,) for bit in range(n)}
    events.update(
        {f"the parity of qubits {bit} and {bit + 1}": (bit, bit + 1) for bit in range(n - 1)}
    )
    events["the parity of all qubits"] = tuple(range(n))
    return events


def _check(circuit, measured, qubits, model, method, options, shots) -> str:
    """Compare the two backends on the case; returns what was compared."""
    n = len(qubits)
    if method == "density_matrix":
        saved = circuit.copy()
        saved.save_probabilities(qubits)
        aer = AerSimulator(method=method, noise_model=model).run(saved).result()
        mine = MaestroBackend(method=method, noise_model=model).run(saved).result()
        worst = float(np.max(np.abs(aer.data(0)["probabilities"] - mine.data(0)["probabilities"])))
        if worst > 1e-9:
            raise AssertionError(f"probabilities differ from Aer's by {worst:.2e}")
        return f"probabilities within {worst:.0e}"
    # Aer's density matrix is the reference while it is quick; beyond that, Aer's own samples
    # of the method, at the run's shots (Aer's noisy trajectories take minutes there).
    check_shots = max(shots, 4000) if n <= 12 else shots
    reference_method = "density_matrix" if n <= 12 and model is not None else method
    reference = AerSimulator(method=reference_method, noise_model=model, **options)
    aer = reference.run(measured, shots=check_shots).result().get_counts()
    mine = MaestroBackend(method=method, noise_model=model, **options)
    counts = mine.run(measured, shots=check_shots).result().get_counts()
    allowed = 5 * np.sqrt(0.5 / check_shots) + 2e-3
    worst, worst_name = 0.0, ""
    for name, bits in _events(n).items():
        p_aer = _odd(aer, bits, n) / check_shots
        p_mine = _odd(counts, bits, n) / check_shots
        if abs(p_aer - p_mine) / allowed > worst:
            worst, worst_name = abs(p_aer - p_mine) / allowed, name
    if worst > 1:
        raise AssertionError(f"{worst_name} is {worst:.1f} times its allowed difference")
    return f"marginals and parities within {worst:.2f} of 5 sigma"


# Above this the exact reference of a truncated run, a statevector, is no longer quick.
EXACT_REFERENCE_QUBITS = 20


def truncation_error(measured, model, n: int, shots: int, sampled) -> tuple[str, dict | None]:
    """How far a bond limit took each simulator from the exact statistics of the case.

    Two matrix product states that truncate keep different approximations, so they are
    not compared with each other. Each is measured against Maestro's statevector, which
    runs the same noise without truncating: the largest difference, over the marginals
    and parities, between the counts of its timed run and the statevector's. This is a
    number to read beside the times, not a check: a bond limit is an approximation, and
    the difference also holds the sampling error of both sets of counts. The sigma
    returned with it bounds that of one difference; the largest of them all reaches two
    to three times it without any truncation.

    ``sampled`` maps a simulator's name to the counts of its last timed run. Returns the
    line to print and, for a record, the errors with the sigma and the reference's shots.
    """
    if n > EXACT_REFERENCE_QUBITS:
        return f"truncated; no exact reference above {EXACT_REFERENCE_QUBITS} qubits", None
    exact_shots = max(shots, 2000) if n <= 16 else shots
    exact = MaestroBackend(method="statevector", noise_model=model)
    reference = exact.run(measured, shots=exact_shots).result().get_counts()
    errors = {
        name: max(
            abs(_odd(counts, bits, n) / shots - _odd(reference, bits, n) / exact_shots)
            for bits in _events(n).values()
        )
        for name, counts in sampled.items()
    }
    sigma = float(np.sqrt(0.25 / shots + 0.25 / exact_shots))
    listed = ", ".join(f"{name} {error:.3f}" for name, error in errors.items())
    report = {"worst_error": errors, "sampling_sigma": sigma, "reference_shots": exact_shots}
    return f"truncated; worst error {listed} (sampling sigma {sigma:.3f})", report


def _odd(counts, bits, n: int) -> int:
    """The shots in which an odd number of ``bits`` read 1; for one bit, its ones."""
    return sum(v for k, v in counts.items() if sum(k[n - 1 - bit] == "1" for bit in bits) % 2)


def _describe(distribution: str, module) -> str:
    """A package's version, with the commit of its checkout when it runs from one."""
    try:
        version = metadata.version(distribution)
    except metadata.PackageNotFoundError:
        version = getattr(module, "__version__", "unknown version")
    git = ["git", "-C", str(Path(module.__file__).resolve().parent)]
    try:
        # Only when the package's own files are tracked there, not a checkout it merely sits in.
        tracked = subprocess.run(
            [*git, "ls-files", "--error-unmatch", "__init__.py"], capture_output=True, timeout=10
        )
        if tracked.returncode == 0:
            commit = subprocess.run(
                [*git, "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=10
            )
            changed = subprocess.run(
                [*git, "status", "--porcelain", "--untracked-files=no"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            version += f" at {commit.stdout.strip()}" + (
                " with local changes" if changed.stdout.strip() else ""
            )
    except (OSError, subprocess.SubprocessError):
        pass
    return version


def environment() -> dict:
    """What a timing depends on besides the case: the versions run and the threads they may use."""
    import qiskit_maestro

    # A Maestro that can, says what it was built from and from which QCSim: the revisions
    # its build saw, which is what runs here whatever a checkout holds by now.
    build = maestro.build_info() if hasattr(maestro, "build_info") else {}
    if build.get("version", "unknown") != "unknown":
        core = build["version"]
        if build.get("source_revision", "unknown") != "unknown":
            core += f" at {build['source_revision']}"
    else:
        core = _describe("qoro-maestro", maestro)
    return {
        "qiskit": metadata.version("qiskit"),
        "qiskit-aer": metadata.version("qiskit-aer"),
        "qoro-maestro": core,
        "qcsim": build.get("qcsim_revision", "unknown"),
        "qiskit-maestro": _describe("qiskit-maestro", qiskit_maestro),
        "processor": platform.processor() or platform.machine(),
        "system": platform.system(),
        "logical_cpus": os.cpu_count(),
        # Both simulators take every core unless told otherwise: Aer through this option
        # (0 means all), both through OMP_NUM_THREADS.
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "unset"),
        "aer_max_parallel_threads": AerSimulator().options.max_parallel_threads,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--methods", nargs="+", default=["density_matrix"])
    parser.add_argument("--qubits", nargs="+", type=int, default=[12])
    parser.add_argument("--noise", nargs="+", default=list(NOISE), choices=NOISE)
    parser.add_argument("--layout", default="device", choices=("device", "chain"))
    parser.add_argument("--depth", type=int, default=20)
    parser.add_argument("--shots", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-bond-dimension", type=int, default=64)
    parser.add_argument("--skip-aer", action="store_true", help="time Maestro only")
    parser.add_argument("--skip-maestro", action="store_true", help="time Aer only")
    parser.add_argument(
        "--settle", type=float, default=1.0, help="seconds to wait before timing a backend"
    )
    parser.add_argument(
        "--no-warm-up", action="store_true", help="time the first run too, for runs of minutes"
    )
    parser.add_argument("--no-check", action="store_true", help="skip the agreement check")
    parser.add_argument("--json", help="append one JSON line per case to this file")
    args = parser.parse_args()

    setup = environment()
    print(", ".join(f"{name} {value}" for name, value in setup.items()))
    header = f"{'method':22s} {'n':>3s} {'noise':6s} {'ops':>5s} {'shots':>6s} "
    print(header + f"{'Aer s':>9s} {'Maestro s':>10s} {'Aer/Maestro':>12s}  check")
    for method in args.methods:
        options = {}
        if method == "matrix_product_state":
            options["matrix_product_state_max_bond_dimension"] = args.max_bond_dimension
        for n in args.qubits:
            device, qubits = placement(n, args.layout)
            circuit = workload(device, qubits, args.depth, args.seed)
            measured = circuit.copy()
            measured.measure(qubits, range(n))
            ops = sum(measured.count_ops().values())
            for kind in args.noise:
                model = noise_model(device, kind)
                truncated = truncates(options, n)
                checked = ""
                if not (args.no_check or args.skip_aer or args.skip_maestro or truncated):
                    checked = _check(circuit, measured, qubits, model, method, options, args.shots)
                mine = theirs = workers = errors = None
                sampled = {}
                if not args.skip_maestro:
                    backend = MaestroBackend(method=method, noise_model=model, **options)
                    mine, last = _timed(
                        lambda: _run(backend, measured, args.shots),
                        args.repeats,
                        args.settle,
                        not args.no_warm_up,
                    )
                    sampled["Maestro"] = last.get_counts()
                    # The workers Maestro ran its noise realizations on, where it reports them.
                    info = getattr(last.results[0], "metadata", None) or {}
                    workers = info.get("maestro", {}).get("realization_workers")
                if not args.skip_aer:
                    aer = AerSimulator(method=method, noise_model=model, **options)
                    theirs, last = _timed(
                        lambda: _run(aer, measured, args.shots),
                        args.repeats,
                        args.settle,
                        not args.no_warm_up,
                    )
                    sampled["Aer"] = last.get_counts()
                if truncated and sampled and not args.no_check:
                    checked, errors = truncation_error(measured, model, n, args.shots, sampled)
                median = statistics.median(mine) if mine else float("nan")
                aer_median = statistics.median(theirs) if theirs else float("nan")
                print(
                    f"{method:22s} {n:3d} {kind:6s} {ops:5d} {args.shots:6d} "
                    f"{aer_median:9.4f} {median:10.4f} {aer_median / median:12.2f}  {checked}",
                    flush=True,
                )
                if args.json:
                    record = dict(
                        method=method, n=n, noise=kind, layout=args.layout, depth=args.depth,
                        seed=args.seed, ops=ops, shots=args.shots, repeats=args.repeats,
                        max_bond_dimension=options.get("matrix_product_state_max_bond_dimension"),
                        device=device.name, maestro_s=mine, aer_s=theirs,
                        maestro_realization_workers=workers, check=checked,
                        truncation_error=errors, environment=setup,
                    )  # fmt: skip
                    with open(args.json, "a", encoding="utf-8") as out:
                        out.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
