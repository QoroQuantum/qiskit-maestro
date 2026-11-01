# SPDX-FileCopyrightText: 2026 Qoro Quantum
# SPDX-License-Identifier: GPL-3.0-or-later

"""A Qiskit ``BackendV2`` that runs circuits on Maestro."""

from __future__ import annotations

import inspect
import time
import uuid
import warnings
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Delay, IfElseOp, Measure, Parameter, Reset
from qiskit.providers import BackendV2, JobStatus, JobV1, Options
from qiskit.result import Result
from qiskit.transpiler import Target

from . import _execution
from .bit_order import outcomes_to_hex
from .converter import NATIVE_GATES, convert
from .options import (
    DEFAULT_OPTIONS,
    OPTION_NAMES,
    normalize_options,
    parallel_workers,
    resolve_options,
    run_info,
)

__all__ = ["MaestroBackend", "MaestroJob", "maestro_target"]

_RUN_DEFAULTS = {"shots": 1024, "memory": False}


def _parametrised(gate_class):
    """An instance of ``gate_class`` whose angles are free parameters."""
    angles = [
        p.name
        for p in inspect.signature(gate_class).parameters.values()
        if p.default is inspect.Parameter.empty and p.kind is p.POSITIONAL_OR_KEYWORD
    ]
    return gate_class(*(Parameter(name) for name in angles))


def maestro_target() -> Target:
    """Maestro's native operations on any number of fully connected qubits."""
    # num_qubits=None makes every operation global: any qubits, any width.
    target = Target(description="Maestro simulator, all-to-all connectivity", num_qubits=None)
    for gate_class in NATIVE_GATES:
        target.add_instruction(_parametrised(gate_class))
    target.add_instruction(Measure())
    target.add_instruction(Reset())
    target.add_instruction(Delay(Parameter("t")))
    target.add_instruction(IfElseOp, name="if_else")
    return target


class MaestroJob(JobV1):
    """A finished Maestro run.

    :meth:`MaestroBackend.run` simulates before it returns, as Qiskit's
    ``BasicSimulator`` does, so the job is already done when you get it.
    """

    def __init__(self, backend: MaestroBackend, job_id: str, result: Result):
        super().__init__(backend, job_id)
        self._result = result

    def submit(self):
        raise RuntimeError("A MaestroJob runs when MaestroBackend.run() is called.")

    def result(self, timeout: float | None = None) -> Result:
        return self._result

    def status(self) -> JobStatus:
        return JobStatus.DONE


class MaestroBackend(BackendV2):
    """Run Qiskit circuits on the Maestro simulator.

    A drop-in for ``AerSimulator``. Options use Aer's names where the concept is
    the same, and Aer's longer MPS option names are accepted as aliases.

    Args:
        method: ``"statevector"`` (default), ``"matrix_product_state"`` (or
            ``"mps"``), ``"stabilizer"``, ``"density_matrix"``,
            ``"tensor_network"`` or ``"pauli_propagation"``.
        device: ``"CPU"`` (default) or ``"GPU"``. A GPU run needs Maestro's GPU
            plugin and licence, and raises rather than falling back to the CPU.
        precision: ``"double"`` (default) or ``"single"``.
        seed_simulator: Seed for reproducible sampling.
        max_bond_dimension: Cap on the MPS bond dimension. Aer's name,
            ``matrix_product_state_max_bond_dimension``, also works.
        truncation_threshold: MPS truncation threshold, as discarded weight.
            Aer's name, ``matrix_product_state_truncation_threshold``, also works.
        max_parallel_experiments: Threads for running circuits side by side.
            ``0`` (default) picks a number, ``1`` runs them one at a time.
        maestro_options: Other :class:`maestro.SimulatorConfig` fields, passed
            through unchanged.
        shots: Default shots for :meth:`run` (1024).
        memory: Whether :meth:`run` returns per-shot results by default
            (``False``).
    """

    def __init__(self, *, name: str = "maestro_simulator", **options):
        from . import __version__

        super().__init__(
            name=name,
            description="Maestro quantum circuit simulator by Qoro Quantum",
            backend_version=__version__,
        )
        self._target = maestro_target()
        self.set_options(**options)

    @classmethod
    def _default_options(cls) -> Options:
        return Options(**_RUN_DEFAULTS, **DEFAULT_OPTIONS)

    @property
    def target(self) -> Target:
        return self._target

    @property
    def max_circuits(self) -> None:
        return None

    def set_options(self, **fields):
        run_only = {k: fields.pop(k) for k in list(fields) if k in _RUN_DEFAULTS}
        given = normalize_options(fields)
        resolve_options(self.simulator_options(), given)
        super().set_options(**run_only, **given)

    def simulator_options(self) -> dict[str, Any]:
        """The current simulator options, in the form the primitives take them."""
        return {name: getattr(self.options, name) for name in DEFAULT_OPTIONS}

    def run(
        self, run_input: QuantumCircuit | Sequence[QuantumCircuit], **run_options
    ) -> MaestroJob:
        """Sample circuits and return a finished job.

        Args:
            run_input: A circuit or a list of circuits, with every parameter bound.
            run_options: ``shots``, ``memory`` or any simulator option, for this
                run only.

        Raises:
            UnsupportedInstructionError: if a circuit holds an instruction Maestro
                can't run.
            MaestroGpuError: if ``device="GPU"`` and the GPU can't be used.
        """
        circuits = [run_input] if isinstance(run_input, QuantumCircuit) else list(run_input)
        shots = run_options.pop("shots", self.options.shots)
        memory = bool(run_options.pop("memory", self.options.memory))
        for unused in [name for name in run_options if name not in OPTION_NAMES]:
            warnings.warn(f"Option {unused!r} isn't used by {self.name}.", stacklevel=2)
            del run_options[unused]
        options = resolve_options(self.simulator_options(), run_options)
        if isinstance(shots, bool) or not isinstance(shots, (int, np.integer)) or shots < 1:
            raise ValueError(f"shots must be a positive integer, got {shots!r}.")
        shots = int(shots)

        converted = [convert(circuit) for circuit in circuits]
        seed = options["seed_simulator"]
        rng = np.random.default_rng(seed)
        seeds = _execution.maestro_seeds(rng, seed is not None, len(circuits))

        def one(i: int) -> _execution.Samples | None:
            if not converted[i].has_measurements:
                return None
            rows = converted[i].angle_rows()
            return _execution.sample(converted[i], rows, shots, options, [seeds[i]])[0]

        start = time.perf_counter()
        widest = max((c.num_qubits for c in converted), default=0)
        samples = _execution.run_parallel(
            one, len(circuits), parallel_workers(options, widest, len(circuits))
        )
        experiments = [
            _experiment(circuit, result, shots, memory, rng, options, seeds[i])
            for i, (circuit, result) in enumerate(zip(circuits, samples))
        ]
        job_id = str(uuid.uuid4())
        result = Result.from_dict(
            {
                "backend_name": self.name,
                "backend_version": self.backend_version,
                "qobj_id": job_id,
                "job_id": job_id,
                "success": True,
                "results": experiments,
                "time_taken": time.perf_counter() - start,
            }
        )
        return MaestroJob(self, job_id, result)


def _experiment(
    circuit: QuantumCircuit,
    samples: _execution.Samples | None,
    shots: int,
    memory: bool,
    rng: np.random.Generator,
    options: Mapping[str, Any],
    seed: int | None,
) -> dict[str, Any]:
    """One experiment of a Result, in the layout Aer uses."""
    phase = circuit.global_phase
    header = {
        "name": circuit.name,
        "n_qubits": circuit.num_qubits,
        "memory_slots": circuit.num_clbits,
        "creg_sizes": [[creg.name, creg.size] for creg in circuit.cregs],
        "clbit_labels": [[creg.name, j] for creg in circuit.cregs for j in range(creg.size)],
        "qreg_sizes": [[qreg.name, qreg.size] for qreg in circuit.qregs],
        "global_phase": 0.0 if getattr(phase, "parameters", None) else float(phase),
        "metadata": circuit.metadata,
    }
    data: dict[str, Any] = {}
    metadata: dict[str, Any] = {"method": options["method"], "device": options["device"]}
    time_taken = 0.0
    if samples is not None:
        # Like Aer, a circuit without measurements has no counts.
        data["counts"] = dict(zip(outcomes_to_hex(samples.outcomes), samples.frequencies.tolist()))
        if memory:
            data["memory"] = outcomes_to_hex(samples.shots(rng))
        metadata["maestro"] = run_info([samples.info])
        time_taken = samples.info.get("time_taken", 0.0)
    return {
        "shots": shots,
        "success": True,
        "data": data,
        "meas_level": 2,
        "header": header,
        "status": "DONE",
        "seed_simulator": seed,
        "time_taken": time_taken,
        "metadata": metadata,
    }
