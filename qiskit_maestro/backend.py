# SPDX-FileCopyrightText: 2026 Qoro Quantum
# SPDX-License-Identifier: GPL-3.0-or-later

"""A Qiskit ``BackendV2`` that runs circuits on Maestro."""

from __future__ import annotations

import inspect
import sys
import time
import uuid
import warnings
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Delay, ForLoopOp, IfElseOp, Measure, Parameter, Reset, SwitchCaseOp
from qiskit.circuit.library import get_standard_gate_name_mapping
from qiskit.providers import BackendV2, JobStatus, JobV1, Options
from qiskit.result import Result
from qiskit.transpiler import Target

from . import _execution
from .bit_order import outcomes_to_hex
from .converter import NATIVE_GATES, ConvertedCircuit, convert
from .noise import NoiseTranslator
from .options import (
    DEFAULT_OPTIONS,
    OPTION_NAMES,
    normalize_options,
    parallel_workers,
    resolve_method,
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


# Operations every target has, whatever its gates.
_NON_GATES = frozenset(
    {"measure", "reset", "delay", "barrier", "if_else", "for_loop", "switch_case"}
)


def maestro_target(basis_gates: Iterable[str] | None = None) -> Target:
    """A target of fully connected qubits, any number of them.

    Args:
        basis_gates: Gate names to target, as in an Aer noise model's
            ``basis_gates``, so that transpiled circuits use the gates the noise
            is attached to. ``None`` targets Maestro's native gates. Names
            that aren't standard Qiskit gates are left out.
    """
    # num_qubits=None makes every operation global: any qubits, any width.
    target = Target(description="Maestro simulator, all-to-all connectivity", num_qubits=None)
    if basis_gates is None:
        gates = [_parametrised(gate_class) for gate_class in NATIVE_GATES]
    else:
        mapping = get_standard_gate_name_mapping()
        names = dict.fromkeys(name for name in basis_gates if name not in _NON_GATES)
        gates = [mapping[name] for name in names if name in mapping]
    for gate in gates:
        target.add_instruction(gate)
    target.add_instruction(Measure())
    target.add_instruction(Reset())
    target.add_instruction(Delay(Parameter("t")))
    target.add_instruction(IfElseOp, name="if_else")
    target.add_instruction(ForLoopOp, name="for_loop")
    target.add_instruction(SwitchCaseOp, name="switch_case")
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
            ``"mps"``), ``"stabilizer"``, ``"extended_stabilizer"``,
            ``"density_matrix"``, ``"tensor_network"``, ``"pauli_propagation"``
            or ``"automatic"``, which picks the density matrix for small noisy
            circuits as Aer does, and the statevector otherwise.
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
        noise_model: A :class:`qiskit_aer.noise.NoiseModel`. Its errors are
            applied where Aer applies them, and the target then uses its basis
            gates, as ``AerSimulator(noise_model=...)``'s does. An error Maestro
            can't reproduce exactly raises
            :class:`~qiskit_maestro.UnsupportedNoiseError`.
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
        if "noise_model" in given:
            model = given["noise_model"]
            self._target = maestro_target(None if model is None else model.basis_gates)

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
            UnsupportedNoiseError: if the noise model has an error, on an
                instruction a circuit uses, that Maestro can't reproduce exactly.
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

        translator = (
            None if options["noise_model"] is None else NoiseTranslator(options["noise_model"])
        )
        prepared = [_prepare(circuit, options, shots, translator) for circuit in circuits]
        seed = options["seed_simulator"]
        rng = np.random.default_rng(seed)
        seeds = _execution.maestro_seeds(rng, seed is not None, len(circuits))
        widest = max((c.num_qubits for c, _ in prepared), default=0)
        workers = parallel_workers(options, widest, len(circuits))
        # Noisy circuits split their shots over threads too; share the threads out.
        inner_workers = max(1, parallel_workers(options, widest, sys.maxsize) // workers)

        def one(i: int) -> tuple[_execution.Samples | None, tuple | None]:
            circuit, circuit_options = prepared[i]
            samples = saved = None
            if circuit.has_measurements:
                rows = circuit.angle_rows()
                samples = _execution.sample(
                    circuit, rows, shots, circuit_options, [seeds[i]], inner_workers
                )[0]
            if circuit.saves:
                row = circuit.angle_rows()[0]
                saved = _execution.saved_data(circuit, row, circuit_options, seeds[i], shots)
            return samples, saved

        start = time.perf_counter()
        outputs = _execution.run_parallel(one, len(circuits), workers)
        experiments = [
            _experiment(circuit, samples, saved, shots, memory, rng, prepared[i][1], seeds[i])
            for i, (circuit, (samples, saved)) in enumerate(zip(circuits, outputs))
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


def _prepare(
    circuit: QuantumCircuit,
    options: Mapping[str, Any],
    shots: int,
    translator: NoiseTranslator | None,
) -> tuple[ConvertedCircuit, dict[str, Any]]:
    """Convert a circuit, choose its method and translate its noise for that method."""
    converted = convert(circuit, noise=translator, saves=True)
    noisy = translator is not None and translator.has_quantum_errors
    method = resolve_method(options, converted, shots, noisy)
    return converted.with_noise(translator, method), {**options, "method": method}


def _experiment(
    circuit: QuantumCircuit,
    samples: _execution.Samples | None,
    saved: tuple[dict[str, Any], list[dict[str, Any]]] | None,
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
    if saved is not None:
        data.update(saved[0])
        if saved[1]:
            infos = saved[1] + ([samples.info] if samples is not None else [])
            metadata["maestro"] = run_info(infos)
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
