# SPDX-FileCopyrightText: 2026 Qoro Quantum
# SPDX-License-Identifier: GPL-3.0-or-later

"""Running circuits on Maestro in the Qoro cloud, through qoro-divi's ``QoroService``.

Pass ``service=QoroService()`` to :class:`~qiskit_maestro.MaestroBackend`,
:class:`~qiskit_maestro.MaestroSampler` or :class:`~qiskit_maestro.MaestroEstimator`
and their circuits run as one Qoro cloud job per call. The plugin still converts
and checks every circuit locally, then sends it as OpenQASM 2.0 with the
simulator options as a Maestro config; ``device="GPU"`` runs on the Qoro GPU
cluster. divi handles authentication, polling and cancelling.

Install the dependency with ``pip install "qiskit-maestro[cloud]"``.
"""

from __future__ import annotations

import threading
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import numpy as np
from qiskit.primitives import BasePrimitiveJob
from qiskit.providers import JobStatus, JobTimeoutError, JobV1
from qiskit.providers.jobstatus import JOB_FINAL_STATES

from .bit_order import qiskit_counts_to_outcomes
from .converter import QASM2_PARAMETER_PREFIX, ConvertedCircuit
from .exceptions import QoroCloudError
from .options import check_device, simulator_config

__all__ = ["QoroCloudJob", "QoroCloudPrimitiveJob"]

T = TypeVar("T")

# The Qoro cluster device="GPU" runs on.
GPU_CLUSTER = "qoro_gpu"

# divi's job statuses, by value, as Qiskit's.
_STATUS = {
    "PENDING": JobStatus.QUEUED,
    "SCHEDULED": JobStatus.QUEUED,
    "PAUSED": JobStatus.QUEUED,
    "RUNNING": JobStatus.RUNNING,
    "COMPLETED": JobStatus.DONE,
    "CANCELLED": JobStatus.CANCELLED,
    "FAILED": JobStatus.ERROR,
    "INSUFFICIENT_CREDITS": JobStatus.ERROR,
    "TIMED_OUT": JobStatus.ERROR,
}


def _divi():
    """divi's backends module, imported only when the cloud is used."""
    try:
        from divi import backends
    except ImportError as exc:
        raise ImportError(
            "Running on the Qoro cloud needs qoro-divi. Install it with "
            "pip install 'qiskit-maestro[cloud]'."
        ) from exc
    return backends


def check_service(service: Any) -> None:
    """Raise unless ``service`` is a ``QoroService`` aimed at a Maestro simulator cluster."""
    backends = _divi()
    if not isinstance(service, backends.QoroService):
        raise TypeError(
            f"service must be a divi.backends.QoroService, got {type(service).__name__}."
        )
    qpu = service.job_config.qpu_system
    if qpu is not None:
        name = getattr(qpu, "name", qpu)
        raise QoroCloudError(
            f"The service's job_config targets the QPU system {name!r}. qiskit-maestro runs "
            "on Qoro's Maestro simulator clusters: give the QoroService a job_config with a "
            "simulator_cluster."
        )


def check_options(options: Mapping[str, Any]) -> None:
    """Raise for simulator options the cloud can't honour yet."""
    if options["noise_model"] is not None:
        raise QoroCloudError(
            "noise_model isn't supported on the Qoro cloud yet. Run noisy circuits "
            "locally, without service=."
        )


def maestro_config(options: Mapping[str, Any]):
    """The divi ``MaestroConfig`` for a full set of options, with one method for the job."""
    config = simulator_config(options, options["seed_simulator"])
    return _divi().MaestroConfig.from_simulator_config(config)


def to_samples(entry: Mapping[str, Any], num_clbits: int, options: Mapping[str, Any]):
    """One circuit's cloud result entry as :class:`~qiskit_maestro._execution.Samples`.

    The entry's counts are keyed by Qiskit-order bitstrings. Its metadata is
    Maestro's result info; when it says where the circuit ran, a GPU run that
    fell back to the CPU raises as it does locally.
    """
    from ._execution import Samples

    outcomes, frequencies = qiskit_counts_to_outcomes(entry["results"], num_clbits)
    return Samples(outcomes, frequencies, entry_info(entry, options))


def to_expectations(
    entry: Mapping[str, Any], terms: Sequence[str], options: Mapping[str, Any]
) -> tuple[np.ndarray, dict[str, Any]]:
    """The values of Maestro-order Pauli ``terms`` from one circuit's cloud result entry."""
    values = entry["results"]
    return np.array([float(values[term]) for term in terms]), entry_info(entry, options)


def entry_info(entry: Mapping[str, Any], options: Mapping[str, Any]) -> dict[str, Any]:
    """A result entry's metadata, Maestro's result info.

    When it says where the circuit ran, a GPU run that fell back to the CPU
    raises as it does locally.
    """
    info = dict(entry.get("metadata") or {})
    if "simulator" in info:
        check_device(info, options)
    return info


def _groups(values: Sequence[T]) -> list[tuple[int, int, T]]:
    """``(start, end, value)`` for each run of equal consecutive values."""
    runs: list[tuple[int, int, T]] = []
    for i, value in enumerate(values):
        if runs and runs[-1][2] == value:
            runs[-1] = (runs[-1][0], i + 1, value)
        else:
            runs.append((i, i + 1, value))
    return runs


@dataclass
class Program:
    """A converted circuit and the parameter sets to run it with, one cloud circuit each.

    ``rows`` are rows of the circuit's :meth:`~qiskit_maestro.converter.ConvertedCircuit.angle_rows`
    and ``labels`` name their results. A sampling job gives each row's
    ``shots``; an expectation-value job gives each row's ``terms`` instead,
    Maestro-order Pauli strings as wide as the circuit.
    """

    circuit: ConvertedCircuit
    rows: np.ndarray
    labels: list[str]
    shots: list[int] | None = None
    terms: list[list[str]] | None = None


def _templates(programs: Sequence[Program]) -> list:
    """Each program as a divi ``CircuitPayload``: a template and its rows' parameter values."""
    from divi.circuits import CircuitPayload
    from qiskit.circuit import Parameter

    payloads = []
    for program in programs:
        template, columns = program.circuit.qasm2_template()
        payloads.append(
            CircuitPayload(
                circuit=template,
                parameters=tuple(Parameter(f"{QASM2_PARAMETER_PREFIX}{j}") for j in columns),
                parameter_sets=tuple(
                    (label, tuple(float(v) for v in row[columns]))
                    for label, row in zip(program.labels, program.rows)
                ),
            )
        )
    return payloads


def _bound(programs: Sequence[Program]) -> dict[str, str]:
    """Every row of every program as its own OpenQASM 2.0 circuit, by label."""
    return {
        label: program.circuit.to_qasm2(row)
        for program in programs
        for label, row in zip(program.labels, program.rows)
    }


class CloudRun:
    """One Qoro cloud job: submitted on creation, then polled, fetched or cancelled.

    When every circuit in the job takes the same shots, or every one the same
    terms, each program goes as one OpenQASM 2.0 template with a table of
    parameter values, which the cloud binds. Otherwise divi can't give circuits
    their own shots or terms in that form, so every parameter set goes as its
    own circuit.

    Args:
        service: The ``QoroService``.
        programs: What to run. With no rows, nothing is submitted and the run
            is already done.
        config: The divi ``MaestroConfig`` the job runs with.
        gpu: Run on the Qoro GPU cluster rather than the service's cluster.
    """

    def __init__(self, service, programs: Sequence[Program], config, *, gpu: bool = False):
        self._service = service
        self._lock = threading.Lock()
        self._execution = None
        self._local_id = str(uuid.uuid4())
        self._results: dict[str, dict[str, Any]] | None = None
        # Whether the circuits went as templates with tables of parameter values.
        self.template = False
        programs = [program for program in programs if len(program.labels)]
        if not programs:
            self._results = {}
            return
        sampling = programs[0].shots is not None
        per_circuit = [
            value
            for program in programs
            for value in (program.shots if sampling else map(tuple, program.terms))
        ]
        uniform = len(set(per_circuit)) == 1
        self.template = uniform
        job_config: dict[str, Any] = {}
        kwargs: dict[str, Any] = {"maestro_config": config}
        if sampling and uniform:
            job_config["shots"] = per_circuit[0]
        elif sampling:
            runs = _groups(per_circuit)
            kwargs["shot_groups"] = [[start, end, count] for start, end, count in runs]
        elif uniform:
            kwargs["ham_ops"] = ";".join(per_circuit[0])
        else:
            # One |-separated group of terms per run of circuits.
            runs = _groups([";".join(terms) for terms in per_circuit])
            kwargs["ham_ops"] = "|".join(group for _, _, group in runs)
            kwargs["circuit_ham_map"] = [[start, end] for start, end, _ in runs]
        if gpu:
            job_config["simulator_cluster"] = GPU_CLUSTER
        if job_config:
            kwargs["job_config"] = service.job_config.override(**job_config)
        circuits = _templates(programs) if uniform else _bound(programs)
        self._execution = service.submit_circuits(circuits, **kwargs)

    @property
    def submitted(self) -> bool:
        """Whether a job was submitted: ``False`` when there was nothing to run."""
        return self._execution is not None

    @property
    def job_id(self) -> str:
        """The cloud's job ID, or a local one when nothing was submitted."""
        return self._local_id if self._execution is None else self._execution.job_id

    def status(self) -> JobStatus:
        if self._results is not None:
            return JobStatus.DONE
        status = self._service.poll_job_status(
            self._execution, loop_until_complete=False, verbose=False
        )
        return _STATUS.get(str(getattr(status, "value", status)), JobStatus.QUEUED)

    def results(self, timeout: float | None = None) -> dict[str, dict[str, Any]]:
        """Wait for the job and return each circuit's result entry by label.

        Raises:
            JobTimeoutError: if ``timeout`` seconds pass first. The job keeps running.
            QoroCloudError: if the job fails, is cancelled, or runs out of credits.
        """
        with self._lock:
            if self._results is None:
                self._wait(timeout)
                fetched = self._service.get_job_results(self._execution)
                self._results = {entry["label"]: entry for entry in fetched.results}
            return self._results

    def _wait(self, timeout: float | None) -> None:
        from divi.exceptions import ExecutionCancelledError

        backends = _divi()
        event = timer = None
        if timeout is not None:
            # divi stops polling when the event is set. Passing our own event also
            # tells divi not to cancel the job, which should outlive a timeout.
            event = threading.Event()
            timer = threading.Timer(timeout, event.set)
            timer.daemon = True
            timer.start()
        try:
            self._service.poll_job_status(
                self._execution, loop_until_complete=True, verbose=False, cancellation_event=event
            )
        except ExecutionCancelledError as exc:
            raise JobTimeoutError(
                f"Qoro cloud job {self.job_id} didn't finish within {timeout} s. It's still "
                "running: call result() again to keep waiting."
            ) from exc
        except backends.QoroJobError as exc:
            message = str(exc)
            if isinstance(exc, backends.InsufficientCreditsError):
                message += " Add credits to your Qoro account and run it again."
            raise QoroCloudError(message, job_id=exc.job_id) from exc
        finally:
            if timer is not None:
                timer.cancel()

    def cancel(self) -> bool:
        """Ask the cloud to cancel the job. Returns whether it accepted."""
        import requests

        if self._execution is None or self._results is not None:
            return False
        try:
            self._service.cancel_job(self._execution)
        except requests.HTTPError:
            # For example 409: the job has already finished.
            return False
        return True


class QoroCloudJob(JobV1):
    """A :meth:`~qiskit_maestro.MaestroBackend.run` on the Qoro cloud.

    ``job_id()`` is the cloud's job ID. :meth:`result` waits for the job.
    """

    def __init__(self, backend, run: CloudRun, build: Callable[[Mapping[str, dict]], Any]):
        super().__init__(backend, run.job_id)
        self._run = run
        self._build = build
        self._result = None

    def submit(self):
        raise RuntimeError("A QoroCloudJob is submitted when MaestroBackend.run() is called.")

    def result(self, timeout: float | None = None):
        """Wait for the job and return its :class:`~qiskit.result.Result`.

        Raises:
            JobTimeoutError: if ``timeout`` seconds pass first. The job keeps running.
            QoroCloudError: if the job fails, is cancelled, or runs out of credits.
        """
        if self._result is None:
            self._result = self._build(self._run.results(timeout))
        return self._result

    def status(self) -> JobStatus:
        return self._run.status()

    def cancel(self) -> bool:
        return self._run.cancel()


class QoroCloudPrimitiveJob(BasePrimitiveJob, Generic[T]):
    """A primitive's run on the Qoro cloud. ``job_id()`` is the cloud's job ID."""

    def __init__(self, run: CloudRun, build: Callable[[Mapping[str, dict]], T]):
        super().__init__(run.job_id)
        self._run = run
        self._build = build
        self._result: T | None = None

    def result(self) -> T:
        """Wait for the job and return its result.

        Raises:
            QoroCloudError: if the job fails, is cancelled, or runs out of credits.
        """
        if self._result is None:
            self._result = self._build(self._run.results())
        return self._result

    def status(self) -> JobStatus:
        return self._run.status()

    def done(self) -> bool:
        return self.status() == JobStatus.DONE

    def running(self) -> bool:
        return self.status() == JobStatus.RUNNING

    def cancelled(self) -> bool:
        return self.status() == JobStatus.CANCELLED

    def in_final_state(self) -> bool:
        return self.status() in JOB_FINAL_STATES

    def cancel(self) -> bool:
        return self._run.cancel()
