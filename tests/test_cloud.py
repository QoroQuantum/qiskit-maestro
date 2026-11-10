# SPDX-FileCopyrightText: 2026 Qoro Quantum
# SPDX-License-Identifier: GPL-3.0-or-later

"""Running on the Qoro cloud through divi's QoroService.

The tests use a QoroService that runs each job on divi's local MaestroSimulator
instead of the network. That covers everything up to the wire: the OpenQASM 2
the plugin writes passes divi's validator and is read by Maestro, the config
and terms are what divi takes, and the results come back in divi's format.
"""

from __future__ import annotations

import re
import threading

import maestro
import numpy as np
import pytest
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter
from qiskit.providers import JobStatus, JobTimeoutError
from qiskit.quantum_info import SparsePauliOp

from qiskit_maestro import (
    MaestroBackend,
    MaestroEstimator,
    MaestroGpuError,
    MaestroSampler,
    QoroCloudError,
    QoroCloudJob,
    QoroCloudPrimitiveJob,
    UnsupportedInstructionError,
)

from .conftest import assert_same_distribution, requires_aer, seeded_random_circuits

backends = pytest.importorskip("divi.backends")
requests = pytest.importorskip("requests")

from divi.exceptions import ExecutionCancelledError
from divi.qasm import validate_qasm

_FAILURES = {
    backends.JobStatus.FAILED: backends.JobFailedError,
    backends.JobStatus.CANCELLED: backends.JobCancelledError,
    backends.JobStatus.INSUFFICIENT_CREDITS: backends.InsufficientCreditsError,
    backends.JobStatus.TIMED_OUT: backends.JobTimedOutError,
}


class FakeQoroService(backends.QoroService):
    """A QoroService that runs each job on divi's local MaestroSimulator, with no network.

    Args:
        job_config: The service's job config.
        final_status: The status the job ends in.
        hang: The job never finishes: waiting for it lasts until cancelled.
        refuse_cancel: ``cancel_job`` fails as the API does for a finished job.
        metadata: Keep the simulator's per-circuit metadata. The real cloud's
            results have none.
    """

    def __init__(
        self,
        job_config=None,
        *,
        final_status: str = "COMPLETED",
        hang: bool = False,
        refuse_cancel: bool = False,
        metadata: bool = False,
    ):
        # QoroService.__init__ authenticates and fetches the targets, so it's skipped.
        self._job_config = job_config or backends.JobConfig()
        self._maestro_config = None
        self._device_config = None
        self.final_status = backends.JobStatus(final_status)
        self.hang = hang
        self.refuse_cancel = refuse_cancel
        self.metadata = metadata
        self.submissions: list[dict] = []
        self.cancelled: list[str] = []
        self._results: dict[str, list[dict]] = {}

    def submit_circuits(self, circuits, *, maestro_config, job_config=None, **kwargs):
        templates = not isinstance(circuits, dict)
        if templates:
            payloads, circuits = circuits, _bind_templates(circuits)
        else:
            for qasm in circuits.values():
                validate_qasm(qasm)
        job_config = job_config or self.job_config
        self.submissions.append(
            {
                "circuits": dict(circuits),
                "templates": len(payloads) if templates else 0,
                "maestro_config": maestro_config,
                "job_config": job_config,
            }
            | kwargs
        )
        simulator = backends.MaestroSimulator(shots=job_config.shots, maestro_config=maestro_config)
        result = simulator.submit_circuits(dict(circuits), **kwargs)
        job_id = f"job-{len(self.submissions)}"
        entries = result.results
        if not self.metadata:
            entries = [{"label": e["label"], "results": e["results"]} for e in entries]
        self._results[job_id] = entries
        return backends.ExecutionResult(job_id=job_id)

    def poll_job_status(
        self,
        execution_result,
        loop_until_complete=False,
        verbose=True,
        progress_callback=None,
        cancellation_event=None,
    ):
        if self.hang:
            if not loop_until_complete:
                return backends.JobStatus.RUNNING
            cancellation_event.wait()
            raise ExecutionCancelledError("Polling cancelled.")
        if loop_until_complete and self.final_status in _FAILURES:
            raise _FAILURES[self.final_status](execution_result.job_id)
        return self.final_status

    def get_job_results(self, execution_result):
        return execution_result.with_results(self._results[execution_result.job_id])

    def cancel_job(self, execution_result):
        if self.refuse_cancel:
            raise requests.HTTPError("409 Conflict: the job can't be cancelled.")
        self.cancelled.append(execution_result.job_id)


def _bind_templates(payloads) -> dict[str, str]:
    """Bind divi CircuitPayloads as the Qoro cloud does: each name becomes ``str(value)``."""
    bound = {}
    for payload in payloads:
        names = payload.parameter_names
        validate_qasm(payload.circuit, names)
        for label, values in payload.parameter_sets:
            text = payload.circuit
            for name, value in zip(names, values):
                text = re.sub(rf"\b{name}\b", str(value), text)
            bound[label] = text
    return bound


def _counts(bit_array) -> dict[str, int]:
    return bit_array.get_counts()


def _two_registers() -> QuantumCircuit:
    a, b = QuantumRegister(3, "a"), ClassicalRegister(2, "low")
    high = ClassicalRegister(3, "high")
    circuit = QuantumCircuit(a, b, high)
    circuit.x(0)
    circuit.x(2)
    circuit.measure(0, b[1])
    circuit.measure(1, b[0])
    circuit.measure(2, high[2])
    circuit.measure(0, high[0])
    return circuit


# --- Sampler ---------------------------------------------------------------


def test_sampler_bit_order_matches_local():
    circuit = _two_registers()
    local = MaestroSampler().run([circuit], shots=10).result()[0].data
    service = FakeQoroService()
    job = MaestroSampler(service=service).run([circuit], shots=10)
    assert isinstance(job, QoroCloudPrimitiveJob)
    remote = job.result()[0].data
    assert _counts(remote.low) == _counts(local.low) == {"10": 10}
    assert _counts(remote.high) == _counts(local.high) == {"101": 10}


def test_sampler_matches_local_distributions():
    circuits = seeded_random_circuits(4, 4, 5, measure=True, seed=11)
    local = MaestroSampler(seed=1).run(circuits, shots=4000).result()
    remote = MaestroSampler(seed=1, service=FakeQoroService()).run(circuits, shots=4000).result()
    for mine, theirs in zip(remote, local):
        assert_same_distribution(mine.join_data().get_counts(), theirs.join_data().get_counts())


def test_sampler_parameter_sets_keep_their_shape_and_values():
    a = Parameter("a")
    circuit = QuantumCircuit(1)
    circuit.rx(a, 0)
    circuit.measure_all()
    values = np.array([0.0, np.pi]).reshape(2, 1)
    service = FakeQoroService()
    result = MaestroSampler(service=service).run([(circuit, values)], shots=50).result()[0]
    assert result.data.meas.shape == (2,)
    assert _counts(result.data.meas[0]) == {"0": 50}
    assert _counts(result.data.meas[1]) == {"1": 50}
    assert len(service.submissions) == 1
    assert len(service.submissions[0]["circuits"]) == 2
    # One template with a row of values per parameter set.
    assert service.submissions[0]["templates"] == 1


def test_templates_bind_derived_angles():
    a, b = Parameter("a"), Parameter("b")
    circuit = QuantumCircuit(3)
    circuit.h([0, 1, 2])
    circuit.ry(a, 0)
    circuit.rz(2 * b - a, 1)
    circuit.cu(a, b, 0.3, a + b, 0, 2)
    circuit.cp(b, 1, 2)
    circuit.h([0, 1, 2])
    values = np.array([[0.1, -0.7], [1.3, 2.2], [-2.5, 0.4]])
    observables = [SparsePauliOp(["ZII", "IZI", "IIZ", "XXI", "YZX"], [1, 2, 3, 4, 5])]
    pub = (circuit, observables, values)
    service = FakeQoroService()
    remote = MaestroEstimator(service=service).run([pub]).result()[0].data.evs
    local = MaestroEstimator().run([pub]).result()[0].data.evs
    np.testing.assert_allclose(remote, local, atol=1e-9)
    assert service.submissions[0]["templates"] == 1
    assert "|" not in service.submissions[0]["ham_ops"]


def test_varying_terms_send_bound_circuits():
    a = Parameter("a")
    circuit = QuantumCircuit(2)
    circuit.ry(a, 0)
    circuit.cx(0, 1)
    # A different observable for each parameter set.
    pub = (circuit, [SparsePauliOp("ZZ"), SparsePauliOp("XI")], np.array([[0.4], [1.1]]))
    service = FakeQoroService()
    remote = MaestroEstimator(service=service).run([pub]).result()[0].data.evs
    np.testing.assert_allclose(remote, MaestroEstimator().run([pub]).result()[0].data.evs)
    assert service.submissions[0]["templates"] == 0
    assert service.submissions[0]["circuit_ham_map"] == [[0, 1], [1, 2]]


def test_sampler_pubs_with_different_shots_use_shot_groups():
    circuit = QuantumCircuit(1)
    circuit.x(0)
    circuit.measure_all()
    service = FakeQoroService()
    result = (
        MaestroSampler(service=service).run([(circuit, None, 30), (circuit, None, 70)]).result()
    )
    assert [r.data.meas.num_shots for r in result] == [30, 70]
    assert service.submissions[0]["shot_groups"] == [[0, 1, 30], [1, 2, 70]]
    # divi takes per-circuit shots only with bound circuits.
    assert service.submissions[0]["templates"] == 0


def test_sampler_job_reports_the_cloud_job():
    service = FakeQoroService()
    circuit = QuantumCircuit(1)
    circuit.measure_all()
    job = MaestroSampler(service=service).run([circuit])
    assert job.job_id() == "job-1"
    assert job.status() == JobStatus.DONE
    assert job.done() and job.in_final_state()
    metadata = job.result()[0].metadata
    assert metadata["qoro_job_id"] == "job-1"
    # The cloud doesn't say which simulator and method ran.
    assert "maestro" not in metadata


def test_sampler_without_measurements_submits_nothing():
    service = FakeQoroService()
    circuit = QuantumCircuit(2, 2)
    circuit.h(0)
    result = MaestroSampler(service=service).run([circuit], shots=5).result()[0]
    assert service.submissions == []
    assert result.data.c.get_counts() == {"00": 5}
    assert "qoro_job_id" not in result.metadata


def test_sampler_options_reach_the_config():
    service = FakeQoroService()
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure_all()
    MaestroSampler(service=service, method="mps", max_bond_dimension=8, seed=5).run(
        [circuit]
    ).result()
    config = service.submissions[0]["maestro_config"]
    assert config.simulation_type == maestro.SimulationType.MatrixProductState
    assert config.max_bond_dimension == 8
    assert config.seed == 5
    assert service.submissions[0]["job_config"].shots == 1024
    # The CPU runs on the service's own cluster.
    assert service.submissions[0]["job_config"].simulator_cluster is None


@pytest.mark.parametrize("cls", [MaestroBackend, MaestroSampler, MaestroEstimator])
def test_gpu_runs_on_the_gpu_cluster(cls):
    service = FakeQoroService()
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    if cls is MaestroEstimator:
        job = cls(service=service, device="GPU").run([(circuit, SparsePauliOp("ZZ"))])
    else:
        circuit.measure_all()
        job = cls(service=service, device="GPU").run(
            circuit if cls is MaestroBackend else [circuit]
        )
    submission = service.submissions[0]
    assert submission["job_config"].simulator_cluster == "qoro_gpu"
    assert submission["maestro_config"].simulator_type == maestro.SimulatorType.Gpu
    assert job.job_id() == "job-1"


def test_sampler_rejects_classical_control():
    circuit = QuantumCircuit(2, 1)
    circuit.h(0)
    circuit.measure(0, 0)
    with circuit.if_test((circuit.clbits[0], 1)):
        circuit.x(1)
    with pytest.raises(UnsupportedInstructionError, match="Qoro cloud"):
        MaestroSampler(service=FakeQoroService()).run([circuit])


# --- Backend ---------------------------------------------------------------


def test_backend_counts_and_memory_match_local():
    circuit = _two_registers()
    local = MaestroBackend().run(circuit, shots=20, memory=True).result()
    service = FakeQoroService()
    job = MaestroBackend(service=service).run(circuit, shots=20, memory=True)
    assert isinstance(job, QoroCloudJob)
    assert job.job_id() == "job-1"
    assert job.status() == JobStatus.DONE
    remote = job.result()
    assert remote.get_counts() == local.get_counts()
    assert remote.get_memory() == local.get_memory()
    assert remote.job_id == "job-1"


def test_backend_matches_local_distributions():
    circuits = seeded_random_circuits(3, 4, 5, measure=True, seed=40)
    local = MaestroBackend().run(circuits, shots=4000).result()
    remote = MaestroBackend(service=FakeQoroService()).run(circuits, shots=4000).result()
    for i in range(len(circuits)):
        assert_same_distribution(remote.get_counts(i), local.get_counts(i))


def test_backend_unmeasured_circuits_have_no_counts():
    measured = QuantumCircuit(1, 1)
    measured.x(0)
    measured.measure(0, 0)
    service = FakeQoroService()
    result = MaestroBackend(service=service).run([QuantumCircuit(1), measured], shots=8).result()
    assert "counts" not in result.data(0)
    assert result.get_counts(1) == {"1": 8}
    assert list(service.submissions[0]["circuits"]) == ["1"]


def test_backend_rejects_save_instructions():
    pytest.importorskip("qiskit_aer")
    from qiskit_aer.library import save_statevector  # noqa: F401

    circuit = QuantumCircuit(1)
    circuit.h(0)
    circuit.save_statevector()
    with pytest.raises(UnsupportedInstructionError, match="save instructions"):
        MaestroBackend(service=FakeQoroService()).run(circuit)


def test_sampler_from_backend_keeps_the_service():
    service = FakeQoroService()
    backend = MaestroBackend(service=service, method="mps")
    assert MaestroSampler.from_backend(backend).service is service
    assert MaestroEstimator.from_backend(backend).service is service


# --- Estimator -------------------------------------------------------------


def _observables(num_qubits: int, seed: int) -> list[SparsePauliOp]:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(3):
        labels = ["".join(rng.choice(list("IXYZ"), num_qubits)) for _ in range(4)]
        out.append(SparsePauliOp(labels, rng.normal(size=4)))
    return out


def test_estimator_matches_local():
    circuits = seeded_random_circuits(3, 4, 5, measure=False, seed=70)
    pubs = [(c, _observables(4, i)) for i, c in enumerate(circuits)]
    local = MaestroEstimator().run(pubs).result()
    service = FakeQoroService()
    job = MaestroEstimator(service=service).run(pubs)
    assert isinstance(job, QoroCloudPrimitiveJob)
    remote = job.result()
    for mine, theirs in zip(remote, local):
        np.testing.assert_allclose(mine.data.evs, theirs.data.evs, atol=1e-9)
    assert len(service.submissions) == 1
    assert service.submissions[0]["ham_ops"].count("|") == 2


def test_estimator_broadcasts_parameters_over_observables():
    a = Parameter("a")
    circuit = QuantumCircuit(2)
    circuit.ry(a, 0)
    circuit.cx(0, 1)
    values = np.linspace(0, np.pi, 3).reshape(3, 1)
    observables = [[SparsePauliOp("ZZ")], [SparsePauliOp("XI")], [SparsePauliOp(["IZ", "II"])]]
    pub = (circuit, observables, values)
    local = MaestroEstimator().run([pub]).result()[0].data.evs
    remote = MaestroEstimator(service=FakeQoroService()).run([pub]).result()[0].data.evs
    assert remote.shape == local.shape == (3, 3)
    np.testing.assert_allclose(remote, local, atol=1e-9)


def test_estimator_identity_needs_no_job():
    service = FakeQoroService()
    circuit = QuantumCircuit(2)
    circuit.h(0)
    result = MaestroEstimator(service=service).run([(circuit, SparsePauliOp("II", 2.5))]).result()
    assert service.submissions == []
    np.testing.assert_allclose(result[0].data.evs, 2.5)


def test_estimator_precision_adds_noise_locally():
    circuit = QuantumCircuit(1)
    pub = (circuit, SparsePauliOp("Z"))
    local = MaestroEstimator(seed=3).run([pub], precision=0.1).result()[0].data
    remote = MaestroEstimator(seed=3, service=FakeQoroService()).run([pub], precision=0.1)
    remote = remote.result()[0].data
    np.testing.assert_allclose(remote.evs, local.evs)
    np.testing.assert_allclose(remote.stds, 0.1)


def test_estimator_rejects_resets():
    circuit = QuantumCircuit(1)
    circuit.h(0)
    circuit.reset(0)
    with pytest.raises(UnsupportedInstructionError, match="resets"):
        MaestroEstimator(service=FakeQoroService()).run([(circuit, SparsePauliOp("Z"))])


def test_estimator_run_evolution_isnt_available():
    step = QuantumCircuit(1)
    step.rx(0.1, 0)
    with pytest.raises(QoroCloudError, match="run_evolution"):
        MaestroEstimator(service=FakeQoroService()).run_evolution(
            QuantumCircuit(1), step, SparsePauliOp("Z"), [0, 1]
        )


# --- Configuration errors --------------------------------------------------


@requires_aer
@pytest.mark.parametrize("cls", [MaestroBackend, MaestroSampler, MaestroEstimator])
def test_noise_models_are_rejected(cls):
    from qiskit_aer.noise import NoiseModel, depolarizing_error

    model = NoiseModel()
    model.add_all_qubit_quantum_error(depolarizing_error(0.1, 1), ["x"])
    with pytest.raises(QoroCloudError, match="noise_model"):
        cls(service=FakeQoroService(), noise_model=model)


@requires_aer
def test_a_noise_model_run_option_is_rejected():
    from qiskit_aer.noise import NoiseModel, depolarizing_error

    model = NoiseModel()
    model.add_all_qubit_quantum_error(depolarizing_error(0.1, 1), ["x"])
    circuit = QuantumCircuit(1)
    circuit.measure_all()
    with pytest.raises(QoroCloudError, match="noise_model"):
        MaestroBackend(service=FakeQoroService()).run(circuit, noise_model=model)


def test_a_qpu_target_is_rejected():
    service = FakeQoroService(backends.JobConfig(qpu_system="ibm_torino"))
    with pytest.raises(QoroCloudError, match="ibm_torino"):
        MaestroSampler(service=service)


def test_service_must_be_a_qoro_service():
    with pytest.raises(TypeError, match="QoroService"):
        MaestroBackend(service=object())


@pytest.mark.skipif(maestro.is_gpu_available(), reason="needs a machine without a GPU")
def test_gpu_isnt_checked_locally_but_a_cpu_fallback_raises():
    circuit = QuantumCircuit(1)
    circuit.measure_all()
    service = FakeQoroService(metadata=True)
    # No local GPU is needed to submit...
    job = MaestroSampler(service=service, device="GPU").run([circuit])
    assert service.submissions[0]["maestro_config"].simulator_type == maestro.SimulatorType.Gpu
    # ...but the stand-in cloud has none either, so the job ran on the CPU.
    with pytest.raises(MaestroGpuError, match="the CPU"):
        job.result()


# --- Job outcomes ----------------------------------------------------------


def _measured() -> QuantumCircuit:
    circuit = QuantumCircuit(1)
    circuit.measure_all()
    return circuit


@pytest.mark.parametrize(
    "status, match",
    [
        ("FAILED", "failed"),
        ("TIMED_OUT", "timed out"),
        ("INSUFFICIENT_CREDITS", "Add credits"),
    ],
)
def test_failed_jobs_raise_with_their_id(status, match):
    service = FakeQoroService(final_status=status)
    job = MaestroBackend(service=service).run(_measured())
    assert job.status() == JobStatus.ERROR
    with pytest.raises(QoroCloudError, match=match) as raised:
        job.result()
    assert raised.value.job_id == "job-1"


def test_timeout_leaves_the_job_running():
    service = FakeQoroService(hang=True)
    job = MaestroBackend(service=service).run(_measured())
    assert job.status() == JobStatus.RUNNING
    with pytest.raises(JobTimeoutError, match="still running"):
        job.result(timeout=0.05)
    assert service.cancelled == []


def test_cancel():
    service = FakeQoroService(hang=True)
    job = MaestroSampler(service=service).run([_measured()])
    assert job.cancel()
    assert service.cancelled == ["job-1"]
    assert (
        not MaestroSampler(service=FakeQoroService(refuse_cancel=True)).run([_measured()]).cancel()
    )


def test_concurrent_waits_fetch_once():
    service = FakeQoroService()
    calls = []
    fetch = service.get_job_results
    service.get_job_results = lambda er: calls.append(er) or fetch(er)
    job = MaestroBackend(service=service).run(_measured())
    threads = [threading.Thread(target=job.result) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(calls) == 1
