# SPDX-FileCopyrightText: 2026 Qoro Quantum
# SPDX-License-Identifier: GPL-3.0-or-later

"""device="GPU": clear errors without a GPU, and results that match the CPU with one.

Tests marked ``gpu`` are skipped when Maestro's GPU backend isn't available.
"""

from __future__ import annotations

import maestro
import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from qiskit_maestro import MaestroBackend, MaestroEstimator, MaestroGpuError, MaestroSampler
from qiskit_maestro.exceptions import GPU_ACCESS_URL
from qiskit_maestro.options import check_device, resolve_options

from .conftest import assert_same_distribution, ghz

no_gpu = pytest.mark.skipif(maestro.is_gpu_available(), reason="a GPU is available")


@no_gpu
@pytest.mark.parametrize("make", [MaestroBackend, MaestroSampler, MaestroEstimator])
def test_unavailable_gpu_raises_with_the_access_link(make):
    with pytest.raises(MaestroGpuError) as error:
        make(device="GPU")
    assert GPU_ACCESS_URL in str(error.value)
    assert "doesn't fall back to the CPU" in str(error.value)


@no_gpu
def test_unavailable_gpu_raises_when_set_later():
    backend = MaestroBackend()
    with pytest.raises(MaestroGpuError):
        backend.set_options(device="GPU")
    assert backend.options.device == "CPU"


@no_gpu
def test_unavailable_gpu_raises_as_a_run_option():
    circuit = QuantumCircuit(1, 1)
    circuit.measure(0, 0)
    with pytest.raises(MaestroGpuError):
        MaestroBackend().run(circuit, device="gpu")


@pytest.mark.parametrize("method", ["stabilizer", "density_matrix"])
def test_methods_without_a_gpu_implementation(method):
    with pytest.raises(MaestroGpuError, match=f"method='{method}' doesn't run on Maestro's GPU"):
        MaestroBackend(device="GPU", method=method)


def test_a_cpu_fallback_by_the_core_is_caught():
    # The core reports where it ran; anything but a GPU simulator is an error.
    options = resolve_options({})
    options["device"] = "GPU"
    cpu = maestro.SimulatorType.QCSim.value
    with pytest.raises(MaestroGpuError, match="ran the circuit on QCSim"):
        check_device({"simulator": cpu}, options)
    check_device({"simulator": maestro.SimulatorType.Gpu.value}, options)


@pytest.mark.gpu
@pytest.mark.parametrize("method", ["statevector", "matrix_product_state"])
def test_gpu_sampling_matches_cpu(method):
    circuit = ghz(20, measure=True)
    circuit.ry(0.3, 5)
    gpu = MaestroSampler(device="GPU", method=method, seed=1).run([circuit], shots=20_000)
    cpu = MaestroSampler(method=method, seed=1).run([circuit], shots=20_000)
    assert_same_distribution(
        gpu.result()[0].data.meas.get_counts(), cpu.result()[0].data.meas.get_counts()
    )
    assert gpu.result()[0].metadata["maestro"]["simulators"] == ["Gpu"]


@pytest.mark.gpu
@pytest.mark.parametrize("method", ["statevector", "matrix_product_state", "tensor_network"])
def test_gpu_estimates_match_cpu(method):
    circuit = ghz(16, measure=False)
    circuit.rx(0.4, range(16))
    observables = [SparsePauliOp("Z" * 16), SparsePauliOp("X" * 16)]
    gpu = MaestroEstimator(device="GPU", method=method).run([(circuit, observables)]).result()[0]
    cpu = MaestroEstimator(method=method).run([(circuit, observables)]).result()[0]
    np.testing.assert_allclose(gpu.data.evs, cpu.data.evs, atol=1e-8)


@pytest.mark.gpu
def test_gpu_backend_run():
    result = MaestroBackend(device="GPU").run(ghz(10, measure=True), shots=100).result()
    assert set(result.get_counts()) <= {"0" * 10, "1" * 10}
    assert result.results[0].metadata["maestro"]["simulators"] == ["Gpu"]
