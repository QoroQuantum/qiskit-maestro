# SPDX-FileCopyrightText: 2026 Qoro Quantum
# SPDX-License-Identifier: GPL-3.0-or-later

"""Qiskit integration for the Maestro quantum circuit simulator by Qoro Quantum.

Swap ``AerSimulator`` for :class:`MaestroBackend`, ``StatevectorSampler`` for
:class:`MaestroSampler`, and ``StatevectorEstimator`` for
:class:`MaestroEstimator`. Pass ``service=QoroService()`` (from qoro-divi) to
the backend or the sampler to run on Maestro in the Qoro cloud.
"""

from importlib.metadata import PackageNotFoundError, version

from .backend import MaestroBackend, MaestroJob
from .cloud import QoroCloudJob, QoroCloudPrimitiveJob
from .estimator import MaestroEstimator
from .exceptions import (
    MaestroError,
    MaestroGpuError,
    QoroCloudError,
    UnsupportedInstructionError,
    UnsupportedNoiseError,
)
from .sampler import MaestroSampler

try:
    __version__ = version("qiskit-maestro")
except PackageNotFoundError:  # Running from a source tree that isn't installed.
    __version__ = "0.0.0"

__all__ = [
    "MaestroBackend",
    "MaestroError",
    "MaestroEstimator",
    "MaestroGpuError",
    "MaestroJob",
    "MaestroSampler",
    "QoroCloudError",
    "QoroCloudJob",
    "QoroCloudPrimitiveJob",
    "UnsupportedInstructionError",
    "UnsupportedNoiseError",
    "__version__",
]
