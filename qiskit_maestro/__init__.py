# SPDX-FileCopyrightText: 2026 Qoro Quantum
# SPDX-License-Identifier: GPL-3.0-or-later

"""Qiskit integration for the Maestro quantum circuit simulator by Qoro Quantum.

Swap ``AerSimulator`` for :class:`MaestroBackend`, ``StatevectorSampler`` for
:class:`MaestroSampler`, and ``StatevectorEstimator`` for
:class:`MaestroEstimator`.
"""

from importlib.metadata import PackageNotFoundError, version

from .backend import MaestroBackend, MaestroJob
from .estimator import MaestroEstimator
from .exceptions import (
    MaestroError,
    MaestroGpuError,
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
    "UnsupportedInstructionError",
    "UnsupportedNoiseError",
    "__version__",
]
