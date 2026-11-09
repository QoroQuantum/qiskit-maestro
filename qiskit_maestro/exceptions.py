# SPDX-FileCopyrightText: 2026 Qoro Quantum
# SPDX-License-Identifier: GPL-3.0-or-later

"""Exceptions raised by qiskit-maestro."""

from __future__ import annotations

from qiskit.exceptions import QiskitError

GPU_ACCESS_URL = "https://qoroquantum.github.io/maestro/getting-started/gpu-access.html"


class MaestroError(QiskitError):
    """Base class for errors raised by qiskit-maestro."""


class UnsupportedInstructionError(MaestroError):
    """A circuit holds an instruction Maestro can't run."""

    def __init__(self, instruction: str, reason: str):
        self.instruction = instruction
        super().__init__(f"Maestro can't run '{instruction}': {reason}")


class UnsupportedNoiseError(UnsupportedInstructionError):
    """A noise model holds an error Maestro can't reproduce exactly.

    ``instruction`` names the instruction the error is attached to, and
    ``qubits`` its qubits when the error is on particular ones.
    """

    def __init__(self, instruction: str, reason: str, qubits: tuple[int, ...] | None = None):
        self.qubits = qubits
        super().__init__(instruction, reason)


class MaestroGpuError(MaestroError):
    """``device="GPU"`` was requested but Maestro's GPU backend can't be used."""

    def __init__(self, reason: str):
        super().__init__(
            f"{reason} qiskit-maestro doesn't fall back to the CPU. To run on GPUs, "
            f"see {GPU_ACCESS_URL}, or use device='CPU'."
        )
