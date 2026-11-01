# SPDX-FileCopyrightText: 2026 Qoro Quantum
# SPDX-License-Identifier: GPL-3.0-or-later

"""A native Qiskit ``SamplerV2`` on Maestro."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
from qiskit.primitives import BaseSamplerV2, BitArray, DataBin, PrimitiveResult, SamplerPubResult
from qiskit.primitives.containers.sampler_pub import SamplerPub, SamplerPubLike
from qiskit.primitives.primitive_job import PrimitiveJob

from . import _execution
from .bit_order import pack_register
from .converter import convert
from .options import merge_aer_options, resolve_options, run_info, validate_combination

__all__ = ["MaestroSampler"]


class MaestroSampler(BaseSamplerV2):
    """Sample circuits on Maestro, a drop-in for ``StatevectorSampler`` and Aer's ``SamplerV2``.

    Each PUB's circuit is converted once; its parameter sets then run in
    parallel. Results have one :class:`~qiskit.primitives.BitArray` per
    classical register, shaped as :class:`~qiskit.primitives.StatevectorSampler`
    shapes them, in Qiskit's bit order.

    Args:
        default_shots: Shots for PUBs that don't set their own.
        seed: Seed for reproducible results.
        options: Aer's form, ``{"backend_options": {...}, "run_options": {...}}``,
            for code moving from ``qiskit_aer.primitives.SamplerV2``.
        **simulator_options: Any :class:`~qiskit_maestro.MaestroBackend` option,
            such as ``method="matrix_product_state"`` or ``device="GPU"``.
    """

    def __init__(
        self,
        *,
        default_shots: int = 1024,
        seed: int | None = None,
        options: Mapping[str, Any] | None = None,
        **simulator_options,
    ):
        simulator_options = merge_aer_options(options, simulator_options)
        if seed is not None:
            simulator_options["seed_simulator"] = seed
        self._options = resolve_options(simulator_options)
        self._default_shots = default_shots

    @classmethod
    def from_backend(cls, backend, **kwargs) -> MaestroSampler:
        """A sampler with the simulator options of a :class:`~qiskit_maestro.MaestroBackend`."""
        return cls(**{**backend.simulator_options(), **kwargs})

    @property
    def default_shots(self) -> int:
        return self._default_shots

    @property
    def seed(self) -> int | None:
        return self._options["seed_simulator"]

    @property
    def options(self) -> dict[str, Any]:
        """The simulator options in use."""
        return dict(self._options)

    def run(
        self, pubs: Iterable[SamplerPubLike], *, shots: int | None = None
    ) -> PrimitiveJob[PrimitiveResult[SamplerPubResult]]:
        if shots is None:
            shots = self._default_shots
        coerced = [SamplerPub.coerce(pub, shots) for pub in pubs]
        # Convert up front, so an unsupported instruction raises here.
        converted = [convert(pub.circuit) for pub in coerced]
        validate_combination(self._options)
        job = PrimitiveJob(self._run, coerced, converted)
        job._submit()
        return job

    def _run(self, pubs: list[SamplerPub], converted) -> PrimitiveResult[SamplerPubResult]:
        rng = np.random.default_rng(self.seed)
        results = [self._run_pub(pub, circuit, rng) for pub, circuit in zip(pubs, converted)]
        return PrimitiveResult(results, metadata={"version": 2})

    def _run_pub(self, pub: SamplerPub, circuit, rng: np.random.Generator) -> SamplerPubResult:
        angle_rows = circuit.angle_rows(pub.parameter_values)
        num_sets = len(angle_rows)
        metadata: dict[str, Any] = {"shots": pub.shots, "circuit_metadata": pub.circuit.metadata}
        if circuit.has_measurements:
            seeds = _execution.maestro_seeds(rng, self.seed is not None, num_sets)
            samples = _execution.sample(circuit, angle_rows, pub.shots, self._options, seeds)
            shots = np.stack([s.shots(rng) for s in samples])
            metadata["maestro"] = run_info([s.info for s in samples])
        else:
            shots = np.zeros((num_sets, pub.shots, circuit.num_clbits), dtype=np.uint8)

        meas = {}
        for creg in pub.circuit.cregs:
            clbits = [pub.circuit.find_bit(bit).index for bit in creg]
            packed = pack_register(shots, clbits)
            meas[creg.name] = BitArray(packed.reshape(pub.shape + packed.shape[1:]), creg.size)
        return SamplerPubResult(DataBin(**meas, shape=pub.shape), metadata=metadata)
