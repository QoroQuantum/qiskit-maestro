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

from . import _execution, cloud
from .bit_order import pack_register
from .converter import ConvertedCircuit, convert
from .noise import NoiseTranslator
from .options import (
    merge_aer_options,
    resolve_method,
    resolve_options,
    run_info,
    validate_combination,
)

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
        service: A ``divi.backends.QoroService`` to run on the Qoro cloud, one
            cloud job per :meth:`run`. ``None`` (default) runs locally. On the
            cloud, circuits can't have classical control or a noise model yet.
        **simulator_options: Any :class:`~qiskit_maestro.MaestroBackend` option,
            such as ``method="matrix_product_state"``, ``device="GPU"`` or
            ``noise_model``.
    """

    def __init__(
        self,
        *,
        default_shots: int = 1024,
        seed: int | None = None,
        options: Mapping[str, Any] | None = None,
        service=None,
        **simulator_options,
    ):
        simulator_options = merge_aer_options(options, simulator_options)
        if seed is not None:
            simulator_options["seed_simulator"] = seed
        if service is not None:
            cloud.check_service(service)
        self._service = service
        self._options = resolve_options(simulator_options, local=service is None)
        if service is not None:
            cloud.check_options(self._options)
        self._default_shots = default_shots

    @classmethod
    def from_backend(cls, backend, **kwargs) -> MaestroSampler:
        """A sampler with the simulator options and service of a :class:`~qiskit_maestro.MaestroBackend`."""
        return cls(**{**backend.simulator_options(), "service": backend.service, **kwargs})

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

    @property
    def service(self):
        """The ``QoroService`` the sampler runs on, or ``None`` when it runs locally."""
        return self._service

    def run(
        self, pubs: Iterable[SamplerPubLike], *, shots: int | None = None
    ) -> PrimitiveJob[PrimitiveResult[SamplerPubResult]]:
        """Sample the PUBs.

        Locally, the job runs in the background. With a ``service``, the PUBs'
        circuits are submitted to the Qoro cloud as one job before this returns,
        and the returned :class:`~qiskit_maestro.cloud.QoroCloudPrimitiveJob`
        has the cloud's job ID.
        """
        if shots is None:
            shots = self._default_shots
        coerced = [SamplerPub.coerce(pub, shots) for pub in pubs]
        # Convert up front, so an unsupported instruction or noise raises here.
        model = self._options["noise_model"]
        translator = None if model is None else NoiseTranslator(model)
        prepared = [self._prepare(pub, translator) for pub in coerced]
        validate_combination(self._options, local=self._service is None)
        if self._service is not None:
            return self._submit_cloud(coerced, prepared)
        job = PrimitiveJob(self._run, coerced, prepared)
        job._submit()
        return job

    def _submit_cloud(self, pubs: list[SamplerPub], prepared) -> cloud.QoroCloudPrimitiveJob:
        """Submit every parameter set of every PUB with measurements as one cloud job."""
        programs = []
        rows = []
        for i, (pub, (circuit, _)) in enumerate(zip(pubs, prepared)):
            angle_rows = circuit.angle_rows(pub.parameter_values)
            rows.append(angle_rows)
            if circuit.has_measurements:
                labels = [f"{i}/{j}" for j in range(len(angle_rows))]
                programs.append(
                    cloud.Program(circuit, angle_rows, labels, shots=[pub.shots] * len(labels))
                )
        # Without noise, every circuit resolves to the same method.
        options = prepared[0][1] if prepared else self._options
        run = cloud.CloudRun(
            self._service,
            programs,
            cloud.maestro_config(options),
            gpu=options["device"] == "GPU",
        )

        def build(results) -> PrimitiveResult[SamplerPubResult]:
            rng = np.random.default_rng(self.seed)
            out = []
            for i, (pub, (circuit, pub_options)) in enumerate(zip(pubs, prepared)):
                samples = None
                if circuit.has_measurements:
                    samples = [
                        cloud.to_samples(results[f"{i}/{j}"], circuit.num_clbits, pub_options)
                        for j in range(len(rows[i]))
                    ]
                result = self._pub_result(pub, circuit, len(rows[i]), samples, rng)
                if run.submitted:
                    result.metadata["qoro_job_id"] = run.job_id
                out.append(result)
            return PrimitiveResult(out, metadata={"version": 2})

        return cloud.QoroCloudPrimitiveJob(run, build)

    def _prepare(
        self, pub: SamplerPub, translator: NoiseTranslator | None
    ) -> tuple[ConvertedCircuit, dict[str, Any]]:
        circuit = convert(pub.circuit, noise=translator)
        noisy = translator is not None and translator.has_quantum_errors
        method = resolve_method(self._options, circuit, pub.shots, noisy)
        return circuit.with_noise(translator, method), {**self._options, "method": method}

    def _run(self, pubs: list[SamplerPub], prepared) -> PrimitiveResult[SamplerPubResult]:
        rng = np.random.default_rng(self.seed)
        results = [
            self._run_pub(pub, circuit, options, rng)
            for pub, (circuit, options) in zip(pubs, prepared)
        ]
        return PrimitiveResult(results, metadata={"version": 2})

    def _run_pub(
        self, pub: SamplerPub, circuit, options: dict[str, Any], rng: np.random.Generator
    ) -> SamplerPubResult:
        angle_rows = circuit.angle_rows(pub.parameter_values)
        num_sets = len(angle_rows)
        samples = None
        if circuit.has_measurements:
            seeds = _execution.maestro_seeds(rng, self.seed is not None, num_sets)
            samples = _execution.sample(circuit, angle_rows, pub.shots, options, seeds)
        return self._pub_result(pub, circuit, num_sets, samples, rng)

    @staticmethod
    def _pub_result(
        pub: SamplerPub,
        circuit,
        num_sets: int,
        samples: list[_execution.Samples] | None,
        rng: np.random.Generator,
    ) -> SamplerPubResult:
        """A PUB's result from the samples of each parameter set (``None`` without measurements)."""
        metadata: dict[str, Any] = {"shots": pub.shots, "circuit_metadata": pub.circuit.metadata}
        if samples is not None:
            # np.stack can't take an empty list, which an empty parameter batch gives.
            shots = (
                np.stack([s.shots(rng) for s in samples])
                if samples
                else np.zeros((0, pub.shots, circuit.num_clbits), dtype=np.uint8)
            )
            if info := run_info([s.info for s in samples]):
                metadata["maestro"] = info
        else:
            shots = np.zeros((num_sets, pub.shots, circuit.num_clbits), dtype=np.uint8)

        meas = {}
        for creg in pub.circuit.cregs:
            clbits = [pub.circuit.find_bit(bit).index for bit in creg]
            packed = pack_register(shots, clbits)
            meas[creg.name] = BitArray(packed.reshape(pub.shape + packed.shape[1:]), creg.size)
        return SamplerPubResult(DataBin(**meas, shape=pub.shape), metadata=metadata)
