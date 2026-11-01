# SPDX-FileCopyrightText: 2026 Qoro Quantum
# SPDX-License-Identifier: GPL-3.0-or-later

"""A native Qiskit ``EstimatorV2`` on Maestro."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
from qiskit import QuantumCircuit
from qiskit.primitives import BaseEstimatorV2, DataBin, PrimitiveResult, PubResult
from qiskit.primitives.containers.estimator_pub import EstimatorPub, EstimatorPubLike
from qiskit.primitives.primitive_job import PrimitiveJob

from . import _execution, evolution
from .bit_order import pauli_label_to_maestro
from .converter import ConvertedCircuit, convert
from .exceptions import UnsupportedInstructionError
from .options import merge_aer_options, resolve_options, run_info, validate_combination

__all__ = ["MaestroEstimator"]

# SparseObservable projector letters as sums of Paulis: |0><0| = (I + Z) / 2, etc.
_PROJECTORS = {
    "0": (("I", 0.5), ("Z", 0.5)),
    "1": (("I", 0.5), ("Z", -0.5)),
    "+": (("I", 0.5), ("X", 0.5)),
    "-": (("I", 0.5), ("X", -0.5)),
    "r": (("I", 0.5), ("Y", 0.5)),
    "l": (("I", 0.5), ("Y", -0.5)),
}


def _pauli_terms(label: str, coeff: float) -> list[tuple[str, float]]:
    """Expand one Qiskit-order term into Pauli strings, expanding projectors."""
    terms = [("", float(coeff))]
    for letter in label:
        if letter in "IXYZ":
            terms = [(pauli + letter, c) for pauli, c in terms]
        elif letter in _PROJECTORS:
            terms = [(pauli + p, c * w) for pauli, c in terms for p, w in _PROJECTORS[letter]]
        else:
            raise ValueError(f"Unsupported letter {letter!r} in observable term {label!r}.")
    return terms


class _Observable:
    """An observable as Maestro-order Pauli strings with coefficients.

    The identity part is kept as a constant, since it needs no simulation.
    """

    def __init__(self, terms: Mapping[str, float]):
        self.constant = 0.0
        self.paulis: list[str] = []
        self.coeffs: list[float] = []
        for label, coeff in terms.items():
            for pauli, c in _pauli_terms(label, coeff):
                if pauli.strip("I"):
                    self.paulis.append(pauli_label_to_maestro(pauli))
                    self.coeffs.append(c)
                else:
                    self.constant += c

    def columns(self, needed: dict[str, int]) -> list[int]:
        """Register this observable's Pauli strings in ``needed``; return their positions."""
        return [needed.setdefault(p, len(needed)) for p in self.paulis]

    def value(self, columns: list[int], values: np.ndarray) -> float:
        """The expectation value, given the values of the strings at ``columns``."""
        return self.constant + float(np.dot(self.coeffs, values[columns]))


def _parse(terms: Mapping[str, float], cache: dict[int, tuple]) -> _Observable:
    """Parse an observable once, however often broadcasting repeats it.

    The cache keeps ``terms`` alive, so its id can't be reused by another
    observable while the cache exists.
    """
    entry = cache.get(id(terms))
    if entry is None:
        entry = cache[id(terms)] = (terms, _Observable(terms))
    return entry[1]


# The methods where Maestro's incremental evolution beats one simulation per
# circuit, and the widest circuit each is used for automatically. Tensor
# networks and Pauli propagation recompute from the start at every step anyway.
_EVOLUTION_LIMITS = {
    "matrix_product_state": None,
    "stabilizer": None,
    "statevector": 30,
    "density_matrix": 15,
}


class MaestroEstimator(BaseEstimatorV2):
    """Expectation values on Maestro, a drop-in for ``StatevectorEstimator`` and Aer's ``EstimatorV2``.

    Values are computed from the simulated state (statevector, MPS, stabilizer
    tableau, density matrix, tensor network or Pauli propagation), never from
    shots. With ``precision > 0``, Gaussian noise of that standard deviation is
    added, as :class:`~qiskit.primitives.StatevectorEstimator` does, and
    ``stds`` reports it.

    Time evolution runs on one persistent simulator. When the PUBs of a call
    are an initial circuit followed by 0, 1, 2, ... repetitions of the same
    step, as when each time point of a Trotter evolution is its own circuit,
    the estimator simulates the longest one once and reads every time point's
    values on the way. :meth:`run_evolution` does the same when you give it
    the initial circuit and the step yourself.

    Args:
        default_precision: Precision for PUBs that don't set their own. ``0``
            (default) returns exact values.
        seed: Seed for the precision noise and for any mid-circuit resets.
        options: Aer's form, ``{"default_precision": ..., "backend_options":
            {...}, "run_options": {...}}``, for code moving from
            ``qiskit_aer.primitives.EstimatorV2``.
        incremental_evolution: Recognise time-evolution PUBs in :meth:`run`
            (default ``True``). Set ``False`` to simulate every circuit from
            the start.
        **simulator_options: Any :class:`~qiskit_maestro.MaestroBackend` option,
            such as ``method="matrix_product_state"`` or ``device="GPU"``.
    """

    def __init__(
        self,
        *,
        default_precision: float = 0.0,
        seed: int | None = None,
        options: Mapping[str, Any] | None = None,
        incremental_evolution: bool = True,
        **simulator_options,
    ):
        simulator_options = merge_aer_options(
            options, simulator_options, allowed=frozenset({"default_precision"})
        )
        if options and "default_precision" in options:
            default_precision = options["default_precision"]
        if seed is not None:
            simulator_options["seed_simulator"] = seed
        self._options = resolve_options(simulator_options)
        self._default_precision = default_precision
        self._incremental_evolution = incremental_evolution

    @classmethod
    def from_backend(cls, backend, **kwargs) -> MaestroEstimator:
        """An estimator with the simulator options of a :class:`~qiskit_maestro.MaestroBackend`."""
        return cls(**{**backend.simulator_options(), **kwargs})

    @property
    def default_precision(self) -> float:
        return self._default_precision

    @property
    def seed(self) -> int | None:
        return self._options["seed_simulator"]

    @property
    def options(self) -> dict[str, Any]:
        """The simulator options in use."""
        return dict(self._options)

    def run(
        self, pubs: Iterable[EstimatorPubLike], *, precision: float | None = None
    ) -> PrimitiveJob[PrimitiveResult[PubResult]]:
        if precision is None:
            precision = self._default_precision
        coerced = [EstimatorPub.coerce(pub, precision) for pub in pubs]
        # Convert up front, so an unsupported instruction raises here.
        converted = [convert(pub.circuit, estimator=True) for pub in coerced]
        validate_combination(self._options)
        job = PrimitiveJob(self._run, coerced, converted)
        job._submit()
        return job

    def run_evolution(
        self,
        initial_state: QuantumCircuit,
        step: QuantumCircuit,
        observables,
        steps: Sequence[int],
        *,
        precision: float | None = None,
    ) -> PrimitiveJob[PrimitiveResult[PubResult]]:
        """Expectation values after repeating ``step`` on ``initial_state``.

        Uses one simulator for the whole evolution: ``initial_state`` runs once,
        then ``step`` is applied again and again, and the observables are
        evaluated on the live state after each count in ``steps``. That costs
        one pass to the largest count, where a circuit per time point costs a
        pass per point.

        Args:
            initial_state: The circuit that prepares the state, with bound
                parameters. Can be empty.
            step: One time step, such as one Trotter step, with bound parameters.
            observables: Anything a PUB takes as observables: a ``SparsePauliOp``,
                a label, or an array of them.
            steps: Step counts to evaluate at, in any order; ``0`` is the
                initial state.
            precision: As in :meth:`run`.

        Returns:
            A job whose result holds one ``PubResult`` with ``evs`` of shape
            ``(len(steps), *observables.shape)``.

        Raises:
            UnsupportedInstructionError: if a circuit holds a ``reset``, which
                Maestro's incremental evolution doesn't apply correctly yet.
        """
        if precision is None:
            precision = self._default_precision
        if initial_state.num_qubits != step.num_qubits:
            raise ValueError(
                f"initial_state has {initial_state.num_qubits} qubits but step has {step.num_qubits}."
            )
        counts = np.asarray(steps)
        if (
            counts.ndim != 1
            or counts.size == 0
            or not np.issubdtype(counts.dtype, np.integer)
            or (counts < 0).any()
        ):
            raise ValueError("steps must be a non-empty list of non-negative integers.")
        # A PUB on the initial state checks the observables and precision.
        pub = EstimatorPub.coerce((initial_state, observables), precision)
        circuits = {}
        for name, circuit in (("initial_state", initial_state), ("step", step)):
            circuits[name] = convert(circuit, estimator=True)
            if circuits[name].parameters:
                names = ", ".join(sorted(p.name for p in circuits[name].parameters))
                raise ValueError(f"{name} has unbound parameters: {names}. Assign them first.")
            if any(op.kind == "reset" for op in circuits[name].ops):
                raise UnsupportedInstructionError(
                    "reset",
                    "Maestro's incremental evolution leaves a qubit in superposition "
                    "unreset, so run_evolution doesn't accept resets. Use run() with "
                    "one circuit per time point instead.",
                )
        validate_combination(self._options)
        job = PrimitiveJob(
            self._run_evolution,
            pub,
            circuits["initial_state"],
            circuits["step"],
            [int(c) for c in counts],
            step.metadata,
        )
        job._submit()
        return job

    def _run_evolution(
        self, pub, initial, step, steps, step_metadata
    ) -> PrimitiveResult[PubResult]:
        observable_of = np.asarray(pub.observables)
        needed: dict[str, int] = {}
        parsed: dict[int, tuple] = {}
        plan = []
        for index in np.ndindex(observable_of.shape):
            observable = _parse(observable_of[index], parsed)
            plan.append((index, observable, observable.columns(needed)))
        values, info = evolution.evolve(initial.ops, step.ops, list(needed), steps, self._options)

        evs = np.empty((len(steps), *observable_of.shape), dtype=float)
        for t in range(len(steps)):
            for index, observable, columns in plan:
                evs[(t, *index)] = observable.value(columns, values[t])
        metadata = {"steps": steps, "circuit_metadata": step_metadata}
        return PrimitiveResult(
            [self._result(pub.precision, evs, metadata, [info], evolved=True)],
            metadata={"version": 2},
        )

    def _run(self, pubs: list[EstimatorPub], converted) -> PrimitiveResult[PubResult]:
        evolved = self._run_chain(pubs, converted)
        results = [
            evolved[i] if i in evolved else self._run_pub(pub, circuit)
            for i, (pub, circuit) in enumerate(zip(pubs, converted))
        ]
        return PrimitiveResult(results, metadata={"version": 2})

    def _run_chain(self, pubs: list[EstimatorPub], converted) -> dict[int, PubResult]:
        """Run the PUBs that form one time evolution on one simulator.

        Returns results by PUB index; PUBs left out run one by one.
        """
        method = self._options["method"]
        if not self._incremental_evolution or method not in _EVOLUTION_LIMITS:
            return {}
        # Only PUBs with one parameter set, no parameters and no resets, which
        # Maestro's incremental evolution doesn't apply correctly yet.
        members = [
            i
            for i, (pub, circuit) in enumerate(zip(pubs, converted))
            if pub.parameter_values.shape == ()
            and not circuit.parameters
            and not any(op.kind == "reset" for op in circuit.ops)
        ]
        widths = {converted[i].num_qubits for i in members}
        limit = _EVOLUTION_LIMITS[method]
        if len(widths) != 1 or (limit is not None and widths.pop() > limit):
            return {}
        chain = evolution.find_chain([converted[i].ops for i in members])
        if chain is None:
            return {}

        needed: dict[str, int] = {}
        parsed: dict[int, tuple] = {}
        plans = []
        for i in members:
            observable_of = np.asarray(pubs[i].observables)
            plans.append(
                [
                    (index, observable, observable.columns(needed))
                    for index in np.ndindex(observable_of.shape)
                    for observable in [_parse(observable_of[index], parsed)]
                ]
            )
        values, info = evolution.evolve(
            chain.initial, chain.step, list(needed), chain.steps, self._options
        )

        results = {}
        for row, (i, plan) in enumerate(zip(members, plans)):
            pub = pubs[i]
            evs = np.empty(pub.shape, dtype=float)
            for index, observable, columns in plan:
                evs[index] = observable.value(columns, values[row])
            metadata = {"circuit_metadata": pub.circuit.metadata}
            row_info = (
                None
                if info is None
                else {**info, "bond_dimensions": [info["bond_dimensions"][row]]}
            )
            results[i] = self._result(pub.precision, evs, metadata, [row_info], evolved=True)
        return results

    def _run_pub(self, pub: EstimatorPub, circuit: ConvertedCircuit) -> PubResult:
        angle_rows = circuit.angle_rows(pub.parameter_values)
        shape = pub.shape
        set_of = np.broadcast_to(
            np.arange(len(angle_rows)).reshape(pub.parameter_values.shape), shape
        )
        observable_of = np.broadcast_to(np.asarray(pub.observables), shape)

        # The Pauli strings each parameter set needs, each listed once, and for
        # every result entry the index of each of its terms in that list.
        labels: list[dict[str, int]] = [{} for _ in angle_rows]
        parsed: dict[int, tuple] = {}
        plan = []
        for index in np.ndindex(shape):
            observable = _parse(observable_of[index], parsed)
            plan.append((index, observable, observable.columns(labels[set_of[index]])))

        # Maestro's seeds (used by mid-circuit resets) come from their own
        # stream, so the precision noise matches StatevectorEstimator's draw
        # for draw.
        seed_rng = np.random.default_rng(None if self.seed is None else [self.seed, 1])
        seeds = _execution.maestro_seeds(seed_rng, self.seed is not None, len(angle_rows))
        estimates = _execution.estimate(
            circuit, angle_rows, [list(needed) for needed in labels], self._options, seeds
        )

        evs = np.empty(shape, dtype=float)
        for index, observable, columns in plan:
            evs[index] = observable.value(columns, estimates[set_of[index]][0])
        metadata = {"circuit_metadata": pub.circuit.metadata}
        return self._result(pub.precision, evs, metadata, [info for _, info in estimates])

    def _result(self, precision, evs, metadata, infos, *, evolved: bool = False) -> PubResult:
        """Add the precision noise and assemble a ``PubResult``."""
        if precision:
            # A generator per PUB, as StatevectorEstimator does, so equal PUBs
            # give equal noise for a given seed.
            rng = np.random.default_rng(self.seed)
            evs = np.asarray(rng.normal(evs, precision), dtype=float)
            stds = np.full(evs.shape, float(precision))
        else:
            stds = np.zeros(evs.shape)
        metadata = {"target_precision": precision, **metadata}
        infos = [info for info in infos if info is not None]
        if infos:
            metadata["maestro"] = run_info(infos)
            if evolved:
                metadata["maestro"]["incremental_evolution"] = True
                bond_dimensions = [d for info in infos for d in info["bond_dimensions"]]
                if any(bond_dimensions):
                    metadata["maestro"]["bond_dimensions"] = bond_dimensions
        return PubResult(DataBin(evs=evs, stds=stds, shape=evs.shape), metadata=metadata)
