# SPDX-FileCopyrightText: 2026 Qoro Quantum
# SPDX-License-Identifier: GPL-3.0-or-later

"""Simulator options shared by the backend and the primitives.

Option names follow Qiskit Aer where the concept is the same, and Aer's longer
names are accepted as aliases, so ``AerSimulator(...)`` keyword arguments carry
over. :func:`simulator_config` maps the options onto a
:class:`maestro.SimulatorConfig`.
"""

from __future__ import annotations

import numbers
import os
from collections.abc import Mapping
from typing import Any

import maestro

from .exceptions import MaestroGpuError
from .noise import is_aer_noise_model

__all__ = [
    "AUTOMATIC",
    "DEFAULT_OPTIONS",
    "METHODS",
    "OPTION_NAMES",
    "merge_aer_options",
    "normalize_options",
    "resolve_method",
    "resolve_options",
    "simulator_config",
]

# Plugin method name -> maestro.SimulationType member name.
METHODS = {
    "statevector": "Statevector",
    "matrix_product_state": "MatrixProductState",
    "stabilizer": "Stabilizer",
    "extended_stabilizer": "ExtendedStabilizer",
    "density_matrix": "DensityMatrix",
    "tensor_network": "TensorNetwork",
    "pauli_propagation": "PauliPropagator",
}
_METHOD_ALIASES = {"mps": "matrix_product_state", "pauli_propagator": "pauli_propagation"}

# Aer's default method. It resolves per circuit: see resolve_method.
AUTOMATIC = "automatic"

# The methods Maestro's GPU backend implements. The core runs the others on the
# CPU when asked for the GPU, so they're rejected up front.
GPU_METHODS = frozenset(
    {"statevector", "matrix_product_state", "tensor_network", "pauli_propagation"}
)

DEFAULT_OPTIONS: dict[str, Any] = {
    "method": "statevector",
    "device": "CPU",
    "precision": "double",
    "seed_simulator": None,
    "max_bond_dimension": None,
    "truncation_threshold": None,
    "max_parallel_experiments": 0,
    "maestro_options": None,
    "noise_model": None,
}

_OPTION_ALIASES = {
    "matrix_product_state_max_bond_dimension": "max_bond_dimension",
    "matrix_product_state_truncation_threshold": "truncation_threshold",
}

# Every name an option can be given under.
OPTION_NAMES = frozenset(DEFAULT_OPTIONS) | frozenset(_OPTION_ALIASES)

# SimulatorConfig fields the options above already set.
_MANAGED_FIELDS = frozenset(
    {
        "simulator_type",
        "simulation_type",
        "precision",
        "seed",
        "max_bond_dimension",
        "singular_value_threshold",
    }
)

_GPU_SIMULATORS = frozenset(
    member.value
    for name in ("Gpu", "DistributedGpu", "DistributedMpiGpu")
    if (member := getattr(maestro.SimulatorType, name, None)) is not None
)

# Names for the ints Maestro reports in result["simulator"] and result["method"].
SIMULATOR_NAMES = {
    getattr(maestro.SimulatorType, name).value: name
    for name in dir(maestro.SimulatorType)
    if name[:1].isupper()
}
METHOD_NAMES = {
    getattr(maestro.SimulationType, maestro_name).value: name
    for name, maestro_name in {
        **METHODS,
        "extended_stabilizer": "ExtendedStabilizer",
        "path_integral": "PathIntegral",
        "matrix_product_operator": "MatrixProductOperator",
    }.items()
}


def _choice(name: str, value: Any, allowed: Mapping[str, Any] | frozenset) -> str:
    if not isinstance(value, str) or value.lower() not in allowed:
        raise ValueError(f"{name} must be one of {sorted(allowed)}, got {value!r}.")
    return value.lower()


def _int(name: str, value: Any, minimum: int) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum} or None, got {value!r}.")
    return int(value)


def normalize_options(options: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve aliases and validate each option on its own.

    Returns only the options passed in, under their canonical names. Checks
    that involve two options are in :func:`validate_combination`.

    Raises:
        ValueError: for an unknown option or an invalid value.
    """
    out: dict[str, Any] = {}
    for key, value in options.items():
        name = _OPTION_ALIASES.get(key, key)
        if name in out:
            raise ValueError(f"Option {name!r} was given twice (as {key!r} and another alias).")
        if name == "method":
            if isinstance(value, str):
                value = _METHOD_ALIASES.get(value.lower(), value)
            value = _choice("method", value, {**METHODS, AUTOMATIC: None})
        elif name == "device":
            if not isinstance(value, str) or value.upper() not in ("CPU", "GPU"):
                raise ValueError(f"device must be 'CPU' or 'GPU', got {value!r}.")
            value = value.upper()
        elif name == "precision":
            value = _choice("precision", value, frozenset({"single", "double"}))
        elif name == "seed_simulator":
            value = _int(name, value, 0)
        elif name == "max_bond_dimension":
            value = _int(name, value, 1)
        elif name == "max_parallel_experiments":
            value = _int(name, value, 0) or 0
        elif name == "truncation_threshold":
            if value is not None:
                if not isinstance(value, numbers.Real) or value < 0:
                    raise ValueError(f"truncation_threshold must be a number >= 0, got {value!r}.")
                value = float(value)
        elif name == "noise_model":
            if value is not None and not is_aer_noise_model(value):
                raise ValueError(
                    "noise_model must be a qiskit_aer.noise.NoiseModel or None, got "
                    f"{type(value).__name__}."
                )
        elif name == "maestro_options":
            if value is not None:
                value = dict(value)
                fields = set(maestro.SimulatorConfig._fields)
                for field in value:
                    if field in _MANAGED_FIELDS:
                        raise ValueError(
                            f"Set {field!r} through the qiskit-maestro option for it, "
                            "not maestro_options."
                        )
                    if field not in fields:
                        raise ValueError(f"{field!r} isn't a maestro.SimulatorConfig field.")
        else:
            raise ValueError(f"Unknown Maestro option {key!r}.")
        out[name] = value
    return out


def merge_aer_options(
    options: Mapping[str, Any] | None,
    simulator_options: Mapping[str, Any],
    allowed: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """Fold Aer's primitive ``options`` dict into plain simulator options.

    Aer's primitives take ``options={"backend_options": {...}, "run_options": {...}}``.
    Accepting that form keeps the switch from Aer to one line. Keyword options
    win over the dict. ``allowed`` names the other top-level keys the caller
    reads itself, such as ``default_precision``.
    """
    options = dict(options or {})
    unknown = set(options) - {"backend_options", "run_options"} - allowed
    if unknown:
        raise ValueError(
            f"Unknown options {sorted(unknown)}; expected backend_options and run_options."
        )
    merged = dict(options.get("backend_options") or {})
    run_options = dict(options.get("run_options") or {})
    # Shots are a primitive argument, not a simulator option.
    run_options.pop("shots", None)
    merged.update(run_options)
    merged.update(simulator_options)
    return merged


def resolve_options(*layers: Mapping[str, Any], local: bool = True) -> dict[str, Any]:
    """Apply option layers over the defaults and validate the result.

    ``local`` is ``False`` for runs on the Qoro cloud; see :func:`validate_combination`.
    """
    options = dict(DEFAULT_OPTIONS)
    for layer in layers:
        options.update(normalize_options(layer))
    validate_combination(options, local=local)
    return options


def validate_combination(options: Mapping[str, Any], *, local: bool = True) -> None:
    """Check a full set of options, including that a requested GPU can be used.

    With ``local=False``, for runs on the Qoro cloud, this machine's GPU isn't
    checked: the cloud's is used.

    Raises:
        MaestroGpuError: if ``device="GPU"`` can't be honoured.
    """
    if options["device"] != "GPU":
        return
    if options["method"] not in GPU_METHODS | {AUTOMATIC}:
        raise MaestroGpuError(
            f"method={options['method']!r} doesn't run on Maestro's GPU backend, which "
            f"supports {sorted(GPU_METHODS)}."
        )
    if local and not maestro.is_gpu_available():
        raise MaestroGpuError(
            "device='GPU' was requested, but Maestro's GPU backend isn't available: the "
            "GPU plugin couldn't be loaded, no CUDA device is visible, or the licence "
            "(MAESTRO_LICENSE_KEY) wasn't accepted."
        )


def _concrete(method: str) -> str:
    """A method Maestro can be configured with: ``"automatic"`` falls back to the default."""
    return DEFAULT_OPTIONS["method"] if method == AUTOMATIC else method


def resolve_method(options: Mapping[str, Any], circuit, shots: int, noisy: bool) -> str:
    """The method a circuit runs with: ``options["method"]``, or Aer's choice for ``"automatic"``.

    Like Aer, ``"automatic"`` picks the density matrix for a noisy circuit whose
    measurements all come at the end, when the shots outnumber the ``2**n``
    amplitudes a statevector would sample one by one. Otherwise it picks the
    statevector, Maestro's default, which Maestro may still run on a faster
    engine, such as the stabilizer for a Clifford circuit.

    Args:
        options: Resolved options.
        circuit: The :class:`~qiskit_maestro.converter.ConvertedCircuit`.
        shots: The shots, or for expectation values the trajectories Aer would use.
        noisy: Whether a noise model with quantum errors applies.
    """
    method = options["method"]
    if method != AUTOMATIC:
        return method
    # Aer counts only the qubits the circuit uses, so a circuit transpiled for a
    # large device still counts as small.
    width = len(circuit.active_qubits)
    if (
        noisy
        and options["device"] == "CPU"
        and width < 64
        and shots > 2**width
        and circuit.can_sample
        and not any(save.kind.startswith("save_statevector") for save in circuit.saves)
    ):
        return "density_matrix"
    return DEFAULT_OPTIONS["method"]


def simulator_config(
    options: Mapping[str, Any], seed: int | None = None
) -> maestro.SimulatorConfig:
    """Build the :class:`maestro.SimulatorConfig` for a full set of options."""
    kwargs = dict(options["maestro_options"] or {})
    kwargs.update(
        simulator_type=maestro.SimulatorType.Gpu
        if options["device"] == "GPU"
        else maestro.SimulatorType.QCSim,
        simulation_type=getattr(maestro.SimulationType, METHODS[_concrete(options["method"])]),
        precision=options["precision"],
        seed=seed,
        max_bond_dimension=options["max_bond_dimension"],
        singular_value_threshold=options["truncation_threshold"],
    )
    return maestro.SimulatorConfig(**kwargs)


def check_device(result: Mapping[str, Any], options: Mapping[str, Any]) -> None:
    """Raise if Maestro ran on the CPU although the GPU was requested.

    The core falls back to the CPU silently when the GPU can't be used, and
    ``result["simulator"]`` is the only record of where a circuit ran.
    """
    if options["device"] == "GPU" and result.get("simulator") not in _GPU_SIMULATORS:
        ran_on = SIMULATOR_NAMES.get(result.get("simulator"), result.get("simulator"))
        raise MaestroGpuError(
            f"device='GPU' was requested, but Maestro ran the circuit on {ran_on} (the CPU)."
        )


def run_info(results: list[Mapping[str, Any]]) -> dict[str, Any]:
    """The simulators and methods Maestro actually used, for result metadata.

    Results that don't say are left out; the Qoro cloud's never do. With none
    that say, the dict is empty. Where Maestro ran noise realizations on its own
    workers and says how many, the most any of the calls used is added as
    ``realization_workers``.
    """
    simulators = sorted(
        {
            SIMULATOR_NAMES.get(r["simulator"], str(r["simulator"]))
            for r in results
            if "simulator" in r
        }
    )
    methods = sorted(
        {METHOD_NAMES.get(r["method"], str(r["method"])) for r in results if "method" in r}
    )
    info: dict[str, Any] = (
        {"simulators": simulators, "methods": methods} if simulators or methods else {}
    )
    workers = [r["realization_workers"] for r in results if r.get("realization_workers")]
    if workers:
        info["realization_workers"] = max(workers)
    return info


def parallel_workers(options: Mapping[str, Any], num_qubits: int, num_tasks: int) -> int:
    """How many threads to run independent Maestro calls on.

    Maestro releases the GIL, so threads run in parallel. ``0`` (the default)
    picks the CPU count, except for the GPU and for dense methods on circuits
    large enough that several copies of the state could exhaust memory.
    """
    requested = options["max_parallel_experiments"]
    if requested == 0:
        dense_limit = {"statevector": 20, "density_matrix": 10}.get(_concrete(options["method"]))
        if options["device"] == "GPU" or (dense_limit is not None and num_qubits > dense_limit):
            return 1
        requested = os.cpu_count() or 1
    return max(1, min(requested, num_tasks))
