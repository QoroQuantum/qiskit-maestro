# SPDX-FileCopyrightText: 2026 Qoro Quantum
# SPDX-License-Identifier: GPL-3.0-or-later

"""The example scripts run, and match the Aer versions they were moved from."""

from __future__ import annotations

import re
import runpy
from pathlib import Path

import numpy as np
import pytest

from .conftest import HAS_AER, assert_same_distribution

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def _aer_version(source: str) -> str:
    """Undo the lines marked "was:", giving back the Aer script."""
    lines = []
    for line in source.splitlines():
        match = re.match(r"^.*  # was: (.*)$", line)
        lines.append(match.group(1) if match else line)
    return "\n".join(lines) + "\n"


@pytest.mark.skipif(not HAS_AER, reason="qiskit-aer isn't installed")
def test_aer_noise_model_example_changes_two_lines_and_matches_aer(tmp_path):
    script = EXAMPLES / "aer_noise_model.py"
    source = script.read_text()
    aer_source = _aer_version(source)
    changed = [
        (mine, theirs)
        for mine, theirs in zip(source.splitlines(), aer_source.splitlines())
        if mine != theirs
    ]
    assert len(changed) == 2
    assert all("MaestroBackend" in mine and "AerSimulator" in theirs for mine, theirs in changed)

    aer_script = tmp_path / "aer_noise_model.py"
    aer_script.write_text(aer_source)
    mine = runpy.run_path(str(script))
    theirs = runpy.run_path(str(aer_script))

    assert_same_distribution(mine["counts"], theirs["counts"])
    shots = sum(mine["counts"].values())
    np.testing.assert_allclose(
        mine["probabilities"], theirs["probabilities"], atol=5 * 0.5 / np.sqrt(shots)
    )
    assert mine["zzz"] == pytest.approx(theirs["zzz"], abs=5 / np.sqrt(shots))
    assert mine["state"].is_valid()
    assert mine["state"].num_qubits == theirs["state"].num_qubits


def test_time_evolution_example_agrees_both_ways():
    example = runpy.run_path(str(EXAMPLES / "time_evolution.py"))
    np.testing.assert_allclose(example["per_circuit"], example["from_step"], atol=1e-10)
    assert example["from_step"][0] == pytest.approx(1.0)
    assert len(example["from_step"]) == 21


def test_getting_started_example_matches_qiskit():
    from qiskit.primitives import StatevectorEstimator

    example = runpy.run_path(str(EXAMPLES / "getting_started.py"))
    assert set(example["counts"]) == {"00", "11"}
    assert set(example["sampled"]) == {"00", "11"}
    pub = (example["bell"], example["observables"])
    reference = StatevectorEstimator().run([pub]).result()[0].data.evs
    np.testing.assert_allclose(example["evs"], reference, atol=1e-12)


def test_simulation_methods_example_runs_each_method_at_its_size():
    example = runpy.run_path(str(EXAMPLES / "simulation_methods.py"))
    assert set(example["results"]) == {"statevector", "matrix_product_state", "stabilizer"}
    for method, (num_qubits, counts, _) in example["results"].items():
        assert num_qubits == example["sizes"][method]
        assert set(counts) == {"all zeros", "all ones"}
        assert sum(counts.values()) == 1000


def test_pauli_propagation_example_converges_to_the_exact_value():
    example = runpy.run_path(str(EXAMPLES / "pauli_propagation.py"))
    assert example["n"] == 115
    results = example["results"]
    exact = results[None][0]
    errors = [abs(results[threshold][0] - exact) for threshold in (1e-2, 1e-3, 1e-4)]
    assert errors == sorted(errors, reverse=True)
    assert errors[-1] < 1e-3
