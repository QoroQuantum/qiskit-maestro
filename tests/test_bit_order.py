# SPDX-FileCopyrightText: 2026 Qoro Quantum
# SPDX-License-Identifier: GPL-3.0-or-later

"""Bit order, from the conversion functions up to every user-facing result.

Maestro writes qubit 0 on the left and Qiskit on the right. A mistake here
silently gives wrong physics, so each angle is tested on states and operators
that look different when reversed.
"""

from __future__ import annotations

import numpy as np
import pytest
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.primitives import BackendSamplerV2, BitArray, StatevectorEstimator, StatevectorSampler
from qiskit.quantum_info import SparsePauliOp

from qiskit_maestro import MaestroBackend, MaestroEstimator, MaestroSampler
from qiskit_maestro.bit_order import (
    maestro_counts_to_outcomes,
    maestro_samples_to_outcomes,
    outcomes_to_bitstrings,
    outcomes_to_hex,
    pack_register,
    pauli_label_to_maestro,
)

# ---- The conversion functions -------------------------------------------------


def test_pauli_label_is_reversed():
    # Qiskit "XIZ" is Z on qubit 0 and X on qubit 2.
    assert pauli_label_to_maestro("XIZ") == "ZIX"
    assert pauli_label_to_maestro("IIIY") == "YIII"


def test_counts_to_outcomes_puts_clbit_zero_in_column_zero():
    # Maestro's "100" means clbit 0 is 1.
    outcomes, frequencies = maestro_counts_to_outcomes({"100": 3, "011": 1}, 3)
    np.testing.assert_array_equal(outcomes, [[1, 0, 0], [0, 1, 1]])
    np.testing.assert_array_equal(frequencies, [3, 1])


def test_counts_to_outcomes_pads_unwritten_clbits():
    # Maestro's strings stop at the highest clbit written: "01" for 4 clbits.
    outcomes, _ = maestro_counts_to_outcomes({"01": 5}, 4)
    np.testing.assert_array_equal(outcomes, [[0, 1, 0, 0]])


def test_counts_to_outcomes_rejects_outcomes_wider_than_the_circuit():
    with pytest.raises(ValueError):
        maestro_counts_to_outcomes({"0001": 1}, 2)


@pytest.mark.parametrize("width,num_clbits", [(3, 3), (2, 5), (62, 62), (63, 63), (70, 72)])
def test_samples_to_outcomes_counts_rows_like_counts(width, num_clbits):
    # Maestro's rows have clbit 0 in column 0 and stop at the highest clbit written.
    rows = np.random.default_rng(width).integers(0, 2, size=(400, width), dtype=np.uint8)
    rows[:5] = rows[0]
    counts: dict[str, int] = {}
    for row in rows:
        key = "".join(map(str, row))
        counts[key] = counts.get(key, 0) + 1
    outcomes, frequencies = maestro_samples_to_outcomes(rows, num_clbits)
    expected_outcomes, expected_frequencies = maestro_counts_to_outcomes(counts, num_clbits)
    assert outcomes.dtype == np.uint8
    assert frequencies.dtype == np.int64
    assert outcomes.shape == (len(counts), num_clbits)
    assert dict(zip(map(bytes, outcomes), frequencies.tolist())) == dict(
        zip(map(bytes, expected_outcomes), expected_frequencies.tolist())
    )


def test_samples_to_outcomes_rejects_rows_wider_than_the_circuit():
    with pytest.raises(ValueError):
        maestro_samples_to_outcomes(np.array([[0, 0, 1]], dtype=np.uint8), 2)


def test_qiskit_strings_put_clbit_zero_on_the_right():
    outcomes = np.array([[1, 0, 0], [1, 1, 0]], dtype=np.uint8)
    assert outcomes_to_bitstrings(outcomes) == ["001", "011"]
    assert outcomes_to_hex(outcomes) == ["0x1", "0x3"]


def test_hex_handles_more_than_64_clbits():
    outcomes = np.zeros((1, 70), dtype=np.uint8)
    outcomes[0, 69] = 1
    assert outcomes_to_hex(outcomes) == [hex(1 << 69)]


def test_pack_register_matches_bitarray_from_bitstrings():
    # A register made of clbits 3, 0 and 2, in that order.
    shots = np.array([[1, 0, 0, 0], [0, 0, 1, 1], [1, 1, 1, 1]], dtype=np.uint8)
    packed = pack_register(shots, [3, 0, 2])
    expected = BitArray.from_bool_array(shots[:, [3, 0, 2]][:, ::-1], order="big")
    assert BitArray(packed, 3) == expected
    # Register value = clbit3 + 2*clbit0 + 4*clbit2.
    np.testing.assert_array_equal(BitArray(packed, 3).array.ravel(), [2, 5, 7])


def test_pack_register_over_eight_bits():
    shots = np.zeros((1, 10), dtype=np.uint8)
    shots[0, 9] = 1
    shots[0, 0] = 1
    bits = BitArray(pack_register(shots, list(range(10))), 10)
    assert bits.get_bitstrings() == ["1000000001"]


# ---- Through the backend and primitives -----------------------------------------


def _flip_qubit_0() -> QuantumCircuit:
    circuit = QuantumCircuit(3, 3)
    circuit.x(0)
    circuit.measure(range(3), range(3))
    return circuit


def test_backend_counts_and_memory_for_001():
    result = MaestroBackend().run(_flip_qubit_0(), shots=20, memory=True).result()
    assert result.get_counts() == {"001": 20}
    assert set(result.get_memory()) == {"001"}


def test_sampler_bitarray_for_001():
    data = MaestroSampler().run([_flip_qubit_0()], shots=20).result()[0].data
    assert data.c.get_counts() == {"001": 20}
    assert data.c.get_int_counts() == {1: 20}


def test_measuring_into_a_higher_clbit():
    # Qubit 0 measured into clbit 2: Qiskit shows "100".
    circuit = QuantumCircuit(3, 3)
    circuit.x(0)
    circuit.measure(0, 2)
    assert MaestroBackend().run(circuit, shots=5).result().get_counts() == {"100": 5}
    assert MaestroSampler().run([circuit], shots=5).result()[0].data.c.get_counts() == {"100": 5}


def _two_registers() -> QuantumCircuit:
    qr = QuantumRegister(3, "q")
    a = ClassicalRegister(2, "a")
    b = ClassicalRegister(1, "b")
    circuit = QuantumCircuit(qr, a, b)
    circuit.x(0)
    circuit.measure(0, a[1])  # a = 0b10
    circuit.measure(1, b[0])  # b = 0b0
    circuit.x(2)
    return circuit


def test_backend_counts_split_registers_like_aer():
    # Aer returns {"0 10": shots} for this circuit.
    assert MaestroBackend().run(_two_registers(), shots=8).result().get_counts() == {"0 10": 8}


def test_sampler_gives_one_bitarray_per_register():
    data = MaestroSampler().run([_two_registers()], shots=8).result()[0].data
    reference = StatevectorSampler().run([_two_registers()], shots=8).result()[0].data
    assert data.a.get_counts() == reference.a.get_counts() == {"10": 8}
    assert data.b.get_counts() == reference.b.get_counts() == {"0": 8}


def test_backend_sampler_v2_through_memory():
    data = (
        BackendSamplerV2(backend=MaestroBackend()).run([_two_registers()], shots=8).result()[0].data
    )
    assert data.a.get_counts() == {"10": 8}
    assert data.b.get_counts() == {"0": 8}


def test_registers_with_non_contiguous_bits():
    # Register r holds clbits 2 and 0 of the circuit, in that order.
    qr = QuantumRegister(3)
    loose = ClassicalRegister(3, "all")
    circuit = QuantumCircuit(qr, loose)
    reordered = ClassicalRegister(bits=[loose[2], loose[0]], name="r")
    circuit.add_register(reordered)
    circuit.x(2)
    circuit.measure([0, 1, 2], [0, 1, 2])
    mine = MaestroSampler().run([circuit], shots=4).result()[0].data
    reference = StatevectorSampler().run([circuit], shots=4).result()[0].data
    assert mine.all.get_counts() == reference.all.get_counts() == {"100": 4}
    assert mine.r.get_counts() == reference.r.get_counts() == {"01": 4}


@pytest.mark.parametrize("label", ["XIZ", "ZIX", "IYZ", "ZZI", "IIZ", "ZII"])
def test_estimator_uses_qiskit_pauli_order(label):
    # An asymmetric product state: qubit 0 in |1>, qubit 1 in |+>, qubit 2 in |+i>.
    circuit = QuantumCircuit(3)
    circuit.x(0)
    circuit.h(1)
    circuit.h(2)
    circuit.s(2)
    observable = SparsePauliOp(label)
    mine = MaestroEstimator().run([(circuit, observable)]).result()[0].data.evs
    reference = StatevectorEstimator().run([(circuit, observable)]).result()[0].data.evs
    np.testing.assert_allclose(mine, reference, atol=1e-12)


def test_estimator_single_z_on_each_qubit():
    circuit = QuantumCircuit(3)
    circuit.x(0)  # Only qubit 0 is flipped.
    observables = [SparsePauliOp("IIZ"), SparsePauliOp("IZI"), SparsePauliOp("ZII")]
    evs = MaestroEstimator().run([(circuit, observables)]).result()[0].data.evs
    np.testing.assert_allclose(evs, [-1, 1, 1], atol=1e-12)


def test_sampler_and_estimator_agree_on_parity():
    # <Z0> computed from samples has to agree with the estimator's value.
    circuit = QuantumCircuit(3)
    circuit.ry(0.7, 0)
    circuit.ry(2.1, 2)
    measured = circuit.copy()
    measured.measure_all()
    counts = MaestroSampler(seed=4).run([measured], shots=40_000).result()[0].data.meas.get_counts()
    z0_sampled = sum((1 if bits[-1] == "0" else -1) * n for bits, n in counts.items()) / 40_000
    z0 = MaestroEstimator().run([(circuit, "IIZ")]).result()[0].data.evs
    assert abs(z0_sampled - z0) < 0.02
    assert abs(z0 - np.cos(0.7)) < 1e-12
