# Contributing to qiskit-maestro

Thanks for helping. Bug reports, fixes, tests and documentation are all welcome.
Everyone taking part follows the [Code of Conduct](CODE_OF_CONDUCT.md).

## Reporting a problem

Open an issue at
[github.com/QoroQuantum/qiskit-maestro/issues](https://github.com/QoroQuantum/qiskit-maestro/issues)
with:

- a short circuit that shows the problem, and what you expected;
- the output of
  `python -c "import qiskit, qiskit_maestro, importlib.metadata as m; print(qiskit.__version__, qiskit_maestro.__version__, m.version('qoro-maestro'))"`;
- for wrong results, the same circuit's result from `StatevectorSampler`,
  `StatevectorEstimator` or Aer.

If the problem is in Maestro itself rather than in the Qiskit layer, the
[Maestro repository](https://github.com/QoroQuantum/maestro) is the place for it.
[docs/maestro-api-notes.md](docs/maestro-api-notes.md) lists the Maestro
limitations this package already works around.

## Development setup

```bash
git clone https://github.com/QoroQuantum/qiskit-maestro
cd qiskit-maestro
python -m venv .venv && source .venv/bin/activate
pip install -e ".[test]" ruff
pytest
```

To test against Qiskit 1.x as well, make a second environment with
`pip install -e ".[test]" "qiskit<2"`. CI runs Python 3.10 to 3.13 against the
latest Qiskit 1.x and 2.x, plus Qiskit 1.1, the oldest version supported.

Tests marked `gpu` run only where Maestro's GPU backend is available, and are
skipped elsewhere. Tests marked `slow` use 100+ qubit MPS and 1,000-qubit
stabilizer circuits; they take a few seconds, and `pytest -m "not slow"` skips
them.

## Making a change

- Keep all conversion between Maestro's bit order (qubit 0 on the left) and
  Qiskit's (qubit 0 on the right) in `qiskit_maestro/bit_order.py`.
- Compare new behaviour with Qiskit's reference primitives or Aer in the tests,
  using circuits whose results change when the bit order is reversed.
- Raise `UnsupportedInstructionError` naming the instruction rather than
  returning a result that ignores it.
- Start every new source file with the SPDX header:

  ```python
  # SPDX-FileCopyrightText: 2026 Qoro Quantum
  # SPDX-License-Identifier: GPL-3.0-or-later
  ```

- Run `ruff check .` and `ruff format .` before you open a pull request, and
  add a line to [CHANGELOG.md](CHANGELOG.md) under "Unreleased".

By contributing, you agree that your contribution is licensed under the
GPL-3.0-or-later.

## Releasing (maintainers)

1. Move the "Unreleased" changelog entries under a new version heading.
2. Set `version` in `pyproject.toml` to match.
3. Push a tag `vX.Y.Z`. The publish workflow tests, builds and publishes to PyPI
   through trusted publishing, then creates the GitHub release.
