# Contributing to qiskit-maestro

Thanks for helping. Bug reports and feature requests are welcome as
[issues](https://github.com/QoroQuantum/qiskit-maestro/issues). Everyone taking
part follows the [Code of Conduct](CODE_OF_CONDUCT.md).

## We don't accept pull requests

Qoro Quantum holds the copyright to all the code in this repository, so we don't
merge code written by anyone outside Qoro Quantum. Pull requests from outside
contributors are closed without review. If you've found the cause of a bug,
describe it in words in an issue rather than as a patch, and we'll write the
fix.

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

This is how maintainers work on the package, and it's also useful for
reproducing a problem before you report it.

```bash
git clone https://github.com/QoroQuantum/qiskit-maestro
cd qiskit-maestro
python -m venv .venv && source .venv/bin/activate
pip install -e ".[test]" ruff
pytest
```

To test against Qiskit 1.x as well, make a second environment with
`pip install -e ".[test]" "qiskit<2"`. The Qoro cloud tests in
`tests/test_cloud.py` skip unless Divi is installed: use `".[test,cloud]"`
(Python 3.11 or later) to run them. They use a stand-in `QoroService` that runs
each job on Divi's local simulator, so they need no network or API key. For
coverage, `pip install pytest-cov` and run `pytest --cov`.

Tests marked `gpu` run only where Maestro's GPU backend is available, and are
skipped elsewhere. Tests marked `slow` use 100+ qubit MPS and 1,000-qubit
stabilizer circuits; they take a few seconds, and `pytest -m "not slow"` skips
them.

### What CI runs

[`.github/workflows/tests.yml`](.github/workflows/tests.yml) runs on every push
to every branch, on pull requests into `main`, every Monday against the latest
releases of the dependencies, and before every release. A new push to a branch
cancels the run still going for the one before. The workflow runs:

- the tests on Python 3.10 to 3.14 with the latest Qiskit 2.x, and on 3.10 to
  3.13 with the latest Qiskit 1.x;
- the Qoro cloud tests, on Python 3.11 and later with Qiskit 2.x, failing if
  Divi doesn't import rather than skipping;
- the same on macOS;
- every dependency at the lowest version `pyproject.toml` allows (Qiskit 1.1,
  Maestro 0.3.14, NumPy 1.23);
- coverage, which fails below `fail_under` in `pyproject.toml` and is
  summarised on the run's page;
- a build of the sdist and wheel, checked by `scripts/verify_dist.py`;
- `ruff`, the SPDX headers, and `actionlint` on the workflows.

To run it by hand, use "Run workflow" on the Tests workflow in the Actions tab.
Dependabot opens one pull request a month to update the workflows' actions.

## Making a change (maintainers)

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

## Releasing (maintainers)

1. Move the "Unreleased" changelog entries under a new version heading.
2. Set `version` in `pyproject.toml` to match.
3. Publish a GitHub release for the tag `vX.Y.Z`. The publish workflow checks the
   tag against `pyproject.toml`, runs the tests, builds and verifies the
   distributions, and uploads them to PyPI through trusted publishing. To
   publish by hand, or to dry-run the build without uploading, use "Run
   workflow" on the Publish to PyPI workflow in the Actions tab.
