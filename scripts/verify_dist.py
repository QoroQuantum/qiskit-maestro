# SPDX-FileCopyrightText: 2026 Qoro Quantum
# SPDX-License-Identifier: GPL-3.0-or-later

"""Check the built source distribution and wheel before they are published.

Run from the repository root after ``python -m build``::

    python scripts/verify_dist.py dist/ [--run-tests]

It checks that:

* the source distribution contains every file under ``tests/``, so the suite can be
  collected from the extracted archive, and that collecting it finds as many tests as
  collecting from the checkout;
* the wheel contains every module of the package and none of the tests;
* the wheel installs into a fresh virtual environment and imports from there, from a
  directory outside the checkout, with Python's isolated mode (``-I``).

``--run-tests`` also runs the extracted suite, without the slow tests.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

PACKAGE = "qiskit_maestro"


def fail(message: str) -> None:
    print(f"error: {message}", file=sys.stderr)
    sys.exit(1)


def run(command: list, *, check: bool = False, **kwargs) -> subprocess.CompletedProcess:
    print("+", " ".join(str(part) for part in command), flush=True)
    return subprocess.run(command, check=check, text=True, **kwargs)


def one(directory: Path, pattern: str) -> Path:
    found = sorted(directory.glob(pattern))
    if len(found) != 1:
        fail(f"expected one {pattern} in {directory}, found {[p.name for p in found]}")
    return found[0]


def source_files(root: Path, folder: str) -> set[str]:
    return {
        path.relative_to(root).as_posix()
        for path in (root / folder).rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    }


def check_sdist(sdist: Path, repo: Path, extract_to: Path) -> Path:
    with tarfile.open(sdist) as archive:
        # The "data" filter, which refuses paths outside the target, needs Python 3.10.12+.
        filters = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
        archive.extractall(extract_to, **filters)
        names = {
            Path(name).relative_to(Path(name).parts[0]).as_posix() for name in archive.getnames()
        }
    (root,) = (path for path in extract_to.iterdir() if path.is_dir())
    missing = sorted((source_files(repo, "tests") | source_files(repo, PACKAGE)) - names)
    if missing:
        fail(f"{sdist.name} is missing: {', '.join(missing)}")
    print(f"ok: {sdist.name} contains every file of tests/ and {PACKAGE}/")
    return root


def check_wheel(wheel: Path, repo: Path) -> None:
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
    missing = sorted(f for f in source_files(repo, PACKAGE) if f not in names and f.endswith(".py"))
    if missing:
        fail(f"{wheel.name} is missing: {', '.join(missing)}")
    stray = sorted(name for name in names if name.startswith(("tests/", "scripts/")))
    if stray:
        fail(f"{wheel.name} contains files that don't belong in it: {', '.join(stray)}")
    print(f"ok: {wheel.name} contains every module of {PACKAGE}/ and no tests")


def collected(python: Path, cwd: Path) -> int:
    result = run(
        [python, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider"],
        cwd=cwd,
        capture_output=True,
    )
    match = re.search(r"(\d+) tests? collected", result.stdout)
    if result.returncode != 0 or not match:
        print(result.stdout[-2000:], result.stderr[-2000:])
        fail(f"collecting the tests in {cwd} failed (exit code {result.returncode})")
    return int(match.group(1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("dist", type=Path, help="directory holding the sdist and the wheel")
    parser.add_argument("--run-tests", action="store_true", help="also run the extracted suite")
    args = parser.parse_args()
    repo = Path.cwd().resolve()
    dist = args.dist.resolve()
    if not (repo / "pyproject.toml").is_file():
        fail("run this from the repository root")
    sdist, wheel = one(dist, "*.tar.gz"), one(dist, "*.whl")

    with tempfile.TemporaryDirectory(prefix="verify-dist-") as scratch:
        scratch = Path(scratch).resolve()
        extracted = check_sdist(sdist, repo, scratch / "sdist")
        check_wheel(wheel, repo)

        env = scratch / "venv"
        run([sys.executable, "-m", "venv", env], check=True)
        python = env / ("Scripts" if sys.platform == "win32" else "bin") / "python"
        run([python, "-m", "pip", "install", "-q", f"{wheel}[test]"], check=True)

        outside = scratch / "outside"
        outside.mkdir()
        probe = (
            "import pathlib, qiskit_maestro as m;"
            "print(pathlib.Path(m.__file__).resolve());"
            "print(m.MaestroBackend, m.MaestroSampler, m.MaestroEstimator)"
        )
        result = run([python, "-I", "-c", probe], cwd=outside, capture_output=True, check=True)
        location = Path(result.stdout.splitlines()[0])
        if repo in location.parents or env not in location.parents:
            fail(f"{PACKAGE} imported from {location}, not from the installed wheel")
        print(f"ok: {PACKAGE} imports from the installed wheel at {location.parent}")

        wanted = collected(python, repo)
        found = collected(python, extracted)
        if found != wanted:
            fail(f"the extracted sdist collects {found} tests, the checkout {wanted}")
        print(f"ok: the extracted sdist collects all {found} tests")

        if args.run_tests:
            result = run(
                [python, "-m", "pytest", "-q", "-m", "not slow", "-p", "no:cacheprovider"],
                cwd=extracted,
            )
            if result.returncode != 0:
                fail("the test suite failed when run from the extracted sdist")


if __name__ == "__main__":
    main()
