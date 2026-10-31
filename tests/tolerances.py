"""Accuracy tolerances, in ulps (componentwise metrics) or eps (normwise metrics), per framework, regime, metric, format.

Values come from ``tests/tolerances_<framework>.json``, written by ``tests/calibrate.py`` from measured error
distributions (provenance — seeds, date, library versions — is stored alongside). A key missing from the file is an error, not a
silent default: every tolerance the suite applies has been measured.
"""

import json
import pathlib

TOL: dict[tuple[str, str, str, str], float] = {
    tuple(r[:4]): float(r[4])
    for path in sorted(pathlib.Path(__file__).resolve().parent.glob("tolerances_*.json"))
    for r in json.loads(path.read_text())["tolerances"]
}


def tol(framework: str, regime: str, metric: str, fmt: str) -> float:
    """Calibrated tolerance for ``metric`` of ``regime`` in format ``fmt``."""
    try:
        return TOL[framework, regime, metric, fmt]
    except KeyError:
        raise KeyError(
            f"no calibrated tolerance for {(framework, regime, metric, fmt)}: run tests/calibrate.py"
        ) from None
