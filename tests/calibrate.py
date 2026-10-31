"""Calibrate accuracy tolerances from measured error distributions and write ``tests/tolerances_<framework>.json``.

Two steps:

- ``measure`` writes one raw file per seed, ``tests/calibration/<framework>/seed_<k>.json``: for every
  (namespace, regime, metric, format) the largest error of the fixed implementation; separately, the largest native
  error in the regimes where native was taken to be accurate (``NATIVE_ACCURATE``: reported by ``merge``, never a
  tolerance since 30/10/2026); and the smallest native error on every metric native must fail (those ``must_fail``
  excludes too), with the library's options. Seeds already measured are skipped unless ``--force``, so the step
  resumes.
- ``merge`` combines the raw files of the given seeds. The tolerance is the next power of two at or above
  2 × the largest error of the library observed, with a floor of 2. Raw files of the previous scheme (before
  30/10/2026: native's errors merged into the tolerances) are refused. It refuses to write if any fixed error is non-finite or above
  ``--cap`` (calibration must not legitimise a defect), if a must-fail native error that ``must_fail`` keeps is
  within its tolerance (the check could not fail), or if the raw files differ in devices, options or the library's
  source (its SHA-256, recorded since 30/10/2026); it warns when
  that separation is under 4×, and reports the excluded must-fail metrics' separation without enforcing it.

    python tests/calibrate.py measure --framework jax --seeds 1 2 3 4 5 6 7 8 9 10
    python tests/calibrate.py merge --framework jax --seeds 1 2 3 4 5 6 7 8 9 10 --write
"""

import os

# Before any framework is imported: JAX must not pre-allocate most of the GPU (the other framework may share it).
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
import datetime
import hashlib
import importlib
import json
import math
import pathlib
import platform
import sys
from collections import defaultdict

HERE = pathlib.Path(__file__).resolve().parent
FACTOR, FLOOR, WEAK = 2.0, 2.0, 4.0
RULE = f"next power of two >= {FACTOR:g} x the library's worst observed (native excluded since 30/10/2026), floor {FLOOR:g}"


def _tolerance(x: float) -> float:
    y = FACTOR * x
    if not math.isfinite(y) or y > 2.0**1000:
        return math.inf
    return float(max(FLOOR, 2.0 ** math.ceil(math.log2(max(y, 1e-300)))))


def _versions(fw: str) -> dict[str, str]:
    import numpy as np

    out = {"python": platform.python_version(), "numpy": np.__version__}
    if fw == "jax":
        import jax
        import optax

        out |= {"jax": jax.__version__, "optax": optax.__version__, "devices": str(jax.devices())}
    else:
        import torch

        out |= {"torch": torch.__version__, "cuda": str(torch.version.cuda)}
    return out


def _raw_path(fw: str, seed: int) -> pathlib.Path:
    return HERE / "calibration" / fw / f"seed_{seed}.json"


def measure_seed(mod, seed: int) -> tuple[dict, dict, dict, list[str]]:
    """(the library's largest error per key: the tolerances; native's largest error where it was taken to be accurate:
    reported only; native's smallest error per must-fail key; devices) for one seed."""
    devices = [p.values[0] for p in mod.DEVICES]
    worst: dict[tuple, float] = defaultdict(float)
    native: dict[tuple, float] = defaultdict(float)
    must_fail: dict[tuple, float] = defaultdict(lambda: math.inf)

    def feed(tk: str, regime: str, errs: dict, fmt: str, store: dict = worst) -> None:
        for k, v in errs.items():
            store[tk, regime, k, fmt] = max(store[tk, regime, k, fmt], v)

    for fmt in mod.FMTS:
        for dev in devices:
            tk = mod.tol_key(dev)
            for regime in mod.FIXED_REGIMES:
                feed(tk, regime, mod.measure("fixed", regime, fmt, dev, seed), fmt)
            for regime, metrics in mod.NATIVE_ACCURATE.items():
                errs = mod.measure("native", regime, fmt, dev, seed)
                feed(tk, regime, {m: e for m, e in errs.items() if metrics is None or m in metrics}, fmt, native)
            for regime, metrics in mod.MUST_FAIL.items():
                # Every must-fail metric, the ones mod.must_fail excludes too: merge enforces the separation on those it
                # keeps and reports the others, so an exclusion can be dropped or restored from the raw files.
                errs = mod.measure("native", regime, fmt, dev, seed)
                for k in metrics:
                    must_fail[tk, regime, k, fmt] = min(must_fail[tk, regime, k, fmt], errs[k])
        print(f"seed {seed} {fmt}: done", flush=True)
    for fmt in mod.HESSIAN_FMTS:
        for regime in mod.HESSIAN_REGIMES:
            feed(mod.tol_key("cpu"), f"hessian_{regime}", mod.measure_hessian(regime, fmt, seed), fmt)
        for case in mod.EDGE_PAIRS:  # Seed-independent; measured with every seed so that each raw file is complete.
            feed(mod.tol_key("cpu"), "edge", mod.measure_edge(case, fmt), fmt)
    return dict(worst), dict(native), dict(must_fail), devices


def _encode(d: dict) -> list:
    return [[*key, v if math.isfinite(v) else repr(v)] for key, v in sorted(d.items())]


def _decode(rows: list) -> dict:
    return {tuple(r[:4]): float(r[4]) for r in rows}


def cmd_measure(args) -> None:
    sys.path[:0] = [str(HERE), str(HERE.parent / "src")]
    mod = importlib.import_module(f"test_{args.framework}")
    for seed in args.seeds:
        path = _raw_path(args.framework, seed)
        if path.exists() and not args.force:
            print(f"seed {seed}: {path} exists, skipped")
            continue
        worst, native, must_fail, devices = measure_seed(mod, seed)
        data = {
            "seed": seed,
            "devices": devices,
            "date": datetime.date.today().strftime("%d/%m/%Y"),
            "versions": _versions(args.framework),
            "options": mod.sf.get_options()._asdict(),
            "source_sha256": hashlib.sha256(
                pathlib.Path(mod.sf.__file__).read_bytes()
            ).hexdigest(),  # The measured code.
            "worst": _encode(worst),
            "native_accurate": _encode(native),
            "must_fail": _encode(must_fail),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=1) + "\n")
        tmp.replace(path)  # Atomic: an interrupted seed leaves no partial raw file.
        print(f"seed {seed}: wrote {path}", flush=True)


def cmd_merge(args) -> None:
    raws = []
    for seed in args.seeds:
        path = _raw_path(args.framework, seed)
        if not path.exists():
            sys.exit(f"missing raw file {path}: run the measure step for seed {seed}")
        raws.append(json.loads(path.read_text()))
        if "native_accurate" not in raws[-1]:
            sys.exit(f"{path} predates 30/10/2026 (native's errors merged into the library's): re-measure it")
    devices = {tuple(r["devices"]) for r in raws}
    if len(devices) != 1:
        sys.exit(f"raw files were measured on different devices: {devices}")
    options = {json.dumps(r.get("options"), sort_keys=True) for r in raws}
    if len(options) != 1:
        sys.exit(f"raw files were measured under different library options: {options}")
    sources = {r.get("source_sha256") for r in raws}
    if len(sources) != 1 or None in sources:
        sys.exit(f"raw files were measured on different (or unrecorded) library sources: {sources}")
    worst: dict[tuple, float] = defaultdict(float)
    must_fail: dict[tuple, float] = defaultdict(lambda: math.inf)
    native: dict[tuple, float] = defaultdict(float)
    for r in raws:
        for key, v in _decode(r["worst"]).items():
            worst[key] = max(worst[key], v)
        for key, v in _decode(r["native_accurate"]).items():
            native[key] = max(native[key], v)
        for key, v in _decode(r["must_fail"]).items():
            must_fail[key] = min(must_fail[key], v)
    tols = {key: _tolerance(v) for key, v in worst.items()}
    print(f"{'namespace':<11} {'regime':<16} {'metric':<26} {'format':<9} {'worst':>10} {'tolerance':>10}")
    for key, v in sorted(worst.items()):
        print(f"{key[0]:<11} {key[1]:<16} {key[2]:<26} {key[3]:<9} {v:>10.3g} {tols[key]:>10g}")
    over = {k: (v, tols.get(k, math.nan)) for k, v in sorted(native.items()) if not v <= tols.get(k, math.nan)}
    print(
        f"\nnative where it was taken to be accurate (reported, not enforced): over the library's tolerance on "
        f"{len(over)} of {len(native)} keys"
    )
    for key, (v, t) in over.items():
        print(f"  {key[0]:<11} {key[1]:<16} {key[2]:<26} {key[3]:<9} native {v:>10.3g}  tolerance {t:>8g}")
    print(
        f"\n{'namespace':<11} {'regime':<16} {'native must fail':<26} {'format':<9} {'min native':>10} {'tol':>8} {'ratio':>9}"
    )
    sys.path[:0] = [str(HERE), str(HERE.parent / "src")]
    mod = importlib.import_module(f"test_{args.framework}")
    weak, undetectable, excluded = [], [], []
    for key, v in sorted(must_fail.items()):
        t = tols.get(key, math.nan)
        ratio = v / t if t else math.inf
        kept = key[2] in mod.must_fail(key[1], key[3], key[0].split("-", 1)[1])  # Namespace "<framework>-<device>".
        print(
            f"{key[0]:<11} {key[1]:<16} {key[2]:<26} {key[3]:<9} {v:>10.3g} {t:>8g} {ratio:>9.3g}{'' if kept else '  (excluded)'}"
        )
        if not kept:
            excluded.append((key, ratio))
        elif not ratio > 1:
            undetectable.append(key)
        elif not ratio > WEAK:
            weak.append(key)
    bad = {k: v for k, v in worst.items() if not v <= args.cap}
    if excluded:
        print(f"\nexcluded by must_fail (not enforced; separation would hold where ratio > 1): {excluded}")
    if weak:
        print(f"\nWARNING: native must-fail errors within {WEAK:g}x of the tolerance: {weak}")
    if bad:
        print(f"\nREFUSING TO WRITE: observed errors over the cap {args.cap:g} (or non-finite): {bad}")
    if undetectable:
        print(f"\nREFUSING TO WRITE: native must-fail errors within the tolerance (check cannot fail): {undetectable}")
    if bad or undetectable:
        sys.exit(1)
    if args.write:
        out = HERE / f"tolerances_{args.framework}.json"
        data: dict = {"tolerances": [[*key, tol] for key, tol in sorted(tols.items())]}
        data["provenance"] = {
            "seeds": args.seeds,
            "devices": raws[0]["devices"],
            "date": datetime.date.today().strftime("%d/%m/%Y"),
            "rule": RULE,
            "options": raws[0].get("options"),
            "source_sha256": raws[0]["source_sha256"],
            "versions": raws[0]["versions"],
        }
        out.write_text(json.dumps(data, indent=1) + "\n")
        print(f"\nwrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("measure", help="write one raw error file per seed")
    m.add_argument("--force", action="store_true", help="re-measure seeds whose raw file exists")
    g = sub.add_parser("merge", help="merge raw files into tolerances")
    g.add_argument("--cap", type=float, default=2048.0, help="largest acceptable observed fixed error")
    g.add_argument("--write", action="store_true", help="write tests/tolerances_<framework>.json")
    for p in (m, g):
        p.add_argument("--framework", choices=["jax", "torch"], required=True)
        p.add_argument("--seeds", type=int, nargs="+", required=True)
    args = ap.parse_args()
    if args.cmd == "measure":
        cmd_measure(args)
    else:
        cmd_merge(args)


if __name__ == "__main__":
    main()
