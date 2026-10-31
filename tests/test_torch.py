"""Tests of softmaxfix.torch: accuracy against the mpmath oracle, against native torch.nn.functional and against the
paper's reference code (ported); native parity across options; the precision policy; autograd and torch.func
transforms; edge cases."""

import itertools
import threading
import warnings
from types import SimpleNamespace

import cases
import numpy as np
import oracle
import paper_torch
import pytest
import torch
import torch.nn.functional as F
from tolerances import tol
from torch.func import grad
from torch.func import hessian
from torch.func import jacfwd
from torch.func import jacrev
from torch.func import vmap

import softmaxfix.torch as sf

FMTS = ("float16", "bfloat16", "float32", "float64")
DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32, "float64": torch.float64}
HAS_CUDA = torch.cuda.is_available()
DEVICES = [pytest.param("cpu", id="cpu")] + (
    [pytest.param("cuda", id="cuda", marks=pytest.mark.gpu)] if HAS_CUDA else []
)
GAPS = cases.GAPS
EDGE_PAIRS = cases.EDGE_PAIRS

HARD_REGIMES = ("safe", "wide", "flat", "cliff", "margin", "margin_bigk", "wrong")
SOFT_REGIMES = ("teacher", "offset", "outlier", "pairs")
FIXED_REGIMES = HARD_REGIMES + SOFT_REGIMES
# Regime → metrics where native was taken to be accurate (None: all of them). Since 30/10/2026 its errors there are
# measured at calibration and reported, not used: tolerances come from the library alone, native being the baseline
# the comparison tests beat, and the known-bad of test_native_fails_where_the_bugs_bite.
NATIVE_ACCURATE = {
    "safe": None,
    "flat": None,
    "wrong": ("ce_int_value", "ce_int_grad", "ce_int_grad_sum", "ce_onehot_value", "ce_onehot_grad", "softmax"),
}
_COLLAPSE = ("log_softmax", "ce_int_value", "ce_int_grad", "ce_onehot_grad", "softmax_vjp_argmax", "ce_int_grad_sum")
_CANCELLATION = ("soft_grad_student", "kl_value", "kl_grad_student")
# The cliff regime stops at margin 40 (24 for 16-bit formats), where rounding logit differences still costs far
# less than the 100 % error of a collapsed value; the full margin sweeps test the fixed code up to underflow.
MUST_FAIL = {"cliff": _COLLAPSE, "teacher": _CANCELLATION, "offset": _CANCELLATION}
# Metrics computed by the paper's reference code as well (it has no probability-label CE and no KL).
PAPER_METRICS = (
    "log_softmax",
    "softmax",
    "ce_int_value",
    "ce_int_grad",
    "ce_int_grad_sum",
    "softmax_vjp_argmax",
    "softmax_vjp",
    "log_softmax_vjp",
    "soft_ce_value",
    "soft_grad_student",
    "soft_grad_teacher",
    "kl_value",  # Against KL = CE − H(p) composed from the paper's functions (the paper has no KL).
    "kl_grad_student",
    "kl_grad_teacher",
)
HESSIAN_REGIMES = ("random", "near", "masked", "margin")
HESSIAN_FMTS = ("float32", "float64")


def must_fail(regime: str, fmt: str, device: str) -> tuple[str, ...]:
    """Metrics native must fail in ``regime``, minus those the format cannot resolve (each exclusion explained)."""
    metrics = MUST_FAIL[regime]
    # The float16 exclusions of fp16 cliff Σg (CPU) and of the teacher / offset student gradients (CPU and CUDA) were
    # dropped on 29/10/2026. Graded at float32's flush threshold, native separates there: on the CPU at seeds 1, 2,
    # 20261026 (Σg 2.3e3–3.0e3 ε against ≤ 0.66; student gradients 3.2e3 ε to inf against ≤ 0.46); on CUDA, for the
    # student gradients, over the calibration seeds 1–10 (tests/calibration/: native's smallest error over the
    # tolerance 530 offset, about 9e307 teacher, a non-finite native).
    if fmt == "float16" and regime == "cliff" and device == "cuda":
        # Kept (maintainer, 29/10/2026). Native computes fp16 in float32 on CUDA: its cliff (16) lies beyond fp16's
        # underflow (9.7), where a collapsed value differs from the true one by little. Measured over the calibration
        # seeds 1–10 (tests/calibration/), native's smallest error over the tolerance: Σg 0.91, value and log_softmax
        # 0.67 (no separation), the gradients 2.38 (thin), the argmax VJP 1.0e3.
        metrics = ()
    return metrics


def tol_key(device: str) -> str:
    """Tolerances are kept per device, so that a device missing at calibration time fails loudly here."""
    return f"torch-{device}"


IMPLS = {
    "fixed": SimpleNamespace(
        log_softmax=lambda z: sf.log_softmax(z, -1),
        softmax=lambda z: sf.softmax(z, -1),
        ce_int=lambda z, t: sf.cross_entropy(z, t, reduction="none"),
        ce_prob=lambda z, p: sf.cross_entropy(z, p, reduction="none"),
        soft_ce=lambda zs, zt: sf.soft_target_cross_entropy(zs, zt, reduction="none"),
        kl=lambda zs, zt: sf.kl_div_with_logits(zs, zt, reduction="none"),
    ),
    "native": SimpleNamespace(
        log_softmax=lambda z: F.log_softmax(z, -1),
        softmax=lambda z: F.softmax(z, -1),
        ce_int=lambda z, t: F.cross_entropy(z, t, reduction="none"),
        ce_prob=lambda z, p: F.cross_entropy(z, p, reduction="none"),
        soft_ce=lambda zs, zt: F.cross_entropy(zs, F.softmax(zt, -1), reduction="none"),
        kl=lambda zs, zt: F.kl_div(F.log_softmax(zs, -1), F.softmax(zt, -1), reduction="none").sum(-1),
    ),
    "paper": SimpleNamespace(
        log_softmax=paper_torch.log_softmax,
        softmax=paper_torch.softmax,
        ce_int=paper_torch.cross_entropy,
        ce_prob=None,
        soft_ce=paper_torch.soft_target_cross_entropy,
        kl=paper_torch.kl_divergence,
    ),
    "paper_naive": SimpleNamespace(
        log_softmax=paper_torch.naive_log_softmax,
        softmax=paper_torch.naive_softmax,
        ce_int=paper_torch.naive_cross_entropy,
        ce_prob=None,
        soft_ce=paper_torch.naive_soft_target_cross_entropy,
        kl=paper_torch.naive_kl_divergence,
    ),
}


def put(a, fmt: str, device: str, requires_grad: bool = False) -> torch.Tensor:
    t = torch.as_tensor(np.asarray(a, np.float64), dtype=torch.float64).to(device=device, dtype=DTYPES[fmt])
    return t.requires_grad_(requires_grad)


def f64(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().double().numpy()


def _one_hot(idx: np.ndarray, k: int) -> np.ndarray:
    out = np.zeros((len(idx), k))
    out[np.arange(len(idx)), idx] = 1.0
    return out


def _vjp(fn, inputs: tuple[torch.Tensor, ...], v: torch.Tensor | None = None):
    """(fn(*inputs), vᵀ J for each input), with v = 1 by default."""
    out = fn(*inputs)
    return out, torch.autograd.grad(out, inputs, torch.ones_like(out) if v is None else v)


def _rowmax(e: np.ndarray) -> np.ndarray:
    e = np.nan_to_num(np.asarray(e, np.float64), nan=np.inf)
    return e.reshape(e.shape[0], -1).max(axis=1) if e.ndim > 1 else e


# ------------------------------------------------------------------------------------------------ measurement
def _grad_sum_error(g: np.ndarray, fmt: str, ftz: np.ndarray) -> np.ndarray:
    """|Σ_k g_k| per row in units of eps · max_k |g_k|, after the flush-to-zero allowance."""
    s = np.maximum(np.abs(g.sum(axis=1)) - ftz[:, 0], 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = s / (np.abs(g).max(axis=1) * oracle.eps(fmt))
    return np.where(s == 0, 0.0, out)


def _result_allowance(fmt: str, ref) -> np.ndarray:
    """Flushed and float16-subnormal-rounded results of a derivative, as they reach a sum over its row."""
    return oracle.subnormal_mass(fmt, ref) + oracle.underflow_rounding(fmt, ref, summed=True)


def _measure_hard(f, z64, t, fmt, device, gen) -> dict[str, np.ndarray]:
    z = put(z64, fmt, device, requires_grad=True)
    zr = f64(z)
    tt = torch.as_tensor(t, dtype=torch.long, device=device)
    k = zr.shape[1]
    e_t = _one_hot(t, k)
    e_max = _one_hot(np.argmax(zr, axis=1), k)
    v = put(gen.normal(size=zr.shape), fmt, device)
    vr = f64(v)
    labels = put(0.9 * e_t + 0.1 * gen.dirichlet(np.ones(k), size=len(t)), fmt, device)  # Smoothed soft labels.
    lr = f64(labels)
    ftz = oracle.flushable(fmt, oracle.softmax(zr)[0])

    def ue(c, r):
        return oracle.ulp_error(f64(c), r, fmt, ftz=ftz)

    # Derivatives: entries of the result below the normal range are flushed too (and feed its zero-sum pivot), or, in
    # float16, rounded in its subnormal range.
    def ued(c, r):
        return oracle.ulp_error(
            f64(c), r, fmt, ftz=ftz + oracle.subnormal_mass(fmt, r) + oracle.underflow_rounding(fmt, r)
        )

    def ned(c, r):
        return oracle.normwise_error(
            f64(c), r, fmt, ftz=ftz + oracle.subnormal_mass(fmt, r) + oracle.underflow_rounding(fmt, r)
        )

    out = {"log_softmax": ue(f.log_softmax(z), oracle.log_softmax(zr)), "softmax": ue(f.softmax(z), oracle.softmax(zr))}
    loss, (g,) = _vjp(lambda z: f.ce_int(z, tt), (z,))
    out["ce_int_value"] = ue(loss, oracle.ce_int(zr, t))
    out["ce_int_grad"] = ued(g, oracle.grad_ce_int(zr, t))
    out["ce_int_grad_sum"] = _grad_sum_error(f64(g), fmt, ftz + _result_allowance(fmt, oracle.grad_ce_int(zr, t)))
    if f.ce_prob is not None:
        loss, (g,) = _vjp(lambda z: f.ce_prob(z, put(e_t, fmt, device)), (z,))
        out["ce_onehot_value"] = ue(loss, oracle.ce_int(zr, t))
        out["ce_onehot_grad"] = ued(g, oracle.grad_ce_int(zr, t))
        loss, (g,) = _vjp(lambda z: f.ce_prob(z, labels), (z,))
        out["ce_soft_label_value"] = ue(loss, oracle.ce_prob(zr, lr))
        out["ce_soft_label_grad"] = ned(g, oracle.grad_ce_prob(zr, lr))
    out["softmax_vjp_argmax"] = ued(_vjp(f.softmax, (z,), put(e_max, fmt, device))[1][0], oracle.softmax_vjp(zr, e_max))
    out["softmax_vjp"] = ned(_vjp(f.softmax, (z,), v)[1][0], oracle.softmax_vjp(zr, vr))
    out["log_softmax_vjp"] = ned(_vjp(f.log_softmax, (z,), v)[1][0], oracle.log_softmax_vjp(zr, vr))
    return {name: _rowmax(e) for name, e in out.items()}


def _measure_soft(f, zs64, zt64, fmt, device) -> dict[str, np.ndarray]:
    zs, zt = put(zs64, fmt, device, True), put(zt64, fmt, device, True)
    zsr, ztr = f64(zs), f64(zt)
    ftz = oracle.flushable(fmt, oracle.softmax(zsr)[0], oracle.softmax(ztr)[0])
    out = {}
    for prefix, fn, ref_value, ref_teacher in [
        ("soft", f.soft_ce, oracle.soft_ce, oracle.grad_teacher_ce),
        ("kl", f.kl, oracle.kl, oracle.grad_teacher_kl),
    ]:
        if fn is None:
            continue
        val, (gs, gt) = _vjp(fn, (zs, zt))
        value_name = "soft_ce_value" if prefix == "soft" else "kl_value"
        out[value_name] = oracle.ulp_error(f64(val), ref_value(zsr, ztr), fmt, ftz=ftz)
        for side, g, ref in (("student", gs, oracle.grad_student(zsr, ztr)), ("teacher", gt, ref_teacher(zsr, ztr))):
            fz = ftz + oracle.subnormal_mass(fmt, ref) + oracle.underflow_rounding(fmt, ref)  # As in _measure_hard.
            out[f"{prefix}_grad_{side}"] = oracle.normwise_error(f64(g), ref, fmt, ftz=fz)
    return {name: _rowmax(e) for name, e in out.items()}


def measure_rows(impl: str, regime: str, fmt: str, device: str, seed: int) -> dict[str, np.ndarray]:
    """Per-row error of every metric of ``regime`` for implementation ``impl`` (fixed, native, paper or paper_naive)."""
    f = IMPLS[impl]
    gen = cases.rng(seed, "torch", regime, fmt)
    if regime == "safe":
        z = cases.random_logits(gen, 128, 8, 1.0)
        return _measure_hard(f, z, gen.integers(0, 8, len(z)), fmt, device, gen)
    if regime == "wide":
        z = np.concatenate([cases.random_logits(gen, 64, 8, s) for s in (10.0, 100.0)])
        z = np.concatenate([z, cases.random_logits(gen, 64, 8, 1.0, offset=1e3)])
        return _measure_hard(f, z, gen.integers(0, 8, len(z)), fmt, device, gen)
    if regime == "flat":
        return _measure_hard(f, *cases.flat(gen), fmt, device, gen)
    if regime == "cliff":
        top = 40.0 if fmt in ("float32", "float64") else 24.0
        return _measure_hard(f, *cases.margin_sweep(gen, fmt, n=64, max_margin=top), fmt, device, gen)
    if regime == "margin":
        return _measure_hard(f, *cases.margin_sweep(gen, fmt), fmt, device, gen)
    if regime == "margin_bigk":
        return _measure_hard(f, *cases.margin_sweep(gen, fmt, n=8, k=4096), fmt, device, gen)
    if regime == "wrong":
        return _measure_hard(f, *cases.wrong_class(gen), fmt, device, gen)
    if regime == "teacher":
        return _measure_soft(f, *cases.near_teacher(gen, GAPS), fmt, device)
    if regime == "offset":
        return _measure_soft(f, *cases.near_teacher_offset(gen), fmt, device)
    if regime == "outlier":
        return _measure_soft(f, *cases.outlier(gen), fmt, device)
    if regime == "pairs":
        return _measure_soft(f, gen.normal(size=(96, 8)) * 3, gen.normal(size=(96, 8)) * 3, fmt, device)
    raise ValueError(regime)


def measure(impl: str, regime: str, fmt: str, device: str, seed: int) -> dict[str, float]:
    """Worst error of every metric of ``regime`` for implementation ``impl``."""
    return {k: float(v.max()) for k, v in measure_rows(impl, regime, fmt, device, seed).items()}


def _over(errs: dict[str, float], regime: str, fmt: str, device: str) -> dict[str, float]:
    return {k: v for k, v in errs.items() if not v <= tol(tol_key(device), regime, k, fmt)}


# --------------------------------------------------------------------------------------- accuracy vs oracle
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("fmt", FMTS)
@pytest.mark.parametrize("regime", FIXED_REGIMES)
def test_fixed_accuracy(regime, fmt, device, master_seed):
    errs = measure("fixed", regime, fmt, device, master_seed)
    assert not _over(errs, regime, fmt, device), f"over tolerance: {_over(errs, regime, fmt, device)}"


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("fmt", FMTS)
@pytest.mark.parametrize("regime", list(MUST_FAIL))
def test_native_fails_where_the_bugs_bite(regime, fmt, device, master_seed):
    metrics = must_fail(regime, fmt, device)
    if not metrics:
        pytest.skip(f"{fmt} on {device} cannot resolve the {regime} failure (see must_fail)")
    errs = measure("native", regime, fmt, device, master_seed)
    undetected = [k for k in metrics if errs[k] <= tol(tol_key(device), regime, k, fmt)]
    assert not undetected, f"native within tolerance on {undetected}: these metrics cannot detect the bug"


# Compared over the rows of a regime, not row by row: where both errors are set by the rounding of logit differences
# (conditioning, hundreds of ulps at large margins), the reference's extra roundings cancel ours on some rows and add
# to it on others, so either code wins single rows by luck. A regression is a worse worst case or a worse mean.
RELATIVE, SLACK = 1.25, 4.0


def _finite_mean(e: np.ndarray) -> float:
    return float(np.mean(e[np.isfinite(e)])) if np.isfinite(e).any() else np.inf


def _less_accurate(ours: dict, ref: dict, metrics, quantile: float | None = None) -> dict:
    """Metrics on which ``ours`` has a worse worst case or a worse mean than ``ref``, with both numbers. With
    ``quantile`` (the pooled regimes, ``_worst_quantile``), that quantile of the rows stands for the worst case, and a
    metric with more non-finite rows than the reference's fails outright (a quantile could hide them)."""
    worse = {}
    for m in metrics:
        if m in ours and m in ref:
            o, r = ours[m], ref[m]
            if quantile is None:
                o_w, r_w, more_nonfinite = float(o.max()), float(r.max()), False
            else:
                o_w, r_w = float(np.quantile(o, quantile)), float(np.quantile(r, quantile))
                more_nonfinite = int(np.sum(~np.isfinite(o))) > int(np.sum(~np.isfinite(r)))
            o_mean, r_mean = _finite_mean(o), _finite_mean(r)
            if more_nonfinite or o_w > RELATIVE * r_w + SLACK or o_mean > RELATIVE * r_mean + 1:
                worse[m] = {"max" if quantile is None else f"q{quantile:g}": (o_w, r_w), "mean": (o_mean, r_mean)}
    return worse


# Regimes whose rows are too few for a stable per-seed comparison: fp32 ``margin_bigk`` has 8 rows of 4 096 classes, its
# worst cases are heavy-tailed, and the per-seed rule fails known-good code on 1–2 seeds in 20 (20-seed data,
# 27/10/2026; VALIDATION.md). The comparison tests pool POOLED[regime] independent draws there, the seed's own rows
# first.
POOLED = {"margin_bigk": 4}
# The pooled regimes' worst-case clause compares the rows' 90th percentile, not their maximum (29/10/2026). On the
# pooled false-alarm data (30 seeds × 8 draws, pools of 4; VALIDATION.md), known-good comparisons
# failed 135 of 4 800 pools under the maximum and 20 under the quantile; known regressions failed 1 200 of 1 440 pools
# under both rules, the rest being torch's paper port in 16-bit, not worse there.
POOLED_QUANTILE = 0.9


def _worst_quantile(regime: str) -> float | None:
    """The quantile that stands for the worst case in ``regime``'s comparisons (None: the maximum)."""
    return POOLED_QUANTILE if regime in POOLED else None


POOL_STRIDE = 1_000_003  # Draw i uses seed + i·stride: for seeds below the stride, no two seeds share a draw.


def compare_rows(impl: str, regime: str, fmt: str, device: str, seed: int) -> dict[str, np.ndarray]:
    """``measure_rows`` for the comparison tests: pooled over POOLED[regime] draws, one draw elsewhere."""
    draws = [measure_rows(impl, regime, fmt, device, seed + i * POOL_STRIDE) for i in range(POOLED.get(regime, 1))]
    return {k: np.concatenate([d[k] for d in draws]) for k in draws[0]}


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("fmt", FMTS)
@pytest.mark.parametrize("regime", FIXED_REGIMES)
def test_not_less_accurate_than_the_paper(regime, fmt, device, master_seed):
    ours = compare_rows("fixed", regime, fmt, device, master_seed)
    paper = compare_rows("paper", regime, fmt, device, master_seed)
    worse = _less_accurate(ours, paper, PAPER_METRICS, _worst_quantile(regime))
    assert not worse, f"metrics where the paper's code is more accurate (ours, paper): {worse}"


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("fmt", FMTS)
@pytest.mark.parametrize("regime", FIXED_REGIMES)
def test_not_less_accurate_than_the_naive_port(regime, fmt, device, master_seed):
    """Against the same port computing in the input dtype as written (no float32 upcast for half precision)."""
    ours = compare_rows("fixed", regime, fmt, device, master_seed)
    naive = compare_rows("paper_naive", regime, fmt, device, master_seed)
    worse = _less_accurate(ours, naive, PAPER_METRICS, _worst_quantile(regime))
    assert not worse, f"metrics where the naive port of the paper's code is more accurate (ours, naive): {worse}"


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("fmt", FMTS)
@pytest.mark.parametrize("regime", FIXED_REGIMES)
def test_not_less_accurate_than_native(regime, fmt, device, master_seed):
    """In every regime, on every device: on CUDA, native computes half-precision inputs in float32 too, so a fixed
    path that computed them in the input dtype would fail here."""
    ours = compare_rows("fixed", regime, fmt, device, master_seed)
    native = compare_rows("native", regime, fmt, device, master_seed)
    worse = _less_accurate(ours, native, ours, _worst_quantile(regime))
    assert not worse, f"metrics where native is more accurate (ours, native): {worse}"


# --------------------------------------------------------------------------------------- native parity: API
def _parity(a: torch.Tensor, b: torch.Tensor, rtol: float) -> None:
    assert a.shape == b.shape and a.dtype == b.dtype, (a.shape, b.shape, a.dtype, b.dtype)
    np.testing.assert_allclose(f64(a), f64(b), rtol=rtol, atol=rtol)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "shape_dim", [((5,), 0), ((4, 5), 1), ((4, 5), 0), ((3, 4, 5), -1), ((3, 4, 5), 1), ((2, 3, 4, 5), 2)]
)
def test_softmax_family_parity(shape_dim, dtype, device, master_seed):
    shape, dim = shape_dim
    gen = cases.rng(master_seed, "torch-parity-softmax", str(shape_dim), str(dtype), device)
    rtol = 2e-6 if dtype == torch.float32 else 1e-14
    x = torch.as_tensor(gen.normal(size=shape) * 3, dtype=dtype, device=device)
    xt = x.transpose(0, -1) if x.dim() > 1 else x  # Non-contiguous view.
    w = torch.as_tensor(gen.normal(size=shape), dtype=dtype, device=device)
    for ours, ref in [(sf.log_softmax, F.log_softmax), (sf.softmax, F.softmax), (sf.softmin, F.softmin)]:
        for inp, ww in [(x, w), (xt, w.transpose(0, -1) if w.dim() > 1 else w)]:
            a = inp.detach().clone().requires_grad_()
            b = inp.detach().clone().requires_grad_()
            ya, yb = ours(a, dim), ref(b, dim)
            _parity(ya, yb, rtol)
            (ga,) = torch.autograd.grad((ya * ww).sum(), a)
            (gb,) = torch.autograd.grad((yb * ww).sum(), b)
            _parity(ga, gb, rtol * 10)
        _parity(ours(x, dim, dtype=torch.float64), ref(x, dim, dtype=torch.float64), 1e-14)


def test_legacy_dim_none_rule_and_warning():
    for shape in [(5,), (3, 5), (2, 3, 5), (2, 3, 4, 5)]:
        x = torch.randn(shape, generator=torch.Generator().manual_seed(len(shape)), dtype=torch.float64)
        for ours, ref in [(sf.log_softmax, F.log_softmax), (sf.softmax, F.softmax), (sf.softmin, F.softmin)]:
            with warnings.catch_warnings(record=True) as wa:
                warnings.simplefilter("always")
                a = ours(x)
            with warnings.catch_warnings(record=True) as wb:
                warnings.simplefilter("always")
                b = ref(x)
            _parity(a, b, 1e-14)
            assert [str(w.message) for w in wa] == [str(w.message) for w in wb]
            assert [w.filename for w in wa] == [w.filename for w in wb] == [__file__]


def test_zero_dimensional_input_matches_native():
    x = torch.tensor(2.0, requires_grad=True)
    for ours, ref in [(sf.log_softmax, F.log_softmax), (sf.softmax, F.softmax), (sf.softmin, F.softmin)]:
        a, b = ours(x, 0), ref(x, 0)
        _parity(a, b, 0.0)
        assert torch.equal(torch.autograd.grad(a, x)[0], torch.autograd.grad(b, x)[0])
        for dim in (1, -2):
            for fn in (ours, ref):
                with pytest.raises(IndexError):
                    fn(x, dim)


def _ce_grid():
    shapes = [(5,), (4, 5), (3, 5, 2, 3)]
    return itertools.product(shapes, [False, True], ["none", "sum", "mean"], [0.0, 0.1], ["idx", "prob"], [False, True])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cross_entropy_option_grid_parity(dtype, device, master_seed):
    gen = cases.rng(master_seed, "torch-ce-grid", str(dtype), device)
    rtol = 1e-5 if dtype == torch.float32 else 1e-13
    n = 0
    for shape, use_w, red, ls, kind, ign in _ce_grid():
        if kind == "prob" and ign:
            continue
        c = shape[0] if len(shape) == 1 else shape[1]
        x = torch.as_tensor(gen.normal(size=shape) * 2, dtype=dtype, device=device)
        w = torch.as_tensor(gen.random(c) + 0.5, dtype=dtype, device=device) if use_w else None
        if kind == "idx":
            tshape = () if len(shape) == 1 else (shape[0],) + shape[2:]
            y = torch.as_tensor(gen.integers(0, c, tshape), dtype=torch.long, device=device)
            if ign and y.dim() > 0:
                y.view(-1)[0] = -100
        else:
            y = torch.softmax(
                torch.as_tensor(gen.normal(size=shape), dtype=dtype, device=device), 0 if len(shape) == 1 else 1
            )
        a = x.clone().requires_grad_()
        b = x.clone().requires_grad_()
        la = sf.cross_entropy(a, y, weight=w, reduction=red, label_smoothing=ls)
        lb = F.cross_entropy(b, y, weight=w, reduction=red, label_smoothing=ls)
        _parity(la, lb, rtol)
        (ga,) = torch.autograd.grad(la.sum(), a)
        (gb,) = torch.autograd.grad(lb.sum(), b)
        _parity(ga, gb, rtol)
        n += 1
    assert n == 108


@pytest.mark.parametrize("device", DEVICES)
def test_cross_entropy_legacy_arguments_and_errors(device):
    x = torch.randn(6, 4, device=device, dtype=torch.float64)
    y = torch.tensor([0, 1, 2, 3, -100, 1], device=device)
    for size_average, reduce in [(None, False), (False, None), (True, True), (False, True)]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _parity(
                sf.cross_entropy(x, y, size_average=size_average, reduce=reduce),
                F.cross_entropy(x, y, size_average=size_average, reduce=reduce),
                1e-14,
            )
    all_ignored = torch.full((6,), -100, device=device)
    assert torch.isnan(sf.cross_entropy(x, all_ignored)) and torch.isnan(F.cross_entropy(x, all_ignored))
    probs = torch.softmax(torch.randn(6, 4, device=device, dtype=torch.float64), 1)
    for fn in (sf.cross_entropy, F.cross_entropy):
        with pytest.raises(RuntimeError):
            fn(x, probs, ignore_index=1)
        with pytest.raises(RuntimeError):
            fn(x, probs.long())
        with pytest.raises(RuntimeError):
            fn(x, y, label_smoothing=1.5)
    if device == "cpu":  # A device-side assert would poison the CUDA context for the rest of the session.
        for fn in (sf.cross_entropy, F.cross_entropy):
            with pytest.raises((IndexError, RuntimeError)):
                fn(x, torch.tensor([0, 1, 2, 3, 4, 1]))


@pytest.mark.parametrize("device", DEVICES)
def test_module_wrappers(device):
    w = torch.tensor([1.0, 2.0, 0.5, 1.5], device=device)
    ours = sf.CrossEntropyLoss(weight=w, ignore_index=2, reduction="sum", label_smoothing=0.05).to(device)
    ref = torch.nn.CrossEntropyLoss(weight=w, ignore_index=2, reduction="sum", label_smoothing=0.05).to(device)
    assert isinstance(ours, torch.nn.CrossEntropyLoss) and repr(ours) == repr(ref)
    assert ours.state_dict().keys() == ref.state_dict().keys()
    ours.load_state_dict(ref.state_dict())
    x = torch.randn(5, 4, device=device)
    y = torch.tensor([0, 1, 2, 3, 1], device=device)
    _parity(ours(x, y), ref(x, y), 1e-6)
    for cls, ref_cls in [
        (sf.LogSoftmax, torch.nn.LogSoftmax),
        (sf.Softmax, torch.nn.Softmax),
        (sf.Softmin, torch.nn.Softmin),
    ]:
        m, r = cls(dim=1), ref_cls(dim=1)
        assert isinstance(m, ref_cls) and repr(m) == repr(r)
        _parity(m(x), r(x), 2e-6)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("shape", [(6,), (4, 6), (3, 6, 2, 2)])
@pytest.mark.parametrize("reduction", ["none", "mean", "sum", "batchmean"])
def test_soft_target_losses_parity(reduction, shape, device, master_seed):
    gen = cases.rng(master_seed, "torch-parity-soft", str(shape), reduction, device)
    zs = torch.as_tensor(gen.normal(size=shape) * 2, dtype=torch.float64, device=device)
    zt = torch.as_tensor(gen.normal(size=shape) * 2, dtype=torch.float64, device=device)
    cdim = 0 if len(shape) == 1 else 1

    def ref_ce(a, b):
        return F.cross_entropy(a, F.softmax(b, cdim), reduction=reduction)

    def ref_kl(a, b):
        per = F.kl_div(F.log_softmax(a, cdim), F.softmax(b, cdim), reduction="none").sum(cdim)
        return {
            "none": per,
            "mean": per.mean(),
            "sum": per.sum(),
            "batchmean": per.sum() / (a.shape[0] if a.dim() > 1 else 1),  # 1-D: one distribution, divided by 1.
        }[reduction]

    pairs = [(sf.kl_div_with_logits, ref_kl)]
    if reduction != "batchmean":
        pairs.append((sf.soft_target_cross_entropy, ref_ce))
    else:
        with pytest.raises(ValueError):
            sf.soft_target_cross_entropy(zs, zt, reduction="batchmean")
    for ours, ref in pairs:
        a1, a2 = zs.clone().requires_grad_(), zt.clone().requires_grad_()
        b1, b2 = zs.clone().requires_grad_(), zt.clone().requires_grad_()
        la, lb = ours(a1, a2, reduction=reduction), ref(b1, b2)
        _parity(la, lb, 1e-12)
        wa = torch.as_tensor(gen.normal(size=la.shape), dtype=torch.float64, device=device)
        ga = torch.autograd.grad((la * wa).sum(), (a1, a2))
        gb = torch.autograd.grad((lb * wa).sum(), (b1, b2))
        for x, y in zip(ga, gb):
            _parity(x, y, 1e-11)
    with pytest.raises(ValueError):
        sf.kl_div_with_logits(zs, zt[..., :-1] if zs.dim() > 1 else zt[:-1])


# ---------------------------------------------------------------------------------------- precision policy
def test_compute_dtype_is_float32_for_half_on_every_device():
    for device in ["cpu"] + (["cuda"] if HAS_CUDA else []):
        for dt in (torch.float16, torch.bfloat16):
            assert sf._compute_dtype(torch.zeros(1, dtype=dt, device=device)) == torch.float32
        for dt in (torch.float32, torch.float64):
            assert sf._compute_dtype(torch.zeros(1, dtype=dt, device=device)) == dt


def _policy_fns(gen, device):
    """(name, function, differentiable inputs, labels) of every function whose value is one fixed-path kernel."""
    z = torch.as_tensor(gen.normal(size=(16, 7)) * 4, dtype=torch.float64, device=device)
    z[0, 0] += 12.0  # One confident row.
    zt = z + torch.as_tensor(gen.normal(size=(16, 7)) * 1e-2, dtype=torch.float64, device=device)
    t = torch.as_tensor(gen.integers(0, 7, 16), device=device)
    return [
        ("log_softmax", lambda a: sf.log_softmax(a, 1), (z,)),
        ("softmax", lambda a: sf.softmax(a, 1), (z,)),
        ("softmin", lambda a: sf.softmin(a, 1), (z,)),
        ("ce_int", lambda a: sf.cross_entropy(a, t, reduction="none"), (z,)),
        ("soft_ce", lambda a, b: sf.soft_target_cross_entropy(a, b, reduction="none"), (z, zt)),
        ("kl", lambda a, b: sf.kl_div_with_logits(a, b, reduction="none"), (z, zt)),
    ]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("fmt", ["float16", "bfloat16"])
def test_half_precision_is_float32_rounded_once(fmt, device, master_seed):
    """Half-precision values are bitwise the float32 computation on the same inputs, rounded once. Gradients are within
    one ulp of it: backward recomputes from the saved half input (in float32), not from the float32 output."""
    for name, fn, args in _policy_fns(cases.rng(master_seed, "torch-half-policy", fmt, device), device):
        half = tuple(a.to(DTYPES[fmt]).requires_grad_() for a in args)
        single = tuple(a.detach().float().requires_grad_() for a in half)
        y, g = _vjp(fn, half)
        y32, g32 = _vjp(fn, single)
        assert y.dtype == DTYPES[fmt] and all(gi.dtype == DTYPES[fmt] for gi in g), name
        assert torch.equal(y, y32.to(DTYPES[fmt])), f"{name}: value is not float32 rounded once"
        for gi, gr in zip(g, g32):
            err = np.abs(f64(gi) - f64(gr.to(DTYPES[fmt])))
            assert np.all(err <= oracle.ulp(f64(gr), fmt)), f"{name}: gradient off by more than one ulp"


# ------------------------------------------------------------------------------------------------ transforms
def _fns(gen, device="cpu"):
    """(name, function, args, n): every public function; the first n args are differentiable, the rest labels."""
    z = torch.as_tensor(gen.normal(size=(4, 5)) * 2, dtype=torch.float64, device=device)
    z[0, 0] += 25.0  # One confident row, so both branches of the log_softmax rule are exercised.
    zt = torch.as_tensor(gen.normal(size=(4, 5)) * 2, dtype=torch.float64, device=device)
    labels = torch.as_tensor(gen.integers(0, 5, 4), device=device)
    probs = torch.as_tensor(gen.dirichlet(np.ones(5), 4), dtype=torch.float64, device=device)
    return [
        ("log_softmax", lambda a: sf.log_softmax(a, 1), (z,), 1),
        ("softmax", lambda a: sf.softmax(a, 1), (z,), 1),
        ("softmin", lambda a: sf.softmin(a, 1), (z,), 1),
        ("ce_int", lambda a, t: sf.cross_entropy(a, t, reduction="none"), (z, labels), 1),
        ("ce_prob", lambda a, b: sf.cross_entropy(a, b, reduction="none", label_smoothing=0.1), (z, probs), 2),
        ("soft_ce", lambda a, b: sf.soft_target_cross_entropy(a, b, reduction="none"), (z, zt), 2),
        ("kl", lambda a, b: sf.kl_div_with_logits(a, b, reduction="none"), (z, zt), 2),
    ]


@pytest.mark.parametrize("device", DEVICES)
def test_gradcheck_gradgradcheck_forward_ad(device, master_seed):
    for name, fn, args, n in _fns(cases.rng(master_seed, "torch-gradcheck"), device):
        args = tuple(a.clone().requires_grad_() if i < n else a for i, a in enumerate(args))
        assert torch.autograd.gradcheck(fn, args, check_forward_ad=True, check_batched_grad=True), name
        assert torch.autograd.gradgradcheck(fn, args, check_fwd_over_rev=True, check_batched_grad=True), name


@pytest.mark.parametrize("device", DEVICES)
def test_third_derivatives(device, master_seed):
    """gradgradcheck of x ↦ ∇⟨w, f(x)⟩ checks the third derivatives of f, backward and forward over backward."""
    gen = cases.rng(master_seed, "torch-third-order")
    for name, fn, args, n in _fns(gen, device):
        w = torch.as_tensor(gen.normal(size=fn(*args).shape), dtype=torch.float64, device=device)

        def gradient(*d, fn=fn, rest=args[n:], w=w):
            return torch.autograd.grad((fn(*d, *rest) * w).sum(), d, create_graph=True)

        d = tuple(a.clone().requires_grad_() for a in args[:n])
        assert torch.autograd.gradgradcheck(gradient, d, check_fwd_over_rev=True), name


def test_torch_func_transforms_consistent(master_seed):
    for name, fn, args, n in _fns(cases.rng(master_seed, "torch-func")):
        direct = fn(*args)

        def row(*a, fn=fn):
            return fn(*(x.unsqueeze(0) for x in a)).squeeze(0)

        np.testing.assert_allclose(f64(vmap(row)(*args)), f64(direct), rtol=1e-14, err_msg=name)
        for argnum in range(n):
            jf = vmap(jacfwd(row, argnum))(*args)
            jr = vmap(jacrev(row, argnum))(*args)
            np.testing.assert_allclose(f64(jf), f64(jr), rtol=1e-12, atol=1e-14, err_msg=f"{name} arg {argnum}")
        if direct.dim() == 1:  # One scalar per row: torch.func.grad applies and must agree with autograd.
            g = vmap(grad(row))(*args)
            a0 = args[0].clone().requires_grad_()
            (ref,) = torch.autograd.grad(fn(a0, *args[1:]).sum(), a0)
            np.testing.assert_allclose(f64(g), f64(ref), rtol=1e-13, atol=1e-15, err_msg=name)


def _row_hessian(fn, zs: torch.Tensor, zt: torch.Tensor | None = None) -> torch.Tensor:
    """Per-row Hessian w.r.t. the first argument, shape (n, K·K), via torch.func."""
    if zt is None:
        h = vmap(hessian(lambda r: fn(r.unsqueeze(0)).squeeze(0)))(zs)
    else:
        h = vmap(hessian(lambda r, s: fn(r.unsqueeze(0), s.unsqueeze(0)).squeeze(0)))(zs, zt)
    return h.reshape(h.shape[0], -1)


def measure_hessian(regime: str, fmt: str, seed: int, device: str = "cpu") -> dict[str, float]:
    """Normwise Hessian error (eps) against diag(q) − q qᵀ, per loss; inf if any entry is non-finite."""
    gen = cases.rng(seed, "torch-hessian", regime, fmt)
    ninf = -np.inf
    zt64 = gen.normal(size=(8, 5)) * 2
    if regime == "random":
        zs64 = gen.normal(size=(8, 5)) * 2
    elif regime == "near":
        zs64 = zt64 + 1e-6 * gen.normal(size=zt64.shape)
    elif regime == "masked":
        zs64 = zt64 + 0.3 * gen.normal(size=zt64.shape)
        zt64[0, 1] = ninf  # Teacher-only mask.
        zs64[1, 2] = zt64[1, 2] = ninf  # Masked in both.
        zs64[2, 1:] = zt64[2, 1:] = ninf  # Only one class left.
        zt64[3, 3] = -130.0  # Teacher probability underflows; student keeps mass.
    else:
        zs64 = gen.normal(size=(8, 5))
        zs64[:, 0] += np.linspace(10.0, 30.0, 8)
    zs, zt = put(zs64, fmt, device), put(zt64, fmt, device)
    ref = oracle.hessian(f64(zs))
    labels = torch.zeros(8, dtype=torch.long, device=device)
    hs = {
        "soft_ce": _row_hessian(lambda a, b: sf.soft_target_cross_entropy(a, b, reduction="none"), zs, zt),
        "kl": _row_hessian(lambda a, b: sf.kl_div_with_logits(a, b, reduction="none"), zs, zt),
    }
    if regime in ("random", "margin"):
        hs["ce_int"] = _row_hessian(lambda a: sf.cross_entropy(a, labels[: a.shape[0]], reduction="none"), zs)
    out = {}
    for name, h in hs.items():
        h = f64(h)
        out[name] = float(oracle.normwise_error(h, ref, fmt).max()) if np.all(np.isfinite(h)) else np.inf
    return out


@pytest.mark.parametrize("fmt", HESSIAN_FMTS)
@pytest.mark.parametrize("regime", HESSIAN_REGIMES)
def test_hessians_match_oracle(regime, fmt, master_seed):
    errs = measure_hessian(regime, fmt, master_seed)
    over = {k: v for k, v in errs.items() if not v <= tol(tol_key("cpu"), f"hessian_{regime}", k, fmt)}
    assert not over, f"normwise Hessian error over tolerance (eps): {over}"


@pytest.mark.slow
def test_torch_compile_matches_eager(master_seed):
    gen = cases.rng(master_seed, "torch-compile")
    x = torch.as_tensor(gen.normal(size=(8, 6)) * 5, dtype=torch.float32)
    y = torch.as_tensor(gen.integers(0, 6, 8))
    zt = torch.as_tensor(gen.normal(size=(8, 6)), dtype=torch.float32)

    def step(x):
        return sf.cross_entropy(x, y) + sf.kl_div_with_logits(x, zt) + sf.softmax(x, 1).square().sum()

    compiled = torch.compile(step, fullgraph=False)
    a, b = x.clone().requires_grad_(), x.clone().requires_grad_()
    la, lb = compiled(a), step(b)
    np.testing.assert_allclose(la.item(), lb.item(), rtol=1e-6)
    np.testing.assert_allclose(
        f64(torch.autograd.grad(la, a)[0]), f64(torch.autograd.grad(lb, b)[0]), rtol=1e-5, atol=1e-7
    )


# ------------------------------------------------------------------------------------------------ edge cases
def test_all_but_one_masked_has_zero_gradients():
    ninf = -float("inf")
    z = torch.tensor([[1.0, ninf, ninf], [ninf, 2.0, ninf]], requires_grad=True)
    _parity(sf.log_softmax(z.detach(), 1), F.log_softmax(z.detach(), 1), 0)
    out = sf.log_softmax(z, 1)
    (g,) = torch.autograd.grad(torch.where(out.isfinite(), out, 0).sum(), z)
    assert torch.all(g == 0)
    loss = sf.cross_entropy(z, torch.tensor([0, 1]), reduction="sum")
    (g,) = torch.autograd.grad(loss, z)
    assert loss.item() == 0.0 and torch.all(g == 0)
    (g,) = torch.autograd.grad(sf.softmax(z, 1)[:, 0].sum(), z)
    assert torch.all(g == 0)


def test_single_class_and_ties():
    z1 = torch.tensor([[2.5], [-1.0]], requires_grad=True)
    assert torch.all(sf.log_softmax(z1, 1) == 0)
    assert torch.all(torch.autograd.grad(sf.log_softmax(z1, 1).sum(), z1)[0] == 0)
    zt = torch.tensor([[3.0, 3.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.float64, requires_grad=True)
    _parity(sf.log_softmax(zt.detach(), 1), F.log_softmax(zt.detach(), 1), 1e-15)
    (g,) = torch.autograd.grad(sf.cross_entropy(zt, torch.tensor([1, 2]), reduction="sum"), zt)
    np.testing.assert_allclose(f64(g), f64(F.softmax(zt.detach(), 1)) - np.eye(3)[[1, 2]], atol=1e-15)


def test_nonfinite_inputs_match_native():
    ninf, inf, nan = -float("inf"), float("inf"), float("nan")
    z = torch.tensor([[ninf, ninf, ninf], [1.0, inf, 0.0], [inf, inf, 0.0], [1.0, nan, 0.0]])
    for ours, ref in [(sf.log_softmax, F.log_softmax), (sf.softmax, F.softmax)]:
        np.testing.assert_array_equal(f64(ours(z, 1)), f64(ref(z, 1)))
    for t in (torch.tensor([0, 1, 0, 2]), torch.tensor([1, 0, 2, 0])):
        a = f64(sf.cross_entropy(z, t, reduction="none"))
        b = f64(F.cross_entropy(z, t, reduction="none"))
        np.testing.assert_array_equal(np.isnan(a), np.isnan(b))


@pytest.mark.parametrize("shape", [(0, 5), (3, 0), (0,)])
def test_empty_inputs(shape):
    x = torch.zeros(shape)
    for ours, ref in [(sf.log_softmax, F.log_softmax), (sf.softmax, F.softmax), (sf.softmin, F.softmin)]:
        a, b = ours(x, -1), ref(x, -1)
        assert a.shape == b.shape and a.dtype == b.dtype
    if len(shape) == 2 and shape[0] == 0:
        y = torch.zeros(0, dtype=torch.long)
        for red in ("none", "sum", "mean"):
            np.testing.assert_array_equal(
                f64(sf.cross_entropy(x, y, reduction=red)), f64(F.cross_entropy(x, y, reduction=red))
            )
        assert sf.soft_target_cross_entropy(x, x, reduction="none").shape == (0,)


def measure_edge(case: str, fmt: str) -> dict[str, float]:
    """Errors on one hand-made soft-target row: value (ulps), student and teacher gradients (normwise eps)."""
    zs64, zt64 = (np.array([r]) for r in EDGE_PAIRS[case])
    zs, zt = put(zs64, fmt, "cpu", True), put(zt64, fmt, "cpu", True)
    zsr, ztr = f64(zs), f64(zt)
    out = {}
    for name, fn, ref_value, ref_teacher in [
        (
            "soft_ce",
            lambda a, b: sf.soft_target_cross_entropy(a, b, reduction="none"),
            oracle.soft_ce,
            oracle.grad_teacher_ce,
        ),
        ("kl", lambda a, b: sf.kl_div_with_logits(a, b, reduction="none"), oracle.kl, oracle.grad_teacher_kl),
    ]:
        val, (gs, gt) = _vjp(fn, (zs, zt))
        out[f"{name}_value"] = float(oracle.ulp_error(f64(val), ref_value(zsr, ztr), fmt).max())
        out[f"{name}_grad_student"] = float(oracle.normwise_error(f64(gs), oracle.grad_student(zsr, ztr), fmt).max())
        if np.all(np.isfinite(ref_teacher(zsr, ztr)[0])):
            out[f"{name}_grad_teacher"] = float(oracle.normwise_error(f64(gt), ref_teacher(zsr, ztr), fmt).max())
        h = _row_hessian(fn, zs.detach(), zt.detach())
        out[f"{name}_hessian_finite"] = 0.0 if bool(torch.isfinite(h).all()) else np.inf
    return out


@pytest.mark.parametrize("fmt", HESSIAN_FMTS)
@pytest.mark.parametrize("case", list(EDGE_PAIRS))
def test_soft_target_edge_rows(case, fmt):
    errs = measure_edge(case, fmt)
    over = {k: v for k, v in errs.items() if not v <= tol(tol_key("cpu"), "edge", k, fmt)}
    assert not over, f"over tolerance: {over}"


def test_log_mass_degenerate_rows():
    """The soft-target support-mass helper on degenerate rows: −inf for an empty mask or one on −inf logits only (also
    beside a +inf outside it), NaN with a +inf inside it or no finite logit, as the difference of two logsumexps."""
    inf, nan = np.inf, np.nan
    rows = [  # logits, mask, expected
        ([1.0, -inf, -inf], [0, 1, 1], -inf),
        ([1.0, inf], [1, 0], -inf),
        ([1.0, inf], [0, 1], nan),
        ([-inf, -inf], [1, 0], nan),
        ([1.0, 2.0], [0, 0], -inf),
        ([inf, inf, 1.0], [1, 0, 1], nan),
        ([2.0, -inf, 0.5, -inf], [0, 1, 0, 1], -inf),
    ]
    for z, m, expected in rows:
        got = f64(sf._log_mass(torch.tensor([z], dtype=torch.float64), torch.tensor([m], dtype=torch.bool)))
        np.testing.assert_array_equal(got.ravel(), [expected], err_msg=f"z={z} mask={m}")


def test_no_nans_under_detect_anomaly():
    rows = [v for k, v in EDGE_PAIRS.items() if len(v[0]) == 3 and k != "student masked"]  # Finite losses only.
    zs = torch.tensor([v[0] for v in rows], requires_grad=True)
    zt = torch.tensor([v[1] for v in rows], requires_grad=True)
    zm = torch.tensor([[1.0, -float("inf"), -float("inf")], [-float("inf"), 2.0, 0.5]], requires_grad=True)
    with torch.autograd.detect_anomaly():
        for fn in (sf.soft_target_cross_entropy, sf.kl_div_with_logits):
            val = fn(zs, zt, reduction="none")
            g = torch.autograd.grad(torch.where(val.isfinite(), val, 0).sum(), (zs, zt), create_graph=True)
            torch.autograd.grad(sum(gi.square().sum() for gi in g), (zs, zt))
        loss = sf.cross_entropy(zm, torch.tensor([0, 1]), reduction="sum")
        (g,) = torch.autograd.grad(loss, zm, create_graph=True)
        torch.autograd.grad(g.square().sum(), zm)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("fmt", FMTS)
def test_kl_finite_when_a_student_probability_underflows(fmt, device):
    """Support is read from the logits: an underflowed q_j is still mass (the first version returned +inf here)."""
    zs, zt = (put([r], fmt, device, True) for r in cases.UNDERFLOW_ROWS[fmt])
    val, (gs, gt) = _vjp(lambda a, b: sf.kl_div_with_logits(a, b, reduction="none"), (zs, zt))
    zsr, ztr = f64(zs), f64(zt)
    assert oracle.ulp_error(f64(val), oracle.kl(zsr, ztr), fmt).max() <= 2
    assert oracle.normwise_error(f64(gs), oracle.grad_student(zsr, ztr), fmt).max() <= 4
    assert oracle.normwise_error(f64(gt), oracle.grad_teacher_kl(zsr, ztr), fmt).max() <= 4


# ------------------------------------------------------------------------------------------------ properties
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("fmt", FMTS)
def test_exact_zero_at_equality(fmt, device, master_seed):
    z = put(cases.rng(master_seed, "torch-equality", fmt).normal(size=(32, 7)) * 3, fmt, device)
    for fn in (sf.soft_target_cross_entropy, sf.kl_div_with_logits):
        a, b = z.clone().requires_grad_(), z.clone().requires_grad_()
        val = fn(a, b, reduction="none")
        gs, gt = torch.autograd.grad(val, (a, b), torch.ones_like(val))
        assert torch.all(gs == 0), "student gradient must be exactly 0 at z_S = z_T"
        if fn is sf.kl_div_with_logits:
            assert torch.all(val == 0) and torch.all(gt == 0)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("fmt", FMTS)
def test_kl_nonnegative_and_softmax_normalised(fmt, device, master_seed):
    gen = cases.rng(master_seed, "torch-properties", fmt)
    zs, zt = cases.near_teacher(gen, GAPS + (1.0, 3.0))
    kl = f64(sf.kl_div_with_logits(put(zs, fmt, device), put(zt, fmt, device), reduction="none"))
    assert np.all(kl >= 0)
    q = f64(sf.softmax(put(zs * 10, fmt, device), 1))
    assert np.max(np.abs(q.sum(axis=1) - 1)) <= q.shape[1] * oracle.eps(fmt)


# ------------------------------------------------------------------------------------------------ the paper
def test_paper_example_softmax_collapse():
    z = torch.tensor([9.0, -9.0, -9.0], requires_grad=True)
    t = torch.tensor(0)
    np.testing.assert_allclose(f64(sf.log_softmax(z, 0)), [-3.046e-8, -18.0, -18.0], rtol=1e-3)
    np.testing.assert_allclose(
        f64(torch.autograd.grad(sf.softmax(z, 0)[0], z)[0]), [3.046e-8, -1.523e-8, -1.523e-8], rtol=1e-3
    )
    loss = sf.cross_entropy(z, t)
    np.testing.assert_allclose(loss.item(), 3.046e-8, rtol=1e-3)
    np.testing.assert_allclose(f64(torch.autograd.grad(loss, z)[0]), [-3.046e-8, 1.523e-8, 1.523e-8], rtol=1e-3)
    # The paper's "standard" outputs, from torch.nn.functional (same cliff as jax.nn / optax).
    np.testing.assert_array_equal(f64(F.log_softmax(z.detach(), 0)), [0.0, -18.0, -18.0])
    loss = F.cross_entropy(z, t)
    assert loss.item() == 0.0 and torch.autograd.grad(loss, z)[0][0].item() == 0.0


def test_paper_example_softmax_cancellation():
    z_t = torch.tensor([2.0, 1.0, 0.0])
    z_s = (z_t + torch.tensor([-1e-7, 1e-7, 0.0])).requires_grad_()
    g = f64(torch.autograd.grad(sf.soft_target_cross_entropy(z_s, z_t), z_s)[0])
    np.testing.assert_allclose(g, [-4.596e-8, 4.144e-8, 4.513e-9], rtol=1e-3)
    exact = oracle.grad_student(f64(z_s)[None], f64(z_t)[None])
    assert oracle.ulp_error(g[None], exact, "float32").max() <= 2
    # The standard gradient is rounding noise of the size of the gradient itself; its digits depend on the
    # implementation (torch 2.16 gives [-5.96e-8, 4.47e-8, 7.45e-9], JAX 0.12 the paper's printed ones), so only the
    # size of the error is pinned.
    std = f64(torch.autograd.grad(F.cross_entropy(z_s, F.softmax(z_t, 0)), z_s)[0])
    assert oracle.normwise_error(std[None], exact, "float32").max() > 1e6


def test_paper_reference_code_defects_are_absent_here():
    """The paper's own code: NaN gradients with one class left unmasked, zero Hessian of the soft-target loss.
    Kept as a record of why the derivative rules differ; ours is checked on the same inputs."""
    z = torch.tensor([1.0, -float("inf"), -float("inf")], requires_grad=True)
    assert torch.all(torch.isnan(torch.autograd.grad(paper_torch.log_softmax(z)[0], z)[0]))
    assert torch.all(torch.autograd.grad(sf.log_softmax(z, 0)[0], z)[0] == 0)
    zs, zt = torch.tensor([0.5, -0.2, 0.1]), torch.tensor([0.3, 0.0, -0.4])
    h_paper = torch.autograd.functional.hessian(lambda a: paper_torch.soft_target_cross_entropy(a, zt), zs)
    assert torch.all(h_paper == 0)
    h = torch.autograd.functional.hessian(lambda a: sf.soft_target_cross_entropy(a, zt), zs)
    assert oracle.normwise_error(f64(h).reshape(1, -1), oracle.hessian(f64(zs)[None]), "float32").max() <= 8


# ------------------------------------------------------------------------------------------------ options
OPTION_SETS = {
    "default": {},
    "rounded_shift": {"exact_shift": False},
    "paper_rule": {"rule": "paper"},
    "rounded_shift_paper_rule": {"exact_shift": False, "rule": "paper"},
    "flush_proof": {"rule": "flush_proof"},
    "rounded_shift_flush_proof": {"exact_shift": False, "rule": "flush_proof"},
}


def test_options_defaults_validation_and_scoping():
    """Defaults; invalid values are refused and change nothing; set_options returns the previous options; the
    context manager nests and restores the previous options on exit, exceptions included."""
    default = sf.Options(exact_shift=True, rule="accurate")
    assert sf.get_options() == default
    for bad, exc in (({"rule": "Paper"}, ValueError), ({"exact_shift": 1}, TypeError)):
        with pytest.raises(exc):
            sf.set_options(**bad)
        assert sf.get_options() == default
    previous = sf.set_options(rule="paper")
    try:
        assert previous == default
        assert sf.get_options() == default._replace(rule="paper")
        assert sf.set_options() == sf.get_options()  # None leaves an option unchanged.
    finally:
        sf.set_options(**previous._asdict())
    with sf.options(exact_shift=False) as outer:
        assert outer == sf.get_options() == default._replace(exact_shift=False)
        with sf.options(rule="paper"):
            assert sf.get_options() == sf.Options(exact_shift=False, rule="paper")
        assert sf.get_options() == outer
    with pytest.raises(RuntimeError, match="inside"), sf.options(exact_shift=False, rule="paper"):
        raise RuntimeError("inside")
    assert sf.get_options() == default


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("opts", list(OPTION_SETS))
def test_option_derivatives(opts, device, master_seed):
    """First and second derivatives of every public function, reverse and forward mode, under every option set, with
    tolerances that catch a 0.1 % error (gradcheck's defaults, rtol 1e-3, pass one)."""
    with sf.options(**OPTION_SETS[opts]):
        for name, fn, args, n in _fns(cases.rng(master_seed, "torch-options-gradcheck"), device):
            args = tuple(a.clone().requires_grad_() if i < n else a for i, a in enumerate(args))
            kw = {"atol": 1e-8, "rtol": 1e-6}
            assert torch.autograd.gradcheck(fn, args, check_forward_ad=True, check_batched_grad=True, **kw), name
            assert torch.autograd.gradgradcheck(fn, args, check_fwd_over_rev=True, **kw), name


# "flush_proof" alone changes nothing on these rows (none is scaled): test_flush_proof_backward_uses_its_forward_call.
@pytest.mark.parametrize("opts", [o for o in OPTION_SETS if o not in ("default", "flush_proof")])
def test_backward_uses_the_options_of_its_forward_call(opts, master_seed):
    """A graph built under some options and differentiated after their scope has ended keeps them: the rules are
    fixed at the forward call, not read when backward runs. The options must also change a derivative here."""
    gen = cases.rng(master_seed, "torch-options-backward")
    z = torch.as_tensor(gen.normal(size=(64, 16)) * 3, dtype=torch.float64)
    z[:32, 0] += 6.0  # Confident rows, where the rules differ.
    v = torch.as_tensor(gen.normal(size=z.shape), dtype=torch.float64)
    t = torch.as_tensor(gen.integers(0, 16, 64))
    fns = {
        "log_softmax": lambda a: (sf.log_softmax(a, 1) * v).sum(),
        "softmax": lambda a: (sf.softmax(a, 1) * v).sum(),
        "cross_entropy": lambda a: sf.cross_entropy(a, t),
        "soft_target_cross_entropy": lambda a: sf.soft_target_cross_entropy(a, z.flip(1)),
    }
    w = torch.as_tensor(gen.normal(size=z.shape), dtype=torch.float64)

    def derivatives(loss, a):  # The gradient and a Hessian-vector product (second-order rules).
        g = torch.autograd.grad(loss, a, create_graph=True)[0]
        return g.detach(), torch.autograd.grad((g * w).sum(), a)[0]

    for name, fn in fns.items():
        out = {}
        for when in ("inside", "after", "default"):
            a = z.clone().requires_grad_()
            with sf.options(**(OPTION_SETS[opts] if when != "default" else {})):
                loss = fn(a)
                if when != "after":
                    out[when] = derivatives(loss, a)
            if when == "after":
                out[when] = derivatives(loss, a)
        for k in range(2):
            assert torch.equal(out["after"][k], out["inside"][k]), (name, k)
        changed = [not torch.equal(out["inside"][k], out["default"][k]) for k in range(2)]
        assert any(changed), f"{name}: {opts} changes neither derivative here"


def test_options_scope_is_per_thread(master_seed):
    """A ``with`` block holds in its own thread only, ``set_options`` in every thread, and threads computing under
    different options at the same time each get the results of their own."""
    gen = cases.rng(master_seed, "torch-options-threads")
    z = torch.as_tensor(gen.normal(size=(64, 16)) * 30, dtype=torch.float64)  # Wide rows: the shift's rounding shows.
    z[:32, 0] += 60.0  # Confident rows, where the rules differ.
    v = torch.as_tensor(gen.normal(size=z.shape), dtype=torch.float64)

    def softmax_grad(a):
        a = a.clone().requires_grad_()
        return torch.autograd.grad((sf.softmax(a, 1) * v).sum(), a)[0]

    def in_thread(fn):
        out = {}
        worker = threading.Thread(target=lambda: out.setdefault("y", fn()))
        worker.start()
        worker.join()
        return out["y"]

    for opts, fn in (({"exact_shift": False}, lambda a: sf.softmax(a, 1)), ({"rule": "paper"}, softmax_grad)):
        default = fn(z)
        with sf.options(**opts):
            changed = fn(z)
            assert not torch.equal(changed, default), f"{opts} changes nothing here"
            assert torch.equal(in_thread(lambda fn=fn: fn(z)), default)
        previous = sf.set_options(**opts)
        try:
            assert torch.equal(in_thread(lambda fn=fn: fn(z)), changed)
        finally:
            sf.set_options(**previous._asdict())
        assert torch.equal(fn(z), default)

        barrier, failures = threading.Barrier(2, timeout=60), []

        def worker(scope, expected, fn=fn, barrier=barrier, failures=failures):
            try:
                with sf.options(**scope):
                    for step in range(20):
                        barrier.wait()  # Both threads inside their blocks, computing at the same time.
                        if not torch.equal(fn(z), expected):
                            failures.append((scope, step))
            except threading.BrokenBarrierError:
                failures.append((scope, "barrier"))

        threads = [threading.Thread(target=worker, args=a) for a in (({}, default), (opts, changed))]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        assert not failures, failures


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("fmt", ("float32", "float64"))
def test_paper_rule_argmax_entry_is_minus_the_others(fmt, device, master_seed):
    """The paper rule's defining structure, in every row: the argmax entry of a reverse-mode result of log_softmax and
    softmax is minus the sum of the others. Three classes make the check exact in any summation order (two others
    and a zeroed argmax round once). The accurate rule breaks it on some rows (plain and C forms), so the check can
    fail; integer-target cross-entropy pivots at the target under both rules."""
    gen = cases.rng(master_seed, "torch-paper-structure")
    n = 512
    scale = np.repeat([0.3, 3.0], n // 2)[:, None]  # Flat rows (often q_max ≤ ½) and confident ones.
    z = torch.as_tensor(gen.normal(size=(n, 3)) * scale, dtype=DTYPES[fmt], device=device)
    v = torch.as_tensor(gen.normal(size=(n, 3)), dtype=DTYPES[fmt], device=device)
    t = torch.as_tensor(gen.integers(0, 3, n), device=device)
    i_max = z.argmax(1, keepdim=True)

    def pivoted(g, at):  # Rows whose entry at ``at`` is exactly minus the sum of the others.
        return g.gather(1, at)[:, 0] == -g.scatter(1, at, 0.0).sum(1)

    def vjp(f):
        a = z.clone().requires_grad_()
        return torch.autograd.grad((f(a) * v).sum(), a)[0]

    for rule in ("paper", "accurate"):
        with sf.options(rule=rule):
            a = z.clone().requires_grad_()
            ce = torch.autograd.grad(sf.cross_entropy(a, t, reduction="sum"), a)[0]
            assert pivoted(ce, t[:, None]).all(), rule
            for name, f in (("log_softmax", lambda a: sf.log_softmax(a, 1)), ("softmax", lambda a: sf.softmax(a, 1))):
                ok = pivoted(vjp(f), i_max)
                if rule == "paper":
                    assert ok.all(), (name, int((~ok).sum()))
                else:
                    assert not ok.all(), f"{name}: the accurate rule pivots every row too, so the check cannot fail"


def _flush_band(gen, fmt: str, n: int, k: int, lo: float = -8.0, hi: float = 6.0) -> np.ndarray:
    """Rows whose non-argmax probabilities, times O(1) cotangents, straddle the bottom of the normal range (margins
    from centre + lo to centre + hi, centre = −ln(smallest normal) − ln K; argmax first), as in test_jax: where a
    flush-to-zero arithmetic would drop the products the centred-Z rule's argmax entry sums; torch keeps them as
    subnormals, at reduced precision."""
    centre = -np.log(2.0 ** oracle.FLUSH_EXPONENT[fmt]) - np.log(k)
    others = gen.normal(size=(n, k - 1))
    lse = np.log(np.exp(others).sum(axis=1))
    return np.concatenate([(np.linspace(centre + lo, centre + hi, n) + lse)[:, None], others], axis=1)


def _scaled_rows(zr: np.ndarray, fmt: str) -> np.ndarray:
    """Rows in the flush-proof rule's scaled band: largest non-argmax probability below 2^−(T+1), T = 63 (float32) or
    511; the rule scales them while their cotangent or tangent, centred at the argmax, is below 2^T in magnitude on
    the classes of nonzero probability."""
    q = oracle.softmax(zr)[0]
    q[np.arange(len(q)), np.argmax(zr, axis=1)] = 0
    return q.max(axis=1) < 2.0 ** (-(511 if fmt == "float64" else 63) - 1)


def _softmax_vjp_jvp(z: torch.Tensor, w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    a = z.clone().requires_grad_()
    vjp = torch.autograd.grad((sf.softmax(a, 1) * w).sum(), a)[0]
    return vjp, torch.func.jvp(lambda b: sf.softmax(b, 1), (z,), (w,))[1]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("fmt", ("float32", "float64"))
def test_flush_proof_rule_is_accurate_where_products_flush(fmt, device, master_seed):
    """``rule="flush_proof"``: the softmax VJP and JVP stay accurate on the rows where XLA CPU's flush-to-zero breaks
    the accurate rule, graded as in test_jax (one flushed entry times the cotangent's size, no sum term). torch does
    not flush, so the accurate rule passes here too: the check guards the scaled construction."""
    gen = cases.rng(master_seed, "torch-flush-proof", fmt)
    zr = f64(put(_flush_band(gen, fmt, 96, 2048, -7.0, -3.5), fmt, device))
    w = f64(put(gen.normal(size=zr.shape), fmt, device))
    ref = oracle.softmax_vjp(zr, w)  # Symmetric Jacobian: the JVP at w is the VJP at w.
    allowance = oracle.flushable(fmt, oracle.softmax(zr)[0]) * (1 + np.abs(w).max(axis=1, keepdims=True))
    with sf.options(rule="flush_proof"):
        outs = _softmax_vjp_jvp(put(zr, fmt, device), put(w, fmt, device))
    worst = [float(oracle.normwise_error(f64(g), ref, fmt, ftz=allowance).max()) for g in outs]
    assert max(worst) <= 8, worst


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("fmt", ("float32", "float64"))
def test_flush_proof_rule_changes_only_the_rows_it_scales(fmt, device, master_seed):
    """``rule="flush_proof"`` against ``"accurate"``: the softmax VJP and the JVP each change on some row it scales
    (torch rounds the unscaled products as subnormals there, the scaled ones as normal numbers) and on no other row:
    its factors are exactly 1 there, so the arithmetic is the accurate rule's."""
    gen = cases.rng(master_seed, "torch-flush-proof-rows", fmt)
    ordinary = gen.normal(size=(64, 256)) * 3
    ordinary[:32, 0] += 12  # Confident rows, where the rule applies, but not scaled.
    zr = f64(put(np.concatenate([_flush_band(gen, fmt, 32, 256), ordinary]), fmt, device))
    z, w = put(zr, fmt, device), put(gen.normal(size=zr.shape), fmt, device)
    scaled = _scaled_rows(zr, fmt)
    assert scaled.any() and not scaled.all()
    out = {}
    for rule in ("flush_proof", "accurate"):
        with sf.options(rule=rule):
            out[rule] = [f64(g) for g in _softmax_vjp_jvp(z, w)]
    for mode, a, b in zip(("VJP", "JVP"), out["flush_proof"], out["accurate"], strict=True):
        differ = ~((a == b) | (np.isnan(a) & np.isnan(b))).all(axis=1)
        assert (differ & scaled).any(), f"{mode}: the scaled construction changes no row it scales"
        assert not (differ & ~scaled).any(), f"{mode}: rows not scaled changed: {np.flatnonzero(differ & ~scaled)}"


def test_flush_proof_backward_uses_its_forward_call(master_seed):
    """As test_backward_uses_the_options_of_its_forward_call, for ``rule="flush_proof"`` on rows it scales: a graph
    built under it and differentiated after its scope has ended keeps it, and it changes the gradient here."""
    gen = cases.rng(master_seed, "torch-flush-proof-backward")
    z = put(_flush_band(gen, "float32", 32, 256), "float32", "cpu")
    v = put(gen.normal(size=z.shape), "float32", "cpu")
    grads = {}
    for when in ("inside", "after", "default"):
        a = z.clone().requires_grad_()
        with sf.options(**({"rule": "flush_proof"} if when != "default" else {})):
            loss = (sf.softmax(a, 1) * v).sum()
            if when != "after":
                grads[when] = torch.autograd.grad(loss, a)[0]
        if when == "after":
            grads[when] = torch.autograd.grad(loss, a)[0]
    assert torch.equal(grads["after"], grads["inside"])
    assert not torch.equal(grads["inside"], grads["default"]), "flush_proof changes nothing here"


@pytest.mark.parametrize("device", DEVICES)
def test_flush_proof_rule_does_not_overflow(device, master_seed):
    """Rows the scaled construction would overflow (4 096 classes, off-argmax probabilities just below 2^−64,
    cotangents and tangents near 1e37 of one sign: exact results ≈ 1e21, finite in float32): not scaled, so finite and
    equal to the accurate rule's, in both modes (29/10/2026; before the bound on the cotangent they were infinite)."""
    gen = cases.rng(master_seed, "torch-flush-proof-overflow")
    z64 = gen.normal(size=(6, 4096)) * 0.1
    z64[:, 0] = 45.0
    w64 = 1e37 * (1 + 0.1 * np.abs(gen.normal(size=z64.shape)))
    w64[:, 0] = -1e37
    z, w = (torch.as_tensor(a, dtype=torch.float32, device=device) for a in (z64, w64))
    assert _scaled_rows(f64(z), "float32").all(), "the rows are not in the scaled band"
    with sf.options(rule="flush_proof"):
        vjp, jvp = _softmax_vjp_jvp(z, w)
    vjp_acc, jvp_acc = _softmax_vjp_jvp(z, w)
    assert torch.isfinite(vjp).all() and torch.isfinite(jvp).all()
    assert torch.equal(vjp, vjp_acc) and torch.equal(jvp, jvp_acc)


def test_flush_proof_bound_ignores_classes_of_zero_probability():
    """The cotangent bound of the flush-proof rule leaves out classes whose probability is 0: a row of 1 024 classes at
    q ≈ 2^−100 with cotangents 2^−30, one class at q = 0 with a cotangent of 1e20. Under flush-to-zero arithmetic
    (``torch.set_flush_denormal``, CPU) its argmax entry, about −2^−120 (normal), is lost by the accurate rule, whose
    products flush, and kept by the flush-proof rule, which scales the row (it did not when the bound counted every
    class; found in review, 29/10/2026)."""
    z = np.full((1, 1026), -100 * np.log(2.0))
    z[0, 0], z[0, 1025] = 0.0, -200.0
    w = np.full_like(z, 2.0**-30)
    w[0, 0], w[0, 1025] = 0.0, 1e20
    zt, wt = (torch.as_tensor(a, dtype=torch.float32) for a in (z, w))
    # The exact argmax entry, −q_0·Σ_j q_j w_j (w_0 = 0), in float64 from the stored inputs: the oracle's working
    # precision follows the inputs' span (1e20 here), and float64 resolves this to about 1e-16 relative.
    zr, wr = f64(zt)[0], f64(wt)[0]
    qr = np.exp(zr - zr.max()) / np.exp(zr - zr.max()).sum()
    exact = -qr[0] * float(np.dot(qr, wr))
    if not torch.set_flush_denormal(True):
        pytest.skip("flush-to-zero not supported on this CPU")
    try:
        accurate = _softmax_vjp_jvp(zt, wt)[0][0, 0].item()
        with sf.options(rule="flush_proof"):
            proof = _softmax_vjp_jvp(zt, wt)[0][0, 0].item()
    finally:
        torch.set_flush_denormal(False)
    assert accurate == 0.0, "the accurate rule's products did not flush: the row does not test the bound"
    assert abs(proof - exact) <= 1e-5 * abs(exact), (proof, exact)


def test_flush_proof_scaled_branch_derivatives(monkeypatch, master_seed):
    """The scaled construction's derivatives, first and second order, reverse and forward mode (fp64 gradcheck and
    gradgradcheck), on ordinary rows, for softmax, log_softmax and the soft-target losses (whose second derivatives go
    through the softmax rule): with T lowered to 4 (scaling where the largest non-argmax probability is below
    2^−5) it runs on rows whose finite differences are meaningful; at the real T it runs only where they are not."""
    called = []

    def scale_factors_t4(q, idx, w):  # As sf._scale_factors, with T = 4.
        called.append(q.shape)
        centred = torch.where(q > 0, (w - w.gather(-1, idx)).abs(), 0.0)
        scaled = (q.scatter(-1, idx, 0.0).amax(-1, keepdim=True) < 2.0**-5) & (centred.amax(-1, keepdim=True) < 2.0**4)
        one = torch.ones_like(q[..., :1])
        return torch.where(scaled, 2.0**-4, one), torch.where(scaled, 2.0**4, one)

    monkeypatch.setattr(sf, "_scale_factors", scale_factors_t4)
    gen = cases.rng(master_seed, "torch-flush-proof-t4")
    z = gen.normal(size=(16, 8))
    z[:12, 0] += np.linspace(3.0, 9.0, 12)  # Confident rows; most of them scaled at T = 4.
    q = oracle.softmax(z)[0]
    q[:, 0] = 0
    assert 4 <= int((q.max(axis=1) < 2.0**-5).sum()) < 16
    x = torch.as_tensor(z, dtype=torch.float64).requires_grad_()
    kw = {"atol": 1e-8, "rtol": 1e-6}
    xt = (x.detach() + 0.3 * torch.as_tensor(gen.normal(size=z.shape))).requires_grad_()
    with sf.options(rule="flush_proof"):
        for f in (lambda a: sf.softmax(a, 1), lambda a: sf.log_softmax(a, 1)):
            assert torch.autograd.gradcheck(f, (x,), check_forward_ad=True, **kw)
            assert torch.autograd.gradgradcheck(f, (x,), check_fwd_over_rev=True, **kw)
        # The soft targets' second derivatives go through the softmax rule.
        for f in (
            lambda a, b: sf.soft_target_cross_entropy(a, b, reduction="none"),
            lambda a, b: sf.kl_div_with_logits(a, b, reduction="none"),
        ):
            assert torch.autograd.gradcheck(f, (x, xt), check_forward_ad=True, **kw)
            assert torch.autograd.gradgradcheck(f, (x, xt), check_fwd_over_rev=True, **kw)
    assert called, "the lowered T was never used"


@pytest.mark.parametrize("seed_offset", range(3))
def test_exact_shift_removes_the_shift_rounding(seed_offset, master_seed):
    """fp64 ``wide`` rows (|x_j − x_max| up to hundreds): exponentiating the rounded shift costs about
    ε·|x_j − x_max|/2 relative error in the softmax and the cross-entropy gradient; the exact shift removes it."""
    with sf.options(exact_shift=False):
        rounded = measure_rows("fixed", "wide", "float64", "cpu", master_seed + seed_offset)
    exact = measure_rows("fixed", "wide", "float64", "cpu", master_seed + seed_offset)
    for m in ("softmax", "ce_int_grad"):
        assert exact[m].max() <= 8, (m, exact[m].max())
        assert rounded[m].max() >= 10 * exact[m].max(), (m, rounded[m].max(), exact[m].max())


# Keys on which an option set loses to the paper's code at the suite's default seed, each diagnosed as noise; the test
# xfails on exactly these metrics there and fails on any other.
OPTION_DEVIATIONS = {
    (o, "flat", "float64"): (
        {"softmax_vjp"},
        (
            "the paper rule's Z pivot cancels on flat rows, in ours and in the paper's code alike, with heavy-tailed, "
            "seed-dependent errors: pooled over seeds 20261026, 1, 2, 3 ours is the more accurate (worst 44.2 against "
            "55.7 ε, mean 13.1 against 14.9 ε), worse on 47 % of the rows"
        ),
    )
    for o in ("paper_rule", "rounded_shift_paper_rule")
}


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("fmt", FMTS)
@pytest.mark.parametrize("regime", FIXED_REGIMES)
@pytest.mark.parametrize("opts", [o for o in OPTION_SETS if o != "default"])
def test_options_not_less_accurate_than_the_paper(opts, regime, fmt, device, master_seed):
    """The non-default option sets against the paper's code, by the default's test: the paper rule promises the
    paper's zero-sum structure without its accuracy losses; the rounded shift trades accuracy for speed."""
    with sf.options(**OPTION_SETS[opts]):
        ours = compare_rows("fixed", regime, fmt, device, master_seed)
    paper = compare_rows("paper", regime, fmt, device, master_seed)
    worse = _less_accurate(ours, paper, PAPER_METRICS, _worst_quantile(regime))
    known, why = OPTION_DEVIATIONS.get((opts, regime, fmt), (set(), ""))
    if worse and set(worse) <= known:
        pytest.xfail(f"known deviation {worse}: {why}")
    assert not worse, f"metrics where the paper's code is more accurate (ours, paper): {worse}"
