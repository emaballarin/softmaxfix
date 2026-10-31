"""Tests of softmaxfix.jax: accuracy against the mpmath oracle, against native jax.nn / optax and against the paper's
reference code; native parity across options; transforms; edge cases."""

import threading
from types import SimpleNamespace

import cases
import jax
import jax.numpy as jnp
import mpmath
import numpy as np
import optax
import oracle
import paper_jax
import pytest
from jax.test_util import check_grads
from tolerances import tol

import softmaxfix.jax as sf

jax.config.update("jax_enable_x64", True)

FMTS = ("float16", "bfloat16", "float32", "float64")
DTYPES = {"float16": jnp.float16, "bfloat16": jnp.bfloat16, "float32": jnp.float32, "float64": jnp.float64}
GAPS = cases.GAPS
EDGE_PAIRS = cases.EDGE_PAIRS
RTOL = {"float32": 2e-6, "float64": 1e-14}

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
    """Metrics native must fail in ``regime`` (every one, on every format and device: no exclusion is left in JAX)."""
    # The float16 exclusions of cliff Σg and of the teacher / offset student gradients (from float16's own flush
    # threshold N) were dropped on 29/10/2026. Graded at float32's N, native separates there, on the CPU (seeds 1, 2,
    # 20261026: Σg 2.3e3–3.0e3 ε against ≤ 0.66; student gradients 3.2e3 ε to inf against ≤ 0.46) and on the GPU, where
    # JAX computes float16 in float16 as on the CPU (calibration seeds 1–30, tests/calibration/: native's smallest
    # error over the tolerance 1.05e3 for Σg, 818 for the offset and about 9e307, a non-finite native, for the teacher
    # gradients, as on the CPU).
    return MUST_FAIL[regime]


def tol_key(device: str) -> str:
    """Tolerances are kept per device, so that a device missing at calibration time fails loudly here."""
    return f"jax-{device}"


def _devices() -> list:
    params = [pytest.param("cpu", id="cpu")]
    try:
        has_gpu = bool(jax.devices("gpu"))
    except RuntimeError:
        has_gpu = False
    if has_gpu:
        params.append(pytest.param("gpu", id="gpu", marks=pytest.mark.gpu))
    return params


DEVICES = _devices()

IMPLS = {
    "fixed": SimpleNamespace(
        log_softmax=sf.log_softmax,
        softmax=sf.softmax,
        ce_int=sf.softmax_cross_entropy_with_integer_labels,
        ce_prob=sf.softmax_cross_entropy,
        soft_ce=sf.soft_target_cross_entropy,
        kl=sf.kl_divergence_with_logits,
    ),
    "native": SimpleNamespace(
        log_softmax=jax.nn.log_softmax,
        softmax=jax.nn.softmax,
        ce_int=optax.losses.softmax_cross_entropy_with_integer_labels,
        ce_prob=optax.losses.softmax_cross_entropy,
        soft_ce=lambda zs, zt: optax.losses.softmax_cross_entropy(zs, jax.nn.softmax(zt)),
        kl=lambda zs, zt: optax.losses.kl_divergence(jax.nn.log_softmax(zs), jax.nn.softmax(zt)),
    ),
    "paper": SimpleNamespace(
        log_softmax=paper_jax.log_softmax,
        softmax=paper_jax.softmax,
        ce_int=paper_jax.cross_entropy,
        ce_prob=None,
        soft_ce=paper_jax.soft_target_cross_entropy,
        kl=paper_jax.kl_divergence,
    ),
}


def put(a, fmt: str, device: str) -> jax.Array:
    return jax.device_put(jnp.asarray(a, DTYPES[fmt]), jax.devices(device)[0])


def f64(x) -> np.ndarray:
    return np.asarray(x).astype(np.float64)


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
    z = put(z64, fmt, device)
    zr = f64(z)
    tj = jax.device_put(jnp.asarray(t), jax.devices(device)[0])
    k = zr.shape[1]
    e_t = np.eye(k)[t]
    e_max = np.eye(k)[np.argmax(zr, axis=1)]
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
    loss, vjp = jax.vjp(lambda z: f.ce_int(z, tj), z)
    g = vjp(jnp.ones_like(loss))[0]
    out["ce_int_value"] = ue(loss, oracle.ce_int(zr, t))
    out["ce_int_grad"] = ued(g, oracle.grad_ce_int(zr, t))
    out["ce_int_grad_sum"] = _grad_sum_error(f64(g), fmt, ftz + _result_allowance(fmt, oracle.grad_ce_int(zr, t)))
    if f.ce_prob is not None:
        loss, vjp = jax.vjp(lambda z: f.ce_prob(z, put(e_t, fmt, device)), z)
        out["ce_onehot_value"] = ue(loss, oracle.ce_int(zr, t))
        out["ce_onehot_grad"] = ued(vjp(jnp.ones_like(loss))[0], oracle.grad_ce_int(zr, t))
        loss, vjp = jax.vjp(lambda z: f.ce_prob(z, labels), z)
        out["ce_soft_label_value"] = ue(loss, oracle.ce_prob(zr, lr))
        out["ce_soft_label_grad"] = ned(vjp(jnp.ones_like(loss))[0], oracle.grad_ce_prob(zr, lr))
    _, vjp = jax.vjp(f.softmax, z)
    out["softmax_vjp_argmax"] = ued(vjp(put(e_max, fmt, device))[0], oracle.softmax_vjp(zr, e_max))
    out["softmax_vjp"] = ned(vjp(v)[0], oracle.softmax_vjp(zr, vr))
    _, vjp = jax.vjp(f.log_softmax, z)
    out["log_softmax_vjp"] = ned(vjp(v)[0], oracle.log_softmax_vjp(zr, vr))
    return {name: _rowmax(e) for name, e in out.items()}


def _measure_soft(f, zs64, zt64, fmt, device) -> dict[str, np.ndarray]:
    zs, zt = put(zs64, fmt, device), put(zt64, fmt, device)
    zsr, ztr = f64(zs), f64(zt)
    ftz = oracle.flushable(fmt, oracle.softmax(zsr)[0], oracle.softmax(ztr)[0])
    out = {}
    for prefix, fn, ref_value, ref_teacher in [
        ("soft", f.soft_ce, oracle.soft_ce, oracle.grad_teacher_ce),
        ("kl", f.kl, oracle.kl, oracle.grad_teacher_kl),
    ]:
        if fn is None:
            continue
        val, vjp = jax.vjp(fn, zs, zt)
        gs, gt = vjp(jnp.ones_like(val))
        value_name = "soft_ce_value" if prefix == "soft" else "kl_value"
        out[value_name] = oracle.ulp_error(f64(val), ref_value(zsr, ztr), fmt, ftz=ftz)
        for side, g, ref in (("student", gs, oracle.grad_student(zsr, ztr)), ("teacher", gt, ref_teacher(zsr, ztr))):
            fz = ftz + oracle.subnormal_mass(fmt, ref) + oracle.underflow_rounding(fmt, ref)  # As in _measure_hard.
            out[f"{prefix}_grad_{side}"] = oracle.normwise_error(f64(g), ref, fmt, ftz=fz)
    return {name: _rowmax(e) for name, e in out.items()}


def measure_rows(impl: str, regime: str, fmt: str, device: str, seed: int) -> dict[str, np.ndarray]:
    """Per-row error of every metric of ``regime`` for implementation ``impl`` (fixed, native or paper)."""
    f = IMPLS[impl]
    gen = cases.rng(seed, "jax", regime, fmt)
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
    errs = measure("native", regime, fmt, device, master_seed)
    undetected = [k for k in must_fail(regime, fmt, device) if errs[k] <= tol(tol_key(device), regime, k, fmt)]
    assert not undetected, f"native within tolerance on {undetected}: these metrics cannot detect the bug"


# Compared over the rows of a regime, not row by row: where both errors are set by the rounding of logit differences
# (conditioning, hundreds of ulps at large margins), the paper's extra roundings cancel ours on some rows and add to
# it on others, so either code wins single rows by luck (measured: ours is worse on a minority of rows in some
# regimes where its worst and mean errors are both lower). A regression is a worse worst case or a worse mean.
PAPER_RELATIVE, PAPER_SLACK = 1.25, 4.0


def _finite_mean(e: np.ndarray) -> float:
    return float(np.mean(e[np.isfinite(e)])) if np.isfinite(e).any() else np.inf


def _less_accurate(ours: dict, ref: dict, metrics, quantile: float | None = None) -> dict:
    """Metrics on which ``ours`` has a worse worst case or a worse mean than ``ref``, with both numbers. With
    ``quantile`` (the pooled regimes, ``_worst_quantile``), that quantile of the rows stands for the worst case, and a
    metric with more non-finite rows than the reference's fails outright (a quantile could hide them). The rule of
    the comparison tests."""
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
            if more_nonfinite or o_w > PAPER_RELATIVE * r_w + PAPER_SLACK or o_mean > PAPER_RELATIVE * r_mean + 1:
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


# Excluded on purpose from the comparison with native (maintainer, 30/10/2026): (regime, format, device) → metrics
# where the library ties native, heavy tails crossing, so that the per-seed rule fires by chance. Measured over base
# seeds 1001–1060, pooled as the test pools (VALIDATION.md): margin_bigk GPU log_softmax_vjp, float32
# mean 2.24 against 2.23 ε (paired difference +0.009, 95 % CI [−0.16, +0.18]), q90 5.56 / 4.97, q99 19.4 / 21.0,
# flagged on 3 of 60 seeds (and at held-out seed 105); float64 mean 2.08 against 2.14 ε (−0.063, [−0.22, +0.10]),
# q99 17.5 / 19.9, max 40.8 / 34.1, flagged on 1 of 60 seeds. On the CPU neither is flagged on any of the 60.
NATIVE_COMPARISON_EXCLUDED = {
    ("margin_bigk", "float32", "gpu"): ("log_softmax_vjp",),
    ("margin_bigk", "float64", "gpu"): ("log_softmax_vjp",),
}


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("fmt", FMTS)
@pytest.mark.parametrize("regime", FIXED_REGIMES)
def test_not_less_accurate_than_native(regime, fmt, device, master_seed):
    """In every regime, on every device, against optax / ``jax.nn`` (30/10/2026; softmaxfix.torch's suite has had it
    from the start): native is the baseline to beat, its errors no longer setting tolerances. Metrics measured to tie
    with native are excluded where listed (``NATIVE_COMPARISON_EXCLUDED``)."""
    ours = compare_rows("fixed", regime, fmt, device, master_seed)
    native = compare_rows("native", regime, fmt, device, master_seed)
    excluded = NATIVE_COMPARISON_EXCLUDED.get((regime, fmt, device), ())
    worse = _less_accurate(ours, native, [m for m in ours if m not in excluded], _worst_quantile(regime))
    assert not worse, f"metrics where native is more accurate (ours, native): {worse}"


# ------------------------------------------------------------------------------------------- native parity
def _finite_grad(fn, x, w) -> np.ndarray:
    def scalar(x):
        y = fn(x)
        return jnp.sum(jnp.where(jnp.isfinite(y), y * w, 0))

    return f64(jax.grad(scalar)(x))


def _assert_grads_match(ours: np.ndarray, ref: np.ndarray, rtol: float) -> None:
    assert np.all(np.isfinite(ours)), "fixed gradient is not finite"
    ok = np.isfinite(ref)
    np.testing.assert_allclose(ours[ok], ref[ok], rtol=rtol, atol=rtol)


@pytest.mark.parametrize("fmt", ["float32", "float64"])
@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("axis", [-1, 0, 1, (0, 2), (2, 0), None])
def test_softmax_family_parity(axis, masked, fmt, master_seed):
    gen = cases.rng(master_seed, "jax-parity-softmax", str(axis), str(masked), fmt)
    x = jnp.asarray(gen.normal(size=(3, 4, 5)), DTYPES[fmt])
    where = None
    if masked:
        mask = gen.random((3, 4, 5)) > 0.3
        mask[0] = False
        where = jnp.asarray(mask)
    w = jnp.asarray(gen.normal(size=x.shape), DTYPES[fmt])
    for ours, ref in [(sf.log_softmax, jax.nn.log_softmax), (sf.softmax, jax.nn.softmax)]:
        a, b = ours(x, axis, where), ref(x, axis=axis, where=where)
        assert a.dtype == b.dtype and a.shape == b.shape
        np.testing.assert_allclose(f64(a), f64(b), rtol=RTOL[fmt], atol=RTOL[fmt])
        _assert_grads_match(
            _finite_grad(lambda x, ours=ours: ours(x, axis, where), x, w),
            _finite_grad(lambda x, ref=ref: ref(x, axis=axis, where=where), x, w),
            RTOL[fmt] * 10,
        )


def test_integer_input_dtype_matches_jax_nn():
    x = jnp.arange(6).reshape(2, 3)
    for ours, ref in [(sf.log_softmax, jax.nn.log_softmax), (sf.softmax, jax.nn.softmax)]:
        assert ours(x).dtype == ref(x).dtype
        np.testing.assert_allclose(f64(ours(x)), f64(ref(x)), rtol=1e-6)


@pytest.mark.parametrize("mask", [None, "example", "full"])
@pytest.mark.parametrize("axis", [-1, 0, 1, (1, 2), (2, 1)])
def test_ce_integer_labels_parity(axis, mask, master_seed):
    gen = cases.rng(master_seed, "jax-parity-ce-int", str(axis), str(mask))
    shape = (3, 4, 5)
    x = jnp.asarray(gen.normal(size=shape) * 3)
    axes = (axis,) if isinstance(axis, int) else axis
    axes = tuple(a % 3 for a in axes)
    batch = tuple(s for i, s in enumerate(shape) if i not in axes)
    k = int(np.prod([shape[a] for a in axes]))
    labels = gen.integers(-k, k, size=batch)
    labels.flat[0] = k  # Out of range: NaN in both.
    labels.flat[-1] = -k - 1
    labels = jnp.asarray(labels)
    where = None
    if mask == "example":
        where = jnp.asarray(gen.random(batch) > 0.2)
    elif mask == "full":
        m = gen.random(shape) > 0.3
        m[0] = False
        where = jnp.asarray(m)

    def ours(x):
        return sf.softmax_cross_entropy_with_integer_labels(x, labels, axis=axis, where=where)

    def ref(x):
        return optax.losses.softmax_cross_entropy_with_integer_labels(x, labels, axis=axis, where=where)

    a, b = ours(x), ref(x)
    assert a.shape == b.shape and a.dtype == b.dtype
    np.testing.assert_allclose(f64(a), f64(b), rtol=1e-13, atol=1e-13)
    w = jnp.asarray(gen.normal(size=b.shape))
    _assert_grads_match(_finite_grad(ours, x, w), _finite_grad(ref, x, w), 1e-12)


@pytest.mark.parametrize(("logits_shape", "labels_shape"), [((1, 5), (8,)), ((8, 5), (1,)), ((2, 1, 5), (2, 3))])
def test_ce_integer_labels_broadcast_like_optax(logits_shape, labels_shape, master_seed):
    """optax broadcasts the labels against the logits' batch (through take_along_axis), in value and under grad: the
    same here, with the gradient summed back to the logits' shape (29/10/2026; differentiation used to raise)."""
    gen = cases.rng(master_seed, "jax-ce-label-broadcast")
    z = jnp.asarray(gen.normal(size=logits_shape))
    t = jnp.asarray(gen.integers(0, logits_shape[-1], labels_shape))
    ref = optax.losses.softmax_cross_entropy_with_integer_labels
    ours = sf.softmax_cross_entropy_with_integer_labels
    np.testing.assert_allclose(ours(z, t), ref(z, t), rtol=1e-13, atol=1e-13)
    g_ref = jax.grad(lambda a: ref(a, t).sum())(z)
    g = jax.grad(lambda a: ours(a, t).sum())(z)
    assert g.shape == g_ref.shape == logits_shape
    np.testing.assert_allclose(g, g_ref, rtol=1e-13, atol=1e-13)
    with pytest.raises(ValueError):  # Incompatible batches raise, as in optax.
        ours(jnp.zeros((3, 5)), jnp.zeros((8,), jnp.int32))


@pytest.mark.parametrize(
    ("logits_shape", "labels_shape"),
    [((5,), (8,)), ((3, 5), (2, 3)), ((1, 5), (2, 8)), ((2, 1, 5), (3,)), ((2, 3, 5), (3,))],
)
def test_ce_integer_labels_rank_mismatch_raises_like_optax(logits_shape, labels_shape):
    """Labels need one axis per batch axis of the logits (optax's take_along_axis): broadcasting does not add axes."""
    z, t = jnp.zeros(logits_shape), jnp.zeros(labels_shape, jnp.int32)
    with pytest.raises(ValueError):
        optax.losses.softmax_cross_entropy_with_integer_labels(z, t)
    with pytest.raises(ValueError):
        sf.softmax_cross_entropy_with_integer_labels(z, t)


@pytest.mark.parametrize("where_shape", [(8,), (8, 5), (1, 5)])
def test_ce_integer_labels_broadcast_with_where_like_optax(where_shape, master_seed):
    """A mask may carry the batch too (optax's logsumexp broadcasts it against the logits), in value and under grad."""
    gen = cases.rng(master_seed, "jax-ce-label-broadcast-where", str(where_shape))
    z = jnp.asarray(gen.normal(size=(1, 5)))
    t = jnp.asarray(gen.integers(0, 5, (8,)))
    w = jnp.asarray(gen.random(where_shape) > 0.3)
    ref = optax.losses.softmax_cross_entropy_with_integer_labels
    ours = sf.softmax_cross_entropy_with_integer_labels
    np.testing.assert_allclose(ours(z, t, where=w), ref(z, t, where=w), rtol=1e-13, atol=1e-13)
    g_ref = jax.grad(lambda a: ref(a, t, where=w).sum())(z)
    g = jax.grad(lambda a: ours(a, t, where=w).sum())(z)
    np.testing.assert_allclose(g, g_ref, rtol=1e-13, atol=1e-13)


def test_ce_integer_labels_where_does_not_widen_the_classes():
    """A mask broadcasts the logits' batch, never their classes: logits (4, 1) under a (4, 5) mask raise (optax
    broadcasts the one class to five, giving NaN for labels ≥ 1 and log 5 too large otherwise)."""
    with pytest.raises(ValueError):
        sf.softmax_cross_entropy_with_integer_labels(
            jnp.zeros((4, 1)), jnp.asarray([0, 3, 4, 1]), where=jnp.ones((4, 5), bool)
        )


def test_ce_integer_labels_masked_target_nonfinite_like_optax():
    """``where`` masking out the target on rows with infinite logits or a mask on −inf logits only, all with an
    infinite loss: the value is optax's, bitwise, and so is the gradient on rows without a +inf (on those, optax's
    logsumexp can give 0 at a finite entry where ours gives NaN, as before 29/10/2026)."""
    inf = jnp.inf
    rows = [  # logits, where, label (masked out)
        ([1.0, -inf, 3.0], [0, 1, 0], 0),
        ([1.0, 2.0, inf], [0, 1, 1], 0),
        ([inf, 2.0, 3.0], [0, 1, 1], 0),
        ([2.0, -inf, 1.0], [1, 0, 1], 1),
        ([-inf, 3.0, 1.0], [1, 0, 0], 1),
        ([1.0, inf, inf], [0, 1, 1], 0),
        ([1.0, -inf, -inf, 2.0], [0, 1, 1, 0], 0),
    ]
    ref = optax.losses.softmax_cross_entropy_with_integer_labels
    ours = sf.softmax_cross_entropy_with_integer_labels
    for z, w, t in rows:
        z, w, t = jnp.asarray([z]), jnp.asarray([w], bool), jnp.asarray([t])
        np.testing.assert_array_equal(f64(ours(z, t, where=w)), f64(ref(z, t, where=w)))
        if not np.any(f64(z) == np.inf):
            g = jax.grad(lambda a, t=t, w=w: ours(a, t, where=w).sum())(z)
            g_ref = jax.grad(lambda a, t=t, w=w: ref(a, t, where=w).sum())(z)
            np.testing.assert_array_equal(f64(g), f64(g_ref))


@pytest.mark.parametrize("fmt", ["float32", "float64"])
def test_ce_integer_labels_masked_target_value_offset_rows(fmt, master_seed):
    """With the target masked out, LSE_valid(x) − x_t on rows offset far from 0, the target's logit near the masked
    maximum m: (m − x_t) is computed exactly, so the error is a few ulp of the largest quantity that must be rounded,
    max(|result|, log(1 + s)). optax's form rounds LSE_valid(x) ≈ m first, off by up to ½ ulp of m (hundreds of those
    ulp here)."""
    gen = cases.rng(master_seed, "jax-ce-masked-target-offset", fmt)
    dt, n, k = DTYPES[fmt], 256, 16
    m = gen.choice([-1.0, 1.0], n) * gen.uniform(300, 3000, n)
    z = m[:, None] - np.concatenate([np.zeros((n, 1)), gen.uniform(1, 20, (n, k - 1))], axis=1)
    t = gen.integers(1, k, n)
    z[np.arange(n), t] = m + gen.uniform(-3, 3, n)
    z = np.asarray(jnp.asarray(z, dt), np.float64)  # The rounded inputs, exactly.
    valid = np.ones(z.shape, bool)
    valid[np.arange(n), t] = False
    got = f64(
        sf.softmax_cross_entropy_with_integer_labels(jnp.asarray(z, dt), jnp.asarray(t), where=jnp.asarray(valid))
    )
    ctx = mpmath.MPContext()
    ctx.dps = 50
    ref, l1p = np.empty(n), np.empty(n)
    for r, (row, w, i) in enumerate(zip(z, valid, t)):
        mx = ctx.mpf(float(row[w].max()))
        log1ps = ctx.log(ctx.fsum(ctx.exp(ctx.mpf(float(v)) - mx) for v in row[w]))
        ref[r], l1p[r] = float(mx + log1ps - ctx.mpf(float(row[i]))), float(log1ps)
    scale = np.spacing(np.maximum(np.abs(ref), l1p).astype(np.dtype(fmt))).astype(np.float64)
    err = np.abs(got - ref) / scale
    assert float(err.max()) <= 4, f"worst {err.max():.1f} ulp of max(|result|, log(1 + s))"


@pytest.mark.parametrize("fmt", ["float32", "float64"])
def test_ce_integer_labels_masked_target_gradient_on_flat_rows(fmt, master_seed):
    """With the target masked out, the gradient is optax's to a few ε on flat rows; autodiff of the value's exact-shift
    form would give the argmax entry as 1 − s/(1 + s), off by ~10³ ε at K = 1 024 (VALIDATION.md,
    "Probes cited in the code")."""
    gen = cases.rng(master_seed, "jax-ce-masked-target-flat", fmt)
    z = jnp.asarray(gen.normal(scale=0.1, size=(16, 1024)), DTYPES[fmt])
    w = jnp.ones(z.shape, bool).at[:, 0].set(False)
    t = jnp.zeros(16, jnp.int32)
    g = jax.grad(lambda a: sf.softmax_cross_entropy_with_integer_labels(a, t, where=w).sum())(z)
    g_ref = jax.grad(lambda a: optax.losses.softmax_cross_entropy_with_integer_labels(a, t, where=w).sum())(z)
    np.testing.assert_allclose(f64(g), f64(g_ref), rtol=8 * float(jnp.finfo(z.dtype).eps), atol=0)


def test_log_mass_degenerate_rows():
    """The soft-target support-mass helper on degenerate rows: −inf for an empty mask or one on −inf logits only (also
    beside a +inf outside it), NaN with a +inf inside it or no finite logit, as the difference of two logsumexps."""
    inf, nan = jnp.inf, jnp.nan
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
        got = f64(sf._log_mass(jnp.asarray([z]), jnp.asarray([m], bool)))
        np.testing.assert_array_equal(got.ravel(), [expected], err_msg=f"z={z} mask={m}")


@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("axis", [-1, 0, (1, 2), None])
def test_ce_probability_labels_parity(axis, masked, master_seed):
    gen = cases.rng(master_seed, "jax-parity-ce-prob", str(axis), str(masked))
    x = jnp.asarray(gen.normal(size=(3, 4, 5)) * 3)
    labels = jnp.asarray(gen.random((3, 4, 5)))
    where = jnp.asarray(gen.random((3, 4, 5)) > 0.3) if masked else None
    a = sf.softmax_cross_entropy(x, labels, axis=axis, where=where)
    b = optax.losses.softmax_cross_entropy(x, labels, axis=axis, where=where)
    assert a.shape == b.shape and a.dtype == b.dtype
    np.testing.assert_allclose(f64(a), f64(b), rtol=1e-13, atol=1e-13)
    w = jnp.asarray(gen.normal(size=b.shape))

    def ours(*args):
        return jnp.sum(sf.softmax_cross_entropy(*args, axis=axis, where=where) * w)

    def ref(*args):
        return jnp.sum(optax.losses.softmax_cross_entropy(*args, axis=axis, where=where) * w)

    for argnum in (0, 1):
        np.testing.assert_allclose(
            f64(jax.grad(ours, argnum)(x, labels)), f64(jax.grad(ref, argnum)(x, labels)), rtol=1e-12, atol=1e-12
        )


@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("axis", [-1, 0, (1, 2)])
def test_soft_target_losses_parity(axis, masked, master_seed):
    gen = cases.rng(master_seed, "jax-parity-soft", str(axis), str(masked))
    zs = jnp.asarray(gen.normal(size=(3, 4, 5)) * 2)
    zt = jnp.asarray(gen.normal(size=(3, 4, 5)) * 2)
    where = jnp.asarray(gen.random((3, 4, 5)) > 0.3) if masked else None

    def ref_ce(zs, zt):
        p = jax.nn.softmax(zt, axis=axis, where=where)
        return optax.losses.softmax_cross_entropy(zs, p, axis=axis, where=where)

    def ref_kl(zs, zt):
        p = jax.nn.softmax(zt, axis=axis, where=where)
        lq = jax.nn.log_softmax(zs, axis=axis, where=where)
        return optax.losses.kl_divergence(jnp.where(p > 0, lq, 0), p, axis=axis, where=where)

    for ours, ref in [(sf.soft_target_cross_entropy, ref_ce), (sf.kl_divergence_with_logits, ref_kl)]:
        a, b = ours(zs, zt, axis=axis, where=where), ref(zs, zt)
        assert a.shape == b.shape and a.dtype == b.dtype
        np.testing.assert_allclose(f64(a), f64(b), rtol=1e-12, atol=1e-13)
        w = jnp.asarray(gen.normal(size=b.shape))

        def ours_scalar(zs, zt, ours=ours, w=w):
            return jnp.sum(ours(zs, zt, axis=axis, where=where) * w)

        def ref_scalar(zs, zt, ref=ref, w=w):
            return jnp.sum(ref(zs, zt) * w)

        for argnum in (0, 1):
            ga = jax.grad(ours_scalar, argnum)(zs, zt)
            gb = jax.grad(ref_scalar, argnum)(zs, zt)
            np.testing.assert_allclose(f64(ga), f64(gb), rtol=1e-11, atol=1e-12)


def test_input_validation_matches_optax():
    x = jnp.ones((2, 3))
    labels = jnp.array([0, 1])
    for fn in (sf.softmax_cross_entropy_with_integer_labels, optax.losses.softmax_cross_entropy_with_integer_labels):
        with pytest.raises((ValueError, TypeError)):
            fn(jnp.ones((2, 3), jnp.int32), labels)
        with pytest.raises((ValueError, TypeError)):
            fn(x, jnp.array([0.0, 1.0]))
        with pytest.raises(ValueError):
            fn(x, labels, axis=None)
    with pytest.raises(ValueError):
        sf.soft_target_cross_entropy(jnp.ones((2, 3)), jnp.ones((2, 4)))


# ---------------------------------------------------------------------------------------- precision policy
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("fmt", ["float16", "bfloat16"])
def test_half_precision_is_float32_rounded_once(fmt, device, master_seed):
    """16-bit inputs are computed in float32 (as softmaxfix.torch): values are bitwise the float32 computation on the
    same inputs, rounded once to the input dtype; gradients within one ulp of it (XLA may fuse the two programs into
    different reduction orders on GPU)."""
    gen = cases.rng(master_seed, "jax-half-policy", fmt, device)
    z = gen.normal(size=(16, 7)) * 4
    z[0, 0] += 12.0  # One confident row.
    zt = z + gen.normal(size=z.shape) * 1e-2
    t = jnp.asarray(gen.integers(0, 7, 16))
    probs = gen.dirichlet(np.ones(7), 16)
    fns = [
        ("log_softmax", sf.log_softmax, (z,)),
        ("softmax", sf.softmax, (z,)),
        ("ce_int", lambda a: sf.softmax_cross_entropy_with_integer_labels(a, t), (z,)),
        ("ce_prob", sf.softmax_cross_entropy, (z, probs)),  # Labels rounded alike in both paths.
        ("soft_ce", sf.soft_target_cross_entropy, (z, zt)),
        ("kl", sf.kl_divergence_with_logits, (z, zt)),
    ]
    for name, fn, args in fns:
        half = tuple(put(a, fmt, device) for a in args)
        single = tuple(a.astype(jnp.float32) for a in half)
        (y, vjp), (y32, vjp32) = jax.vjp(fn, *half), jax.vjp(fn, *single)
        assert y.dtype == DTYPES[fmt], name
        assert np.array_equal(f64(y), f64(y32.astype(DTYPES[fmt]))), f"{name}: value is not float32 rounded once"
        w = put(gen.normal(size=y.shape), fmt, device)  # A generic cotangent (all-ones makes softmax's VJP 0).
        for g, g32 in zip(vjp(w), vjp32(w.astype(jnp.float32))):
            assert g.dtype == DTYPES[fmt], name
            err = np.abs(f64(g) - f64(g32.astype(DTYPES[fmt])))
            assert np.all(err <= oracle.ulp(f64(g32), fmt)), f"{name}: gradient off by more than one ulp"


# ------------------------------------------------------------------------------------------------ transforms
def _fns(gen) -> list:
    """(name, function, args, n): every public function; the first n args are differentiable, the rest labels."""
    z = jnp.asarray(gen.normal(size=(4, 5)) * 2)
    zt = jnp.asarray(gen.normal(size=(4, 5)) * 2)
    z = z.at[0, 0].add(25.0)  # One confident row, so both branches of the log_softmax rule are exercised.
    labels = jnp.asarray(gen.integers(0, 5, 4))
    probs = jnp.asarray(gen.dirichlet(np.ones(5), 4))
    return [
        ("log_softmax", sf.log_softmax, (z,), 1),
        ("softmax", sf.softmax, (z,), 1),
        ("ce_int", sf.softmax_cross_entropy_with_integer_labels, (z, labels), 1),
        ("ce_prob", sf.softmax_cross_entropy, (z, probs), 2),
        ("soft_ce", sf.soft_target_cross_entropy, (z, zt), 2),
        ("kl", sf.kl_divergence_with_logits, (z, zt), 2),
    ]


@pytest.mark.parametrize("order", [2, 3])
def test_check_grads(order, master_seed):
    for _name, fn, args, n in _fns(cases.rng(master_seed, "jax-check-grads")):

        def diff(*d, fn=fn, rest=args[n:]):
            return fn(*d, *rest)

        check_grads(diff, args[:n], order=order, modes=("fwd", "rev"), atol=1e-6, rtol=1e-6)


def test_jit_vmap_jacfwd_jacrev_consistency(master_seed):
    for name, fn, args, n in _fns(cases.rng(master_seed, "jax-transforms")):
        direct = f64(fn(*args))
        np.testing.assert_allclose(f64(jax.jit(fn)(*args)), direct, rtol=1e-14, err_msg=name)

        def row(*a, fn=fn):
            return fn(*(x[None] for x in a))[0]

        np.testing.assert_allclose(f64(jax.vmap(row)(*args)), direct, rtol=1e-14, err_msg=name)
        for argnum in range(n):
            jf = jax.jacfwd(fn, argnum)(*args)
            jr = jax.jacrev(fn, argnum)(*args)
            np.testing.assert_allclose(f64(jf), f64(jr), rtol=1e-12, atol=1e-14, err_msg=f"{name} arg {argnum}")


def _row_hessian(fn, zs, zt=None) -> jax.Array:
    """Per-row Hessian w.r.t. the first argument, shape (n, K·K)."""
    if zt is None:
        h = jax.vmap(jax.hessian(lambda r: fn(r[None])[0]))(zs)
    else:
        h = jax.vmap(jax.hessian(lambda r, s: fn(r[None], s[None])[0]))(zs, zt)
    return h.reshape(h.shape[0], -1)


def measure_hessian(regime: str, fmt: str, seed: int) -> dict[str, float]:
    """Normwise Hessian error (eps) against diag(q) − q qᵀ, per loss; inf if any entry is non-finite."""
    gen = cases.rng(seed, "jax-hessian", regime, fmt)
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
    # On the CPU explicitly, as its tolerance key says (JAX's default device is the GPU where there is one).
    zs, zt = (jax.device_put(jnp.asarray(a, DTYPES[fmt]), jax.devices("cpu")[0]) for a in (zs64, zt64))
    ref = oracle.hessian(f64(zs))
    labels = jnp.zeros(8, jnp.int32)
    hs = {
        "soft_ce": _row_hessian(sf.soft_target_cross_entropy, zs, zt),
        "kl": _row_hessian(sf.kl_divergence_with_logits, zs, zt),
    }
    if regime in ("random", "margin"):
        # Each row with its own label (the library broadcasts labels against one row like optax since 29/10/2026;
        # before, differentiating through that broadcast raised).
        hs["ce_int"] = _row_hessian(sf.softmax_cross_entropy_with_integer_labels, zs, labels)
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


# ------------------------------------------------------------------------------------------------ edge cases
def test_all_but_one_masked_has_zero_gradients():
    ninf = -jnp.inf
    z = jnp.array([[1.0, ninf, ninf], [ninf, 2.0, ninf]])
    t = jnp.array([0, 1])
    np.testing.assert_array_equal(f64(sf.log_softmax(z)), f64(jax.nn.log_softmax(z)))
    g = jax.grad(lambda z: jnp.sum(jnp.where(jnp.isfinite(sf.log_softmax(z)), sf.log_softmax(z), 0)))(z)
    assert np.all(f64(g) == 0)
    loss, g = jax.value_and_grad(lambda z: sf.softmax_cross_entropy_with_integer_labels(z, t).sum())(z)
    assert float(loss) == 0.0 and np.all(f64(g) == 0)
    assert np.all(f64(jax.grad(lambda z: sf.softmax(z)[:, 0].sum())(z)) == 0)


def test_single_class_and_ties():
    z1 = jnp.array([[2.5], [-1.0]])
    assert np.all(f64(sf.log_softmax(z1)) == 0)
    assert np.all(f64(jax.grad(lambda z: sf.log_softmax(z).sum())(z1)) == 0)
    zt = jnp.array([[3.0, 3.0, 0.0], [1.0, 1.0, 1.0]])
    np.testing.assert_allclose(f64(sf.log_softmax(zt)), f64(jax.nn.log_softmax(zt)), rtol=1e-15)
    g = f64(jax.grad(lambda z: sf.softmax_cross_entropy_with_integer_labels(z, jnp.array([1, 2])).sum())(zt))
    np.testing.assert_allclose(g, f64(jax.nn.softmax(zt)) - np.eye(3)[[1, 2]], atol=1e-15)


def test_nonfinite_inputs_match_native():
    ninf, inf, nan = -jnp.inf, jnp.inf, jnp.nan
    z = jnp.array([[ninf, ninf, ninf], [1.0, inf, 0.0], [inf, inf, 0.0], [1.0, nan, 0.0]])
    for ours, ref in [(sf.log_softmax, jax.nn.log_softmax), (sf.softmax, jax.nn.softmax)]:
        np.testing.assert_array_equal(f64(ours(z)), f64(ref(z)))
    for t in (jnp.array([0, 1, 0, 2]), jnp.array([1, 0, 2, 0])):
        a = f64(sf.softmax_cross_entropy_with_integer_labels(z, t))
        b = f64(optax.losses.softmax_cross_entropy_with_integer_labels(z, t))
        assert np.all(~np.isfinite(b)) and np.all(np.isnan(a)), (a, b)


@pytest.mark.parametrize("shape", [(0, 5), (3, 0), (0,)])
def test_empty_inputs(shape):
    x = jnp.zeros(shape)
    for ours, ref in [(sf.log_softmax, jax.nn.log_softmax), (sf.softmax, jax.nn.softmax)]:
        a, b = ours(x), ref(x)
        assert a.shape == b.shape and a.dtype == b.dtype
    if len(shape) == 2 and shape[0] == 0:
        assert sf.softmax_cross_entropy_with_integer_labels(x, jnp.zeros(0, jnp.int32)).shape == (0,)
        assert sf.soft_target_cross_entropy(x, x).shape == (0,)


def measure_edge(case: str, fmt: str) -> dict[str, float]:
    """Errors on one hand-made soft-target row: value (ulps), student and teacher gradients (normwise eps)."""
    zs64, zt64 = (np.array([r]) for r in EDGE_PAIRS[case])
    # On the CPU explicitly, as its tolerance key says (JAX's default device is the GPU where there is one).
    zs, zt = (jax.device_put(jnp.asarray(a, DTYPES[fmt]), jax.devices("cpu")[0]) for a in (zs64, zt64))
    zsr, ztr = f64(zs), f64(zt)
    out = {}
    for name, fn, ref_value, ref_teacher in [
        ("soft_ce", sf.soft_target_cross_entropy, oracle.soft_ce, oracle.grad_teacher_ce),
        ("kl", sf.kl_divergence_with_logits, oracle.kl, oracle.grad_teacher_kl),
    ]:
        val, vjp = jax.vjp(fn, zs, zt)
        gs, gt = vjp(jnp.ones_like(val))
        out[f"{name}_value"] = float(oracle.ulp_error(f64(val), ref_value(zsr, ztr), fmt).max())
        out[f"{name}_grad_student"] = float(oracle.normwise_error(f64(gs), oracle.grad_student(zsr, ztr), fmt).max())
        if np.all(np.isfinite(ref_teacher(zsr, ztr)[0])):
            out[f"{name}_grad_teacher"] = float(oracle.normwise_error(f64(gt), ref_teacher(zsr, ztr), fmt).max())
        out[f"{name}_hessian_finite"] = 0.0 if np.all(np.isfinite(f64(_row_hessian(fn, zs, zt)))) else np.inf
    return out


@pytest.mark.parametrize("fmt", HESSIAN_FMTS)
@pytest.mark.parametrize("case", list(EDGE_PAIRS))
def test_soft_target_edge_rows(case, fmt):
    errs = measure_edge(case, fmt)
    over = {k: v for k, v in errs.items() if not v <= tol(tol_key("cpu"), "edge", k, fmt)}
    assert not over, f"over tolerance: {over}"


def test_no_nans_under_debug_nans():
    rows = [v for k, v in EDGE_PAIRS.items() if len(v[0]) == 3 and k != "student masked"]  # Finite losses only.
    zs = jnp.array([v[0] for v in rows])
    zt = jnp.array([v[1] for v in rows])
    labels = jnp.array([0, 1])
    zm = jnp.array([[1.0, -jnp.inf, -jnp.inf], [-jnp.inf, 2.0, 0.5]])
    with jax.debug_nans(True):
        for fn in (sf.soft_target_cross_entropy, sf.kl_divergence_with_logits):

            def finite_sum(a, b, fn=fn):
                y = fn(a, b)
                return jnp.sum(jnp.where(jnp.isfinite(y), y, 0))

            def row(a, b, fn=fn):
                return fn(a[None], b[None])[0]

            jax.grad(finite_sum, (0, 1))(zs, zt)
            jax.vmap(jax.hessian(row))(zs, zt)
        jax.grad(lambda z: sf.softmax_cross_entropy_with_integer_labels(z, labels).sum())(zm)
        jax.hessian(lambda z: sf.softmax_cross_entropy_with_integer_labels(z, labels).sum())(zm)


@pytest.mark.parametrize("fmt", FMTS)
def test_kl_finite_when_a_student_probability_underflows(fmt):
    """Support is read from the logits: an underflowed q_j is still mass (the first version returned +inf here)."""
    zs, zt = (jnp.asarray([r], DTYPES[fmt]) for r in cases.UNDERFLOW_ROWS[fmt])
    val, vjp = jax.vjp(sf.kl_divergence_with_logits, zs, zt)
    gs, gt = vjp(jnp.ones_like(val))
    zsr, ztr = f64(zs), f64(zt)
    assert oracle.ulp_error(f64(val), oracle.kl(zsr, ztr), fmt).max() <= 2
    assert oracle.normwise_error(f64(gs), oracle.grad_student(zsr, ztr), fmt).max() <= 4
    assert oracle.normwise_error(f64(gt), oracle.grad_teacher_kl(zsr, ztr), fmt).max() <= 4


# ------------------------------------------------------------------------------------------------ properties
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("fmt", FMTS)
def test_exact_zero_at_equality(fmt, device, master_seed):
    z = put(cases.rng(master_seed, "jax-equality", fmt).normal(size=(32, 7)) * 3, fmt, device)
    for fn in (sf.soft_target_cross_entropy, sf.kl_divergence_with_logits):
        val, vjp = jax.vjp(fn, z, z)
        gs, _ = vjp(jnp.ones_like(val))
        assert np.all(f64(gs) == 0), "student gradient must be exactly 0 at z_S = z_T"
    val, vjp = jax.vjp(sf.kl_divergence_with_logits, z, z)
    assert np.all(f64(val) == 0) and np.all(f64(vjp(jnp.ones_like(val))[1]) == 0)


@pytest.mark.parametrize("fmt", FMTS)
def test_kl_nonnegative_and_softmax_normalised(fmt, master_seed):
    gen = cases.rng(master_seed, "jax-properties", fmt)
    zs, zt = cases.near_teacher(gen, GAPS + (1.0, 3.0))
    kl = f64(sf.kl_divergence_with_logits(jnp.asarray(zs, DTYPES[fmt]), jnp.asarray(zt, DTYPES[fmt])))
    assert np.all(kl >= 0)
    q = f64(sf.softmax(jnp.asarray(zs * 10, DTYPES[fmt])))
    assert np.max(np.abs(q.sum(axis=1) - 1)) <= q.shape[1] * oracle.eps(fmt)


# ------------------------------------------------------------------------------------------------ the paper
def test_paper_example_softmax_collapse():
    with jax.default_device(jax.devices("cpu")[0]):
        z, t = jnp.array([9.0, -9.0, -9.0], jnp.float32), jnp.array(0)
        np.testing.assert_allclose(f64(sf.log_softmax(z)), [-3.046e-8, -18.0, -18.0], rtol=1e-3)
        np.testing.assert_allclose(
            f64(jax.grad(lambda z: sf.softmax(z)[0])(z)), [3.046e-8, -1.523e-8, -1.523e-8], rtol=1e-3
        )
        loss, g = jax.value_and_grad(sf.softmax_cross_entropy_with_integer_labels)(z, t)
        np.testing.assert_allclose(float(loss), 3.046e-8, rtol=1e-3)
        np.testing.assert_allclose(f64(g), [-3.046e-8, 1.523e-8, 1.523e-8], rtol=1e-3)
        # The paper's "standard" outputs, from jax.nn and optax.
        np.testing.assert_array_equal(f64(jax.nn.log_softmax(z)), [0.0, -18.0, -18.0])
        loss, g = jax.value_and_grad(optax.losses.softmax_cross_entropy_with_integer_labels)(z, t)
        assert float(loss) == 0.0 and float(g[0]) == 0.0


def test_paper_example_softmax_cancellation():
    with jax.default_device(jax.devices("cpu")[0]):
        z_t = jnp.array([2.0, 1.0, 0.0], jnp.float32)
        z_s = z_t + jnp.array([-1e-7, 1e-7, 0.0], jnp.float32)
        g = f64(jax.grad(sf.soft_target_cross_entropy)(z_s, z_t))
        np.testing.assert_allclose(g, [-4.596e-8, 4.144e-8, 4.513e-9], rtol=1e-3)
        exact = oracle.grad_student(f64(z_s)[None], f64(z_t)[None])
        assert oracle.ulp_error(g[None], exact, "float32").max() <= 2
        std = f64(jax.grad(lambda a, b: -jnp.sum(jax.nn.softmax(b) * jax.nn.log_softmax(a)))(z_s, z_t))
        # The HTML revision prints the standard gradient to 4 digits; the PDF's 1.451859e-8 differs from JAX 0.12's
        # 1.451589e-8 by a transposition of two digits, so only the 4-digit values are pinned.
        np.testing.assert_allclose(std, [-5.960e-8, 1.452e-8, -1.123e-9], rtol=1e-3)


def test_paper_reference_code_defects_are_absent_here():
    """The paper's own code: NaN gradients with one class left unmasked, zero Hessian of the soft-target loss.
    Kept as a record of why the derivative rules differ; ours is checked on the same inputs."""
    z = jnp.array([1.0, -jnp.inf, -jnp.inf])
    assert np.all(np.isnan(f64(jax.grad(lambda z: paper_jax.log_softmax(z)[0])(z))))
    assert np.all(f64(jax.grad(lambda z: sf.log_softmax(z)[0])(z)) == 0)
    zs, zt = jnp.array([0.5, -0.2, 0.1]), jnp.array([0.3, 0.0, -0.4])
    assert np.all(f64(jax.hessian(paper_jax.soft_target_cross_entropy)(zs, zt)) == 0)
    h = f64(jax.hessian(sf.soft_target_cross_entropy)(zs, zt))
    assert oracle.normwise_error(h.reshape(1, -1), oracle.hessian(f64(zs)[None]), "float32").max() <= 8


# ------------------------------------------------------------------------------------------------ options
OPTION_SETS = {
    "default": {},
    "rounded_shift": {"exact_shift": False},
    "paper_rule": {"rule": "paper"},
    "rounded_shift_paper_rule": {"exact_shift": False, "rule": "paper"},
    # Since 29/10/2026 the default rule is "flush_proof": "default" and "rounded_shift" above are the flush-proof sets
    # (their duplicates "flush_proof" and "rounded_shift_flush_proof" were removed), the accurate rule has its own.
    "accurate": {"rule": "accurate"},
    "rounded_shift_accurate": {"exact_shift": False, "rule": "accurate"},
}


def test_options_defaults_validation_and_scoping():
    """Defaults; invalid values are refused and change nothing; set_options returns the previous process-wide
    options; the context manager nests and restores the previous options on exit, exceptions included."""
    default = sf.Options(exact_shift=True, rule="flush_proof")
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


@pytest.mark.parametrize("opts", list(OPTION_SETS))
def test_option_derivatives(opts, master_seed):
    """check_grads of every public function, second order, forward and reverse mode, under every option set."""
    with sf.options(**OPTION_SETS[opts]):
        for _name, fn, args, n in _fns(cases.rng(master_seed, "jax-options-check-grads")):

            def diff(*d, fn=fn, rest=args[n:]):
                return fn(*d, *rest)

            check_grads(diff, args[:n], order=2, modes=("fwd", "rev"), atol=1e-6, rtol=1e-6)


def _in_thread(fn):
    out = {}
    worker = threading.Thread(target=lambda: out.setdefault("y", fn()))
    worker.start()
    worker.join()
    return out["y"]


def test_options_key_the_jit_caches(master_seed):
    """Options are read when a function is traced and are part of JAX's cache keys, so a jitted function (the
    caller's own included) is traced again under new options and never runs with stale ones. A ``with`` block holds
    in its own thread only; ``set_options`` holds in every thread."""
    gen = cases.rng(master_seed, "jax-options-jit")
    z = jnp.asarray(gen.normal(size=(64, 16)) * 30)  # Wide rows: the shift's rounding shows in the softmax value.
    z = z.at[:32, 0].add(60.0)  # Confident rows, where the rules differ.
    v = jnp.asarray(gen.normal(size=z.shape))

    def softmax_grad(a):
        return jax.grad(lambda b: jnp.sum(sf.softmax(b) * v))(a)

    for opts, fn in (({"exact_shift": False}, sf.softmax), ({"rule": "paper"}, softmax_grad)):
        user = jax.jit(fn)
        default = f64(user(z))
        assert np.array_equal(f64(fn(z)), default)
        with sf.options(**opts):
            changed = f64(fn(z))
            assert not np.array_equal(changed, default), f"{opts} changes nothing here"
            assert np.array_equal(f64(user(z)), changed)
            assert np.array_equal(f64(jax.jit(fn)(z)), changed)
            assert np.array_equal(f64(_in_thread(lambda user=user: user(z))), default)
        assert np.array_equal(f64(user(z)), default)
        previous = sf.set_options(**opts)
        try:
            assert np.array_equal(f64(user(z)), changed)
            assert np.array_equal(f64(_in_thread(lambda user=user: user(z))), changed)
        finally:
            sf.set_options(**previous._asdict())
        assert np.array_equal(f64(user(z)), default)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("fmt", ("float32", "float64"))
def test_paper_rule_argmax_entry_is_minus_the_others(fmt, device, master_seed):
    """The paper rule's defining structure, in every row: the argmax entry of a reverse-mode result of log_softmax and
    softmax is minus the sum of the others. Three classes make the check exact in any summation order (two others
    and a zeroed argmax round once). The accurate rule breaks it on some rows (plain and C forms), so the check can
    fail; integer-target cross-entropy pivots at the target under both rules."""
    gen = cases.rng(master_seed, "jax-paper-structure")
    n = 512
    scale = np.repeat([0.3, 3.0], n // 2)[:, None]  # Flat rows (often q_max ≤ ½) and confident ones.
    z = put(gen.normal(size=(n, 3)) * scale, fmt, device)
    v = put(gen.normal(size=(n, 3)), fmt, device)
    t = jax.device_put(jnp.asarray(gen.integers(0, 3, n)), jax.devices(device)[0])
    i_max = jnp.argmax(z, axis=-1)

    def pivoted(g, at):  # Rows whose entry at ``at`` is exactly minus the sum of the others.
        hot = jnp.arange(3) == at[:, None]
        return f64(jnp.sum(jnp.where(hot, g, 0), axis=-1)) == -f64(jnp.sum(jnp.where(hot, 0, g), axis=-1))

    for rule in ("paper", "accurate"):
        with sf.options(rule=rule):
            ce = jax.grad(lambda a: jnp.sum(sf.softmax_cross_entropy_with_integer_labels(a, t)))(z)
            assert pivoted(ce, t).all(), rule
            for name, f in (("log_softmax", sf.log_softmax), ("softmax", sf.softmax)):
                ok = pivoted(jax.vjp(f, z)[1](v)[0], i_max)
                if rule == "paper":
                    assert ok.all(), (name, int((~ok).sum()))
                else:
                    assert not ok.all(), f"{name}: the accurate rule pivots every row too, so the check cannot fail"


def _flush_band(gen, fmt: str, n: int, k: int, lo: float = -8.0, hi: float = 6.0) -> np.ndarray:
    """Rows whose non-argmax probabilities, times O(1) cotangents, straddle the bottom of the normal range of the
    arithmetic (margins from centre + lo to centre + hi, centre = −ln(smallest normal) − ln K; argmax first): where XLA
    CPU flushes the products that the centred-Z rule's argmax entry sums (the comparison harness's regime ``flush_bigk``
    spans −8 to +6; its losses above the tight allowance sit between about −6 and −4)."""
    centre = -np.log(2.0 ** oracle.FLUSH_EXPONENT[fmt]) - np.log(k)
    others = gen.normal(size=(n, k - 1))
    lse = np.log(np.exp(others).sum(axis=1))
    return np.concatenate([(np.linspace(centre + lo, centre + hi, n) + lse)[:, None], others], axis=1)


def _scaled_rows(zr: np.ndarray, fmt: str) -> np.ndarray:
    """Rows the flush-proof rule scales: largest non-argmax probability below 2^−(T+1), T = 63 (float32) or 511."""
    q = oracle.softmax(zr)[0]
    q[np.arange(len(q)), np.argmax(zr, axis=1)] = 0
    return q.max(axis=1) < 2.0 ** (-(511 if fmt == "float64" else 63) - 1)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("fmt", ("float32", "float64"))
def test_flush_proof_rule_is_accurate_where_products_flush(fmt, device, master_seed):
    """``rule="flush_proof"``: the softmax VJP and JVP stay accurate on rows whose products fall below the normal range,
    graded with the allowance of one flushed entry times the cotangent's size and no sum term (the comparison harness's
    tight metric). On XLA CPU the accurate rule's argmax entry, which sums the flushed products, fails it, so the check
    can fail there."""
    gen = cases.rng(master_seed, "jax-flush-proof", fmt)
    zr = f64(put(_flush_band(gen, fmt, 96, 2048, -7.0, -3.5), fmt, device))
    w = f64(put(gen.normal(size=zr.shape), fmt, device))
    z, wd = put(zr, fmt, device), put(w, fmt, device)
    ref = oracle.softmax_vjp(zr, w)  # Symmetric Jacobian: the JVP at w is the VJP at w.
    allowance = oracle.flushable(fmt, oracle.softmax(zr)[0]) * (1 + np.abs(w).max(axis=1, keepdims=True))
    worst = {}
    for rule in ("flush_proof", "accurate"):
        with sf.options(rule=rule):
            outs = (jax.vjp(sf.softmax, z)[1](wd)[0], jax.jvp(sf.softmax, (z,), (wd,))[1])
        worst[rule] = [float(oracle.normwise_error(f64(g), ref, fmt, ftz=allowance).max()) for g in outs]
    assert max(worst["flush_proof"]) <= 8, worst
    if device == "cpu":
        assert max(worst["accurate"]) > 8, f"the accurate rule passes too, so the check cannot fail: {worst}"


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("fmt", ("float32", "float64"))
def test_flush_proof_rule_changes_only_the_rows_it_scales(fmt, device, master_seed):
    """``rule="flush_proof"`` against ``"accurate"``, jointly jitted derivatives: the softmax VJP and the JVP each
    change on some row it scales (the scaled construction is engaged in both modes) and, on CPU, on no other row (they
    take the accurate rule's branch; in the comparison harness over 20 seeds XLA CPU rounded one float16 row otherwise,
    ε-level). Not asserted: on GPU XLA's graph context may reorder a reduction (ε-level, either way)."""
    gen = cases.rng(master_seed, "jax-flush-proof-rows", fmt)
    ordinary = gen.normal(size=(64, 256)) * 3
    ordinary[:32, 0] += 12  # Confident rows, where the rule applies, but not scaled.
    zr = f64(put(np.concatenate([_flush_band(gen, fmt, 32, 256), ordinary]), fmt, device))
    w = put(gen.normal(size=zr.shape), fmt, device)
    z = put(zr, fmt, device)
    scaled = _scaled_rows(zr, fmt)
    assert scaled.any() and not scaled.all()
    vjp = jax.jit(lambda a, b: jax.vjp(sf.softmax, a)[1](b)[0])
    jvp = jax.jit(lambda a, b: jax.jvp(sf.softmax, (a,), (b,))[1])
    out = {}
    for rule in ("flush_proof", "accurate"):
        with sf.options(rule=rule):
            out[rule] = (f64(vjp(z, w)), f64(jvp(z, w)))
    for mode, a, b in zip(("VJP", "JVP"), out["flush_proof"], out["accurate"], strict=True):
        differ = ~((a == b) | (np.isnan(a) & np.isnan(b))).all(axis=1)
        assert (differ & scaled).any(), f"{mode}: the scaled construction changes no row it scales"
        if device == "cpu":
            assert not (differ & ~scaled).any(), f"{mode}: rows not scaled changed: {np.flatnonzero(differ & ~scaled)}"


def test_flush_proof_scaled_branch_derivatives(monkeypatch, master_seed):
    """The scaled construction's derivatives, second order, both modes (fp64 check_grads), on ordinary rows, for softmax,
    log_softmax and the soft-target losses (whose second derivatives go through the softmax rule): with T
    lowered to 4 (scaling where the largest non-argmax probability is below 2^−5) it runs on rows whose
    finite differences are meaningful; at the real T it runs only where they are not. The kernels are called
    unjitted, so the lowered T is traced in."""

    traced = []

    def scale_factors_t4(q, is_max):  # As sf._scale_factors (the row flag and both factors), with T = 4.
        traced.append(q.shape)
        scaled = jnp.max(jnp.where(is_max, 0, q), axis=-1, keepdims=True) < 2.0**-5
        dn, up = jnp.where(scaled, 2.0**-4, 1.0).astype(q.dtype), jnp.where(scaled, 2.0**4, 1.0).astype(q.dtype)
        return scaled, dn, up

    monkeypatch.setattr(sf, "_scale_factors", scale_factors_t4)
    gen = cases.rng(master_seed, "jax-flush-proof-t4")
    z = gen.normal(size=(16, 8))
    z[:12, 0] += np.linspace(3.0, 9.0, 12)  # Confident rows; most of them scaled at T = 4.
    q = oracle.softmax(z)[0]
    q[:, 0] = 0
    assert 4 <= int((q.max(axis=1) < 2.0**-5).sum()) < 16
    opts = sf.Options(exact_shift=True, rule="flush_proof")
    for f in (lambda a: sf._softmax_last(a, opts), lambda a: sf._log_softmax_last(a, opts)):
        check_grads(f, (jnp.asarray(z),), order=2, modes=("fwd", "rev"), atol=1e-6, rtol=1e-6)
    zt = z + 0.3 * gen.normal(size=z.shape)  # The soft targets' second derivatives go through the softmax rule.
    for f in (lambda a, b: sf._soft_ce_last(a, b, opts), lambda a, b: sf._kl_last(a, b, opts)):
        check_grads(f, (jnp.asarray(z), jnp.asarray(zt)), order=2, modes=("fwd", "rev"), atol=1e-6, rtol=1e-6)
    assert traced, "the lowered T was never traced in (a cached trace ran instead)"


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


# Keys on which an option set loses to the paper's code at the suite's default seed, each diagnosed; the test xfails on
# exactly these metrics there and fails on any other.
OPTION_DEVIATIONS = {
    (o, "wide", "float64"): (
        {"ce_int_value"},
        (
            "the rounded shift's own error (the previous library's rounding, identical numbers): one row of 192 above the "
            "paper's worst (15.2 against 8.2 ulp) and a better mean (0.37 against 0.48); across seeds 1-3 the worst case "
            "goes either way (57.5 / 34.5, 22.9 / 68.1, 1.46 / 6.84) and our mean is better on each; the exact shift, the "
            "default, removes it"
        ),
    )
    for o in ("rounded_shift", "rounded_shift_paper_rule", "rounded_shift_accurate")
}


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("fmt", FMTS)
@pytest.mark.parametrize("regime", FIXED_REGIMES)
@pytest.mark.parametrize("opts", [o for o in OPTION_SETS if o != "default"])
def test_options_not_less_accurate_than_the_paper(opts, regime, fmt, device, master_seed):
    """The option sets other than ``default`` against the paper's code, by the default's test: the paper rule
    promises the paper's zero-sum structure without its accuracy losses; the rounded shift trades accuracy for speed."""
    with sf.options(**OPTION_SETS[opts]):
        ours = compare_rows("fixed", regime, fmt, device, master_seed)
    paper = compare_rows("paper", regime, fmt, device, master_seed)
    worse = _less_accurate(ours, paper, PAPER_METRICS, _worst_quantile(regime))
    known, why = OPTION_DEVIATIONS.get((opts, regime, fmt), (set(), ""))
    if worse and set(worse) <= known:
        pytest.xfail(f"known deviation {worse}: {why}")
    assert not worse, f"metrics where the paper's code is more accurate (ours, paper): {worse}"
