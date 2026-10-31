"""Arbitrary-precision reference for softmax-family values and derivatives, plus error metrics.

Every quantity is evaluated with mpmath from the exact binary value of each input, using closed forms only
(q = softmax(z), dL/dz = (Σp)·q − p, ...), never the formulas under test. Those closed forms cancel exactly where
the code under test must not (1 − q_t at margin m is e^{−m} next to O(1)), so each row is evaluated with enough
digits to resolve e^{−span} beside the logits' magnitude, plus 60 guard digits (at least 80). Results are
returned as double-double pairs ``(hi, lo)`` of float64 arrays (``hi + lo`` carries ~106 bits), which keeps the
error metrics vectorised and exact enough to grade float64 results in ulps.
"""

import functools
from collections.abc import Callable
from collections.abc import Sequence

import mpmath
import numpy as np

_CTX = mpmath.MPContext()
_CTX.dps = 80

# Precision p (significand bits, implicit bit included) and minimum normal exponent of each format.
FORMATS = {
    "float16": (11, -14),
    "bfloat16": (8, -126),
    "float32": (24, -126),
    "float64": (53, -1022),
}

type DD = tuple[np.ndarray, np.ndarray]


def eps(fmt: str) -> float:
    """Machine epsilon (spacing of the format just above 1.0)."""
    return 2.0 ** (1 - FORMATS[fmt][0])


def _mpf(x: float):
    """Exact mpmath value of a float64 (inf and nan included)."""
    return _CTX.mpf(float(x))


def _to_dd(values: Sequence) -> DD:
    """Split mpmath values into double-double (hi, lo) float64 arrays."""
    hi = np.empty(len(values))
    lo = np.zeros(len(values))
    for i, v in enumerate(values):
        h = float(v)
        hi[i] = h
        if np.isfinite(h):
            lo[i] = float(v - _CTX.mpf(h))
    return hi, lo


def _memo(fn: Callable) -> Callable:
    """Cache results by the exact bytes of the input arrays (returned arrays are shared: do not mutate)."""
    cache: dict = {}

    @functools.wraps(fn)
    def wrapper(*arrays: np.ndarray) -> DD:
        arrays = tuple(np.ascontiguousarray(a) for a in arrays)
        key = tuple((a.shape, a.dtype.str, a.tobytes()) for a in arrays)
        if key not in cache:
            cache[key] = fn(*arrays)
        return cache[key]

    return wrapper


def _row_dps(rows: list[np.ndarray]) -> int:
    """Digits resolving e^{−span} next to the largest |logit| of a row, plus 60 guard digits (at least 80)."""
    vals = np.concatenate([r[np.isfinite(r)] for r in rows])
    if vals.size == 0:
        return 80
    span = float(vals.max() - vals.min())
    return max(80, int(60 + np.log10(1.0 + np.abs(vals).max()) + span / np.log(10.0)))


def _rowwise(fn: Callable, *arrays: np.ndarray) -> DD:
    """Apply ``fn`` (lists of mpf per row → list of mpf or a single mpf) to each row of 2-D inputs."""
    n = arrays[0].shape[0]
    out = []
    for r in range(n):
        with _CTX.workdps(_row_dps([a[r] for a in arrays if a.ndim == 2])):
            args = [[_mpf(v) for v in a[r]] if a.ndim == 2 else a[r] for a in arrays]
            out.append(fn(*args))
    if isinstance(out[0], list):
        flat = [v for row in out for v in row]
        hi, lo = _to_dd(flat)
        return hi.reshape(n, -1), lo.reshape(n, -1)
    return _to_dd(out)


# ----------------------------------------------------------------------------------------------- row kernels
def _lse(z: list) -> object:
    """log Σ e^{z_k}, shifted by the row maximum: exact (log 1 = 0) when a single logit is finite, where the unshifted
    form leaves a working-precision residue (q = 1 − 1e-81, and a Hessian of 1e-81 where the exact one is 0)."""
    m = max(z)
    if _CTX.isinf(m):  # Every logit −inf (−inf), or a +inf one (inf): as the unshifted form.
        return _CTX.log(_CTX.fsum(_CTX.exp(v) for v in z))
    return m + _CTX.log(_CTX.fsum(_CTX.exp(v - m) for v in z))


def _log_softmax(z: list) -> list:
    lse = _lse(z)
    return [v - lse for v in z]


def _softmax(z: list) -> list:
    return [_CTX.exp(v) for v in _log_softmax(z)]


def _xlogy(x, y):
    """x·log(y) with 0·log(anything) := 0."""
    return _CTX.mpf(0) if x == 0 else x * _CTX.log(y)


# ------------------------------------------------------------------------------------------------ public API
@_memo
def log_softmax(z: np.ndarray) -> DD:
    """log softmax(z) along the last axis of a 2-D array."""
    return _rowwise(_log_softmax, z)


@_memo
def softmax(z: np.ndarray) -> DD:
    """softmax(z) along the last axis of a 2-D array."""
    return _rowwise(_softmax, z)


@_memo
def ce_int(z: np.ndarray, t: np.ndarray) -> DD:
    """Hard-target cross-entropy −log softmax(z)_t per row."""
    return _rowwise(lambda zr, tr: -_log_softmax(zr)[int(tr)], z, t)


@_memo
def grad_ce_int(z: np.ndarray, t: np.ndarray) -> DD:
    """d CE / dz = q − e_t per row."""
    return _rowwise(lambda zr, tr: [qk - (1 if k == int(tr) else 0) for k, qk in enumerate(_softmax(zr))], z, t)


@_memo
def ce_prob(z: np.ndarray, p: np.ndarray) -> DD:
    """Soft-label cross-entropy −Σ p_k log softmax(z)_k per row (labels given as probabilities)."""
    return _rowwise(lambda zr, pr: -_CTX.fsum(pk * lk for pk, lk in zip(pr, _log_softmax(zr)) if pk != 0), z, p)


@_memo
def grad_ce_prob(z: np.ndarray, p: np.ndarray) -> DD:
    """d CE / dz = (Σp)·q − p per row."""

    def row(zr, pr):
        s = _CTX.fsum(pr)
        return [s * qk - pk for qk, pk in zip(_softmax(zr), pr)]

    return _rowwise(row, z, p)


@_memo
def log_softmax_vjp(z: np.ndarray, v: np.ndarray) -> DD:
    """vᵀ J_logsoftmax(z) = v − q·Σv per row."""

    def row(zr, vr):
        s = _CTX.fsum(vr)
        return [vk - qk * s for vk, qk in zip(vr, _softmax(zr))]

    return _rowwise(row, z, v)


@_memo
def log_softmax_jvp(z: np.ndarray, dx: np.ndarray) -> DD:
    """J_logsoftmax(z) ẋ = ẋ − ⟨q, ẋ⟩ per row. (The softmax JVP is :func:`softmax_vjp`: its Jacobian is symmetric.)"""

    def row(zr, dr):
        s = _CTX.fsum(qk * dk for qk, dk in zip(_softmax(zr), dr))
        return [dk - s for dk in dr]

    return _rowwise(row, z, dx)


@_memo
def softmax_vjp(z: np.ndarray, v: np.ndarray) -> DD:
    """vᵀ J_softmax(z) = q ⊙ (v − ⟨q, v⟩) per row."""

    def row(zr, vr):
        q = _softmax(zr)
        s = _CTX.fsum(qk * vk for qk, vk in zip(q, vr))
        return [qk * (vk - s) for qk, vk in zip(q, vr)]

    return _rowwise(row, z, v)


@_memo
def soft_ce(zs: np.ndarray, zt: np.ndarray) -> DD:
    """Soft-target cross-entropy −Σ p log q with p = softmax(z_T), q = softmax(z_S)."""
    return _rowwise(lambda a, b: -_CTX.fsum(_xlogy(pk, qk) for pk, qk in zip(_softmax(b), _softmax(a))), zs, zt)


@_memo
def kl(zs: np.ndarray, zt: np.ndarray) -> DD:
    """KL(p ‖ q) = Σ p (log p − log q) with p = softmax(z_T), q = softmax(z_S)."""

    def row(a, b):
        p, lp, lq = _softmax(b), _log_softmax(b), _log_softmax(a)
        return _CTX.fsum(pk * (lpk - lqk) for pk, lpk, lqk in zip(p, lp, lq) if pk != 0)

    return _rowwise(row, zs, zt)


@_memo
def grad_student(zs: np.ndarray, zt: np.ndarray) -> DD:
    """d/dz_S of soft-target CE (and of KL): q − p."""
    return _rowwise(lambda a, b: [qk - pk for qk, pk in zip(_softmax(a), _softmax(b))], zs, zt)


@_memo
def grad_teacher_ce(zs: np.ndarray, zt: np.ndarray) -> DD:
    """d/dz_T of soft-target CE: −p ⊙ (log q − ⟨p, log q⟩) = −p ⊙ (z_S − ⟨p, z_S⟩).

    Evaluated through the logit form (log q and z_S differ by a row constant), so an exactly zero gradient comes
    out as exactly 0 rather than as 10^−dps of cancellation noise; test_oracle checks it by differentiation.
    """

    def row(a, b):
        p = _softmax(b)
        m = _CTX.fsum(pk * ak for pk, ak in zip(p, a) if pk != 0)
        return [_CTX.mpf(0) if pk == 0 else -pk * (ak - m) for pk, ak in zip(p, a)]

    return _rowwise(row, zs, zt)


@_memo
def grad_teacher_kl(zs: np.ndarray, zt: np.ndarray) -> DD:
    """d/dz_T of KL(p ‖ q): p ⊙ (r − KL) with r = log p − log q, i.e. −p ⊙ (δ − ⟨p, δ⟩) with δ = z_S − z_T.

    Logit form for the same reason as :func:`grad_teacher_ce`.
    """

    def row(a, b):
        p = _softmax(b)
        delta = [ak - bk for ak, bk in zip(a, b)]
        m = _CTX.fsum(pk * dk for pk, dk in zip(p, delta) if pk != 0)
        return [_CTX.mpf(0) if pk == 0 else -pk * (dk - m) for pk, dk in zip(p, delta)]

    return _rowwise(row, zs, zt)


@_memo
def hessian(z: np.ndarray) -> DD:
    """diag(q) − q qᵀ per row, flattened to (n, K·K): the logit Hessian of any CE with Σp = 1."""

    def row(zr):
        q = _softmax(zr)
        return [(qi if i == j else 0) - qi * qj for i, qi in enumerate(q) for j, qj in enumerate(q)]

    return _rowwise(row, z)


# ---------------------------------------------------------------------------------------------- error metrics
def ulp(x: np.ndarray, fmt: str) -> np.ndarray:
    """Spacing of format ``fmt`` at |x| (subnormal spacing below the normal range)."""
    p, emin = FORMATS[fmt]
    ax = np.abs(np.asarray(x, np.float64))
    _, e = np.frexp(np.where(np.isfinite(ax), ax, 1.0))
    e = np.where(ax == 0, emin, np.maximum(e - 1, emin))
    return np.ldexp(1.0, e - (p - 1))


def _diff(computed: np.ndarray, ref: DD) -> tuple[np.ndarray, np.ndarray]:
    """|computed − ref| in float64 and a mask of entries where non-finite values agree exactly."""
    hi, lo = ref
    c = np.asarray(computed, np.float64)
    with np.errstate(invalid="ignore", over="ignore"):
        d = np.abs((c - hi) - lo)
    same_nonfinite = (np.isnan(c) & np.isnan(hi)) | (np.isinf(hi) & (c == hi))
    bad_nonfinite = ~same_nonfinite & (~np.isfinite(c) | ~np.isfinite(hi))
    d = np.where(same_nonfinite, 0.0, np.where(bad_nonfinite, np.inf, d))
    return d, same_nonfinite


# Normal threshold of the arithmetic that can flush a result of each format. float16 results come from float32
# arithmetic (or float16 arithmetic, whose subnormals neither XLA CPU nor torch flushes), so only float32's applies.
FLUSH_EXPONENT = {
    "float16": FORMATS["float32"][1],
    "bfloat16": FORMATS["bfloat16"][1],
    "float32": FORMATS["float32"][1],
    "float64": FORMATS["float64"][1],
}


def flushable(fmt: str, *probs: np.ndarray) -> np.ndarray:
    """Per-row flush-to-zero allowance for operands, shape (n, 1): the normal threshold N of the arithmetic producing
    ``fmt`` (``FLUSH_EXPONENT``; float16 subnormals are kept, so float32's) plus the total of the given
    probability-sized quantities (and of each relative to its row maximum) that lie below N. That is what flushing
    those operands to zero (the XLA / CUDA convention) can remove from any sum over the row; rows without subnormal
    quantities get N alone. Derivative outputs add their own flushed results (:func:`subnormal_mass`)."""
    tiny = 2.0 ** FLUSH_EXPONENT[fmt]
    total = np.full((probs[0].shape[0], 1), tiny)
    for x in probs:
        x = np.abs(np.where(np.isfinite(x), x, 0.0))
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.nan_to_num(x / x.max(axis=1, keepdims=True))
        for v in (x, rel):
            total += np.where(v < tiny, v, 0.0).sum(axis=1, keepdims=True)
    return total


def subnormal_mass(fmt: str, ref: DD) -> np.ndarray:
    """Per-row total of a reference output's entries that flushing can remove, shape (n, 1): those below the normal
    threshold of the arithmetic producing results of ``fmt`` (``FLUSH_EXPONENT``; float16 subnormals are kept).

    Flushing those results to zero (the XLA CPU convention) can remove that much from each entry and from any sum over
    the row, such as the zero-sum pivot of a derivative; :func:`flushable` covers the operands, this the results.
    """
    tiny = 2.0 ** FLUSH_EXPONENT[fmt]
    x = np.abs(np.where(np.isfinite(ref[0]), ref[0], 0.0)).reshape(ref[0].shape[0], -1)
    return np.where(x < tiny, x, 0.0).sum(axis=1, keepdims=True)


def underflow_rounding(fmt: str, ref: DD, summed: bool = False) -> np.ndarray:
    """What rounding to ``fmt`` adds below its normal range, per row, shape (n, 1): half the subnormal spacing for one
    entry, or that times the number of nonzero reference entries there when the entries are summed (``summed``; an
    exact zero, or a non-finite entry, rounds to itself).

    Gradual underflow keeps a rounding's absolute error, not its relative one, so a row of subnormal size has fewer
    significant bits than the format. It matters for float16, whose subnormals are kept (``FLUSH_EXPONENT``); for the
    other formats that range is the flush range, already covered by :func:`flushable` and :func:`subnormal_mass`.
    """
    p, emin = FORMATS[fmt]
    x = np.abs(np.where(np.isfinite(ref[0]), ref[0], 0.0)).reshape(ref[0].shape[0], -1)
    n = ((x > 0) & (x < 2.0**emin)).sum(axis=1, keepdims=True) if summed else np.ones((x.shape[0], 1))
    return n * 2.0 ** (emin - p)


def ulp_error(
    computed: np.ndarray, ref: DD, fmt: str, scale: np.ndarray | None = None, ftz: np.ndarray | None = None
) -> np.ndarray:
    """Componentwise error in ulps of ``fmt`` at the reference value (or at ``scale`` when given).

    ``ftz`` (per row, broadcastable; default the flush threshold N of ``FLUSH_EXPONENT``) is absolute error attributed to
    flush-to-zero (see :func:`flushable`): it is subtracted first, and the remainder is graded in ulps.
    """
    d, _ = _diff(computed, ref)
    s = ref[0] if scale is None else scale
    allowance = 2.0 ** FLUSH_EXPONENT[fmt] if ftz is None else np.reshape(ftz, (-1,) + (1,) * (d.ndim - 1))
    return np.maximum(d - allowance, 0.0) / ulp(s, fmt)


def normwise_error(computed: np.ndarray, ref: DD, fmt: str, ftz: np.ndarray | None = None) -> np.ndarray:
    """Per-row max |error| / (eps · max |reference|): error relative to the row's largest component.

    The flush-to-zero allowance (as in :func:`ulp_error`) is subtracted from the absolute error first; a row whose
    reference is all zero then scores 0 if within it, and inf otherwise.
    """
    d, _ = _diff(computed, ref)
    allowance = 2.0 ** FLUSH_EXPONENT[fmt] if ftz is None else np.reshape(ftz, (-1,))
    d = np.maximum(d.reshape(d.shape[0], -1).max(axis=1) - allowance, 0.0)
    scale = np.max(np.abs(np.where(np.isfinite(ref[0]), ref[0], 0.0)).reshape(d.shape[0], -1), axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = d / (scale * eps(fmt))  # scale·eps can underflow to 0 for a subnormal row: then any d > 0 is inf.
    return np.where(d == 0, 0.0, out)
