"""JAX back-end: drop-in replacements for the softmax family of ``jax.nn`` and ``optax.losses``.

Replaces, with identical signatures and semantics (``axis`` as int / tuple / None, ``where`` masking, optax's
label wrapping and empty-row conventions):

- ``jax.nn.log_softmax``, ``jax.nn.softmax``
- ``optax.losses.softmax_cross_entropy``, ``optax.losses.softmax_cross_entropy_with_integer_labels``

and adds two losses that take *teacher logits* instead of teacher probabilities, which is what removing Softmax
Cancellation requires:

- ``soft_target_cross_entropy(logits, teacher_logits)``: −Σ p log q, with p = softmax(teacher), q = softmax(student)
- ``kl_divergence_with_logits(logits, teacher_logits)``: KL(p ‖ q)

Softmax Collapse is removed by never adding 1.0 to the exponentials that are small next to it (paper Eq. 5); the
soft-target gradient q − p is evaluated in log space from z_S − z_T (Eq. 9, HTML revision). README.md ("Relation to
the paper") maps the paper's listings to these functions. Differences from the paper's reference listings, each for a
case they do not target, with its measured effect:

- The shift x − x_max is kept exactly, as hi + lo (TwoSum), and exponentiated as e^{hi} + e^{hi}·lo (option
  ``exact_shift``). Exponentiating the rounded shift turns its rounding into a relative error of about ε·|x_j −
  x_max|/2: the whole componentwise error of the values and of the cross-entropy gradient that native, the paper's code
  and this library with the option off share (fp64 ``wide``, CPU / GPU: softmax 306 → 4.5 / 3.1 ulp, cross-entropy
  gradient 306 → 4.7 / 3.5 ulp, cross-entropy Hessian 74 → 2.7 / 2.8 ε).
- ``log_softmax`` carries a ``jax.custom_jvp`` choosing per row between the paper's form (argmax gradient as minus
  the sum of the others: exact when the argmax holds most of the mass) and the native form v − q·Σv (exact
  otherwise). Autodiff of Eq. 5 alone cancels on flat rows whose target is not the argmax: 7·10⁴ ε relative error on
  the argmax gradient at 32k classes.
- ``softmax`` takes its value as e^{x − x_max}/Σ (exp(log_softmax) adds |log q|·ε of error). On rows with q_max > ½ its
  derivative rule centres the tangent at the argmax (ẋ − ẋ_max: exact, since the softmax Jacobian maps constants to 0),
  applies q ⊙ (· − ⟨q, ·⟩), and sets the argmax entry to minus the sum of the others; other rows use q ⊙ (ẋ − ⟨q, ẋ⟩).
  The construction is its own transpose, so the VJP is centred too: it removes the cancellation between large, nearly
  equal cotangent entries (fp32 ``wide``: worst 410 ε with the previous rule, 151 ε for the paper's, 1.8 ε centred;
  numpy probe). Centring also keeps the argmax sum accurate where an uncentred one can lose up to 5.4·10⁴ ε to flushed
  subnormal products (XLA CPU, fp32 ``margin_bigk``: the paper's code evaluated eagerly, and an uncentred pivot on
  confident rows under jit; how much is lost depends on how XLA compiles the sum, and jitted the paper's code lost 18.8
  ε there and ``rule="paper"`` 2.9 ε). Limit of this construction under ``rule="accurate"`` (XLA CPU only; torch does
  not flush by default): where the products q_j·(v_j − s) themselves fall below the normal range (rows whose non-argmax
  probabilities sit near its bottom: logit gaps near 79 in fp32 and 700 in fp64 at 4 096 classes), they are flushed and
  the argmax entry, which sums them, loses their mass, in either mode and however the derivative is compiled. Measured
  with the comparison harness (VALIDATION.md; regime ``flush_bigk``, seeds 1–3, normwise against an allowance of the
  row's flushable operands times the cotangent, without a sum term): softmax JVP up to 263 ε (fp32) and 2.3·10¹¹ ε
  (fp64: 1.7·10⁻⁴ of an argmax entry of 5.7·10⁻³⁰⁴ lost; rows that lose the whole entry score 0, being below the
  allowance). The paper's code loses as much in the VJP but not in the JVP, whose forward rule sums nothing at the
  argmax (17 ε and 0 there): on these rows the accurate rule's forward mode is less accurate than the paper's. The
  default rule, ``"flush_proof"`` (below), forms these products at scale and loses nothing there.
- Hard-target cross-entropy takes its value from that ``log_softmax`` instead of Eq. 6, whose differences z_j − z_t
  are rounded at the size of the target's deficit (7 % gradient error in fp16 at a deficit of 262). Its derivative
  is centred at the target, as Eq. 6's is, but with the accurate probabilities: q_j off the target and −Σ_{j≠t} q_j
  on it, so componentwise accurate and zero-sum by construction.
- logsumexp over an all −inf set has zero, not NaN, gradient (one class left unmasked).
- Soft targets: δ = z_S − z_T is formed with its exact rounding remainder (TwoSum) and measured from the teacher's
  mode, so a common logit offset cancels exactly (Eq. 9 on raw δ: 10⁵ ε gradient error at offset 64); classes
  more than 1 away from the mode use their own probabilities (no e^d amplification of δ's rounding), with the
  normaliser corrected by log(1 − Q_far) − log(1 − P_far). Teacher gradients come from logit identities.
- The soft-target losses carry a ``jax.custom_jvp`` whose rule is itself differentiable, so Hessians are correct
  (the paper's ``stop_gradient`` trick makes them 0).

float16 / bfloat16 inputs are computed in float32 and returned in the input dtype (as softmaxfix.torch does), so 16-bit
results are the float32 ones rounded once; jax.nn computes them in the input dtype. Soft-target inputs are pinned to
their dtype first (``lax.reduce_precision``): XLA may otherwise hand a 16-bit producer's unrounded float32 value over
inside a fusion (excess precision), which breaks the exactness at z_S = z_T and the TwoSum compensation.

Numerical options: ``set_options(...)`` sets them process-wide, ``with options(...):`` for a block in the current
thread (as JAX's own configuration contexts), and ``get_options()`` reads those in force. They are read when a
function is traced, and they are part of the key of JAX's tracing and compilation caches (``jax.make_user_context``):
a jitted function, the caller's own included, is traced again under new options and never runs with stale ones.
Ahead-of-time artefacts are the exception: ``jax.jit(f).lower(...).compile()`` and ``jax.export`` keep the options in
force when they were lowered.

- ``exact_shift`` (default True): as above; False restores the rounded shift. Two values use the exact shift either way:
  the soft-target support-mass correction, and the integer-label cross-entropy where ``where`` masks out the target
  (whose derivatives are the logsumexp form's, with the rounded shift, as optax's). Under jit XLA fuses the extra
  elementwise work: forward + backward with the exact shift and ``rule="accurate"`` costs ×0.95–1.02 the previous rules'
  time on CPU and ×1.15–1.30 on GPU, nearly all of it the exact shift (fp32; GPU 2048 × 32768 on an RTX 3050 Ti, CPU 256
  × 32768; VALIDATION.md, "Cost"); against native on GPU, ×1.5–2.0 in fp32 and ×3.0–3.3 in bf16, which is computed in
  float32 (log_softmax, softmax and cross-entropy; the soft-target losses ×5.4–7.7 and ×6.3–8.9). The default rule,
  ``"flush_proof"``, costs more in softmax derivatives only (below).
- ``rule`` (default ``"flush_proof"``): the derivative rules of softmax and log_softmax, and so of everything built on
  them. ``"accurate"`` is the per-row construction above; ``"paper"`` is the paper's rule in every row, the one its code
  gets from autodiff: forward mode centred at the argmax, and, by transposition, reverse-mode results whose argmax entry
  is minus the sum of the others (zero-sum by construction; it cancels on flat rows, log_softmax VJP on ``flat`` 37–50 ε
  in fp32 and fp64 against 0.44–0.47 ε, and its uncentred softmax VJP is exposed to the flushing above). It is the
  paper's rule, not its code: it runs on this library's kernels (the exact shift unless ``exact_shift=False``, the
  target-pivot cross-entropy gradient, the fixes listed above); the paper's code is transcribed in
  ``tests/paper_jax.py`` and serves as a reference only. ``"flush_proof"``, the default (``softmaxfix.torch``'s is
  ``"accurate"``: torch does not flush by default), is ``"accurate"`` with the centred softmax construction also formed
  at scale, as a second branch selected per row, in the rows where its products could fall below the normal range
  (tangent × 2^−T, q × 2^{2T}, result × 2^−T; T = 63 in float32, 511 in float64); the other rows take the accurate
  rule's branch (XLA may still round some of them differently in the compiled program, neither way: normwise at the ε
  level, componentwise up to thousands of ulp on single entries far below the row's largest, with every key tied; on
  GPU, and on one float16 row of 20 seeds on CPU). On the flush-band rows above (XLA CPU, 20 seeds) the softmax VJP and
  JVP go from 2 088 / 426 ε (fp32) and 8.3·10¹¹ / 2.3·10¹¹ ε (fp64) to 0; those rows' stored results then no longer sum
  to zero (their off-argmax entries flush when written: Σg from 0 to 84.8 ε in fp32 and 5.65·10⁹ ε in fp64, without an
  allowance for that mass). It ties with ``"accurate"`` on every other measured key, and on GPU, where nothing flushes;
  the first derivatives of log_softmax and cross-entropy are unchanged (their second derivatives go through the softmax
  rule). On a scaled row a result above 2^65 in float32 (2^513 in float64) overflows where the accurate rule's is finite
  (a bound on the tangent, as in ``softmaxfix.torch``, would make the rule nonlinear in it). Cost (GTX 1080 Ti, and 8
  pinned CPU cores, idle machine; VALIDATION.md, "Cost"; measured on the harness's arms of both rules, whose
  single-branch arm timed as this library's single-branch form): softmax forward + backward ×1.32–1.84 the accurate
  rule's time on GPU and ×1.32–2.56 on CPU, the most with few classes on CPU; log_softmax and cross-entropy run as under
  ``"accurate"``. A single-branch form (factors of exactly 1 off the scaled rows; this library's until 30/10/2026) cost
  ×0.40–0.92 of that, but on GPU rounded other rows' VJPs slightly worse on 4 keys. Integer-target cross-entropy pivots
  at the target under every rule.

Measured over 20 seeds and 356 (regime, format, metric) keys per device (VALIDATION.md, "Twenty-seed comparisons"), for
the exact shift with ``rule="accurate"`` (the defaults until 29/10/2026). ``"flush_proof"`` ties with it outside the
flush band above, on every key (664 per device, CPU and GPU; 20 seeds, 29–30/10/2026). The accurate rule is more
accurate than this library's previous rules (rounded shift, softmax VJP pivoted with r·v_max) on 94 / 89 keys (CPU /
GPU), less accurate on 0 / 3 and mixed on 1 / 1: single-row worst cases, except a mean shift of +0.08 ε with the same
top rows (GPU, fp64 ``margin_bigk`` log_softmax Σg); against the paper's code it is better on 254 / 240 keys and worse
or mixed on 27 / 33, all of them the zero-sum residual Σg, which only the paper's rule makes exactly 0, but one (CPU,
fp64 ``safe`` softmax JVP: worst 8.36 against 4.05 ε, with the better mean).

Reverse-mode results are **not projected onto Σ = 0**, here or in ``softmaxfix.torch`` (whose rules and options are
the same, apart from a cotangent bound of its ``"flush_proof"`` and its default rule, ``"accurate"``). Subtracting the row mean of the computed gradient shifts every entry by about ε·Σ|g|/K and so destroys the
entries much smaller than that (the gradient of q_max: 2·10¹⁵⁸ ulp on fp64 ``wide`` in torch, 4.4 unprojected). Here
it would also project the forward-mode input, since JAX derives reverse mode by transposing the forward-mode rule
(``jax.custom_jvp``; JAX 0.12 has no public custom transpose): log_softmax JVP mean +0.05–0.09 ε on 11–12 keys, worst
0.455 → 0.907 ε on fp32 ``flat``. The same transposition makes the evaluation order of the softmax rule part of its
result, since the transpose accumulates the branches' cotangents in reverse order of use; ``_sm_tangent`` keeps the
order that was measured.
"""

import contextlib
import math
from collections.abc import Generator
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

type Axis = int | tuple[int, ...] | None

__all__ = [
    "Options",
    "get_options",
    "kl_divergence_with_logits",
    "log_softmax",
    "options",
    "set_options",
    "soft_target_cross_entropy",
    "softmax",
    "softmax_cross_entropy",
    "softmax_cross_entropy_with_integer_labels",
]


# ----------------------------------------------------------------------------------------------- options
class Options(NamedTuple):
    """Numerical options (see the module docstring); hashable, so jit can key its caches on them."""

    exact_shift: bool = True
    rule: str = "flush_proof"  # XLA CPU flushes subnormals (module docstring); softmaxfix.torch defaults to "accurate".


_RULES = ("accurate", "flush_proof", "paper")
# A jit-cache-sensitive context: JAX keys its tracing, lowering and compilation caches on its value, so whatever runs
# under new options is traced again; a compiled function never runs with options other than those in force.
_OPTIONS = jax.make_user_context(default_value=Options())


def _updated(current: Options, exact_shift: bool | None, rule: str | None) -> Options:
    if exact_shift is not None and not isinstance(exact_shift, bool):
        raise TypeError(f"exact_shift must be a bool, got {type(exact_shift).__name__}")
    if rule is not None and rule not in _RULES:
        raise ValueError(f"rule must be one of {_RULES}, got {rule!r}")
    return current._replace(**{k: v for k, v in (("exact_shift", exact_shift), ("rule", rule)) if v is not None})


def get_options() -> Options:
    """The numerical options in force: those of the innermost ``options`` block of this thread, else the
    process-wide ones."""
    return _OPTIONS.value


def set_options(*, exact_shift: bool | None = None, rule: str | None = None) -> Options:
    """Set the process-wide numerical options (``None`` leaves one unchanged) and return the previous process-wide
    ones. An enclosing ``options`` block still takes precedence in its thread until it exits."""
    previous = _OPTIONS.get_global()
    _OPTIONS.set_global(_updated(previous, exact_shift, rule))
    return previous


@contextlib.contextmanager
def options(*, exact_shift: bool | None = None, rule: str | None = None) -> Generator[Options]:
    """Set numerical options for the ``with`` block, in this thread (as JAX's own configuration contexts); the
    previous ones are restored on exit, exceptions included."""
    new = _updated(get_options(), exact_shift, rule)
    with _OPTIONS(new):
        yield new


# --------------------------------------------------------------------------------------------- axis handling
def _canonical_axes(axis: Axis, ndim: int) -> tuple[int, ...]:
    """Normalise ``axis`` to a tuple of non-negative axes, keeping the caller's order."""
    axes = tuple(range(ndim)) if axis is None else (axis,) if isinstance(axis, int) else tuple(axis)
    for a in axes:
        if not -ndim <= a < ndim:
            raise ValueError(f"axis {a} is out of bounds for an array of dimension {ndim}")
    out = tuple(a % ndim for a in axes)
    if len(set(out)) != len(out):
        raise ValueError(f"repeated axis in {axis}")
    return out


def _to_last(x: jax.Array, axes: tuple[int, ...]) -> tuple[jax.Array, tuple[int, ...], tuple[int, ...]]:
    """Move ``axes`` to the end and flatten them into one; also return the permutation and reduced shape."""
    batch = tuple(i for i in range(x.ndim) if i not in axes)
    perm = batch + axes
    y = jnp.transpose(x, perm)
    nb = len(batch)
    return y.reshape(y.shape[:nb] + (math.prod(y.shape[nb:]),)), perm, y.shape[nb:]


def _from_last(y: jax.Array, perm: tuple[int, ...], reduced: tuple[int, ...]) -> jax.Array:
    """Inverse of :func:`_to_last`."""
    y = y.reshape(y.shape[:-1] + reduced)
    return jnp.transpose(y, [perm.index(i) for i in range(len(perm))])


def _inexact(x: ArrayLike) -> jax.Array:
    """Promote like ``jax.nn`` does (integers go to the dtype ``jnp.exp`` would return)."""
    x = jnp.asarray(x)
    if not jnp.issubdtype(x.dtype, jnp.inexact):
        x = x.astype(jax.eval_shape(jnp.exp, jax.ShapeDtypeStruct((), x.dtype)).dtype)
    return x


# ------------------------------------------------------------------------------------------ last-axis kernels
def _argmax_parts(x: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Argmax (first on ties, keepdims), the max, its mask, and x − x_max with an exact 0 at the argmax."""
    i_max = jnp.argmax(x, axis=-1, keepdims=True)
    x_max = jnp.take_along_axis(x, i_max, axis=-1)
    is_max = jnp.arange(x.shape[-1]) == i_max
    return i_max, x_max, is_max, jnp.where(is_max, 0, x - x_max)


def _two_diff(a: jax.Array, b: jax.Array) -> tuple[jax.Array, jax.Array]:
    """hi = fl(a − b) and lo with hi + lo = a − b exactly (TwoSum: six operations, for any magnitudes); lo is 0 where hi
    is not finite."""
    s = a - b
    bb = s - a
    return s, jnp.where(jnp.isfinite(s), (a - (s - bb)) - (b + bb), 0)


def _exact_shift(x: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """x_max, the argmax mask, hi and lo with hi + lo = x − x_max exactly (both 0 at the argmax, lo 0 where hi is not
    finite), and e^{x − x_max} as e^{hi} + e^{hi}·lo."""
    _, x_max, is_max, _ = _argmax_parts(x)
    hi, lo = _two_diff(x, x_max)
    hi = jnp.where(is_max, 0, hi)
    lo = jnp.where(is_max | ~jnp.isfinite(hi), 0, lo)
    eh = jnp.exp(hi)
    return x_max, is_max, hi, lo, eh + eh * lo


def _ls_tangent(x: jax.Array, q: jax.Array, dx: jax.Array, rule: str) -> jax.Array:
    """Tangent ẋ − ⟨q, ẋ⟩ of log_softmax, in whichever of two exact rewritings is accurate for the row.

    Centred, [j≠max](ẋ_j − ẋ_max) + (r·ẋ_max − Σ_{k≠max} q_k ẋ_k) with r = Σ_{k≠max} q_k = 1 − q_max: its transpose
    gives the argmax entry as r·Σv − Σ_{j≠max} v_j and v_j − q_j·Σv elsewhere, keeping the e^{−m}-sized gradient that
    v − q·Σv rounds away once q_max rounds to 1. (The paper's reverse mode computes the same argmax value as minus the
    sum of the other entries, −Σ_{j≠max} g_j: zero-sum by construction, but cancelling on flat rows.)
    Plain, ẋ − ⟨q, ẋ⟩: its transpose v − q·Σv is accurate when q_max is not close to 1, where the paper's argmax sum
    cancels instead (flat rows). Rows with q_max > 1/2 (logit margin m > 0) use the centred form. Both are linear and
    differentiable, so reverse mode, forward mode and higher orders all follow.
    ``rule == "paper"``: the paper's tangent, (ẋ − ẋ_max) − Σ_{k≠max} q_k (ẋ_k − ẋ_max) in every row, whose
    transpose is the argmax entry −Σ_{j≠max} g_j.
    """
    i_max, _, is_max, _ = _argmax_parts(x)
    q_rest = jnp.where(is_max, 0, q)
    dx_max = jnp.take_along_axis(dx, i_max, axis=-1)
    if rule == "paper":
        w = jnp.where(is_max, 0, dx - dx_max)
        return w - jnp.sum(q_rest * w, axis=-1, keepdims=True)
    confident = jnp.take_along_axis(q, i_max, axis=-1) > 0.5
    plain = dx - jnp.sum(q * dx, axis=-1, keepdims=True)
    r = jnp.sum(q_rest, axis=-1, keepdims=True)
    # The argmax slot is masked rather than computed as ẋ_max − ẋ_max: that identically-zero term would carry a
    # cotangent that is added and subtracted again under transposition, −1 + (1 − r) → 0, re-creating the collapse.
    centred = jnp.where(is_max, 0, dx - dx_max) + (r * dx_max - jnp.sum(q_rest * dx, axis=-1, keepdims=True))
    return jnp.where(confident, centred, plain)


def _scale_factors(q: jax.Array, is_max: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """``rule == "flush_proof"``: per row, whether it is scaled (the largest non-argmax probability below 2^−(T+1), T
    half the normal exponent range of q's format: 63 in float32, 511 in float64), and the factors 2^−T and 2^T there
    (exactly 1 and 1 elsewhere). Exact powers of two, so scaling by them is exact (no ``ldexp``), short of overflow: a
    scaled result above 2^(emax − T) (2^65 in float32, 2^513 in float64) is infinite."""
    t = -jnp.finfo(q.dtype).minexp // 2
    scaled = jnp.max(jnp.where(is_max, 0, q), axis=-1, keepdims=True) < 2.0 ** (-t - 1)
    return scaled, jnp.where(scaled, 2.0**-t, 1.0).astype(q.dtype), jnp.where(scaled, 2.0**t, 1.0).astype(q.dtype)


def _sm_tangent(x: jax.Array, q: jax.Array, dx: jax.Array, rule: str) -> jax.Array:
    """Tangent q ⊙ (ẋ − ⟨q, ẋ⟩) of softmax. In rows with q_max > ½: the tangent centred at the argmax (exactly 0
    there), the plain Jacobian, then the argmax entry set to minus the sum of the others; this graph is its own
    transpose, so the VJP is the same construction on the cotangent (see the module docstring). Other rows: plain.
    ``rule == "flush_proof"``: the same, with the centred construction formed at scale in the rows where its products
    could fall below the normal range: tangent × 2^−T, q × 2^{2T}, result × 2^−T (``_scale_factors``), a second branch
    selected per row, so that the other rows compute exactly the accurate rule's construction. Symmetric, so the
    transpose forms its products at scale too, and XLA's flush-to-zero cannot drop the products the argmax entry sums,
    in either mode.
    ``rule == "paper"``: q ⊙ the paper's log_softmax tangent, in every row."""
    if rule == "paper":
        return q * _ls_tangent(x, q, dx, rule)
    if rule == "flush_proof":
        # Two branches, the accurate rule's and the scaled one, selected per row (30/10/2026; it replaced a single
        # branch with factors of exactly 1 off the scaled rows: as accurate on the CPU, but on the GPU XLA rounded the
        # other rows' VJPs differently and worse on 4 keys of 20 seeds; ×0.40–0.92 of this form's cost). Evaluated in
        # the order of the measured harness arm (row tests, unscaled branch, scaled branch, plain).
        i_max, _, is_max, _ = _argmax_parts(x)
        confident = jnp.take_along_axis(q, i_max, axis=-1) > 0.5
        scaled, dn, up = _scale_factors(q, is_max)
        xc = jnp.where(is_max, 0, dx - jnp.take_along_axis(dx, i_max, axis=-1))
        y = q * (xc - jnp.sum(q * xc, axis=-1, keepdims=True))
        unscaled = jnp.where(is_max, -jnp.sum(jnp.where(is_max, 0, y), axis=-1, keepdims=True), y)
        dxs = dx * dn
        xa = jnp.where(is_max, 0, dxs - jnp.take_along_axis(dxs, i_max, axis=-1))
        ys = (q * up * up) * (xa - jnp.sum(q * xa, axis=-1, keepdims=True))
        at_scale = dn * jnp.where(is_max, -jnp.sum(jnp.where(is_max, 0, ys), axis=-1, keepdims=True), ys)
        centred = jnp.where(scaled, at_scale, unscaled)
        plain = q * (dx - jnp.sum(q * dx, axis=-1, keepdims=True))
        return jnp.where(confident, centred, plain)
    # Evaluated in this order (row test, centred branch, plain branch): the transpose accumulates the branches'
    # cotangents in reverse order of use, and this order is the one that was measured.
    i_max, _, is_max, _ = _argmax_parts(x)
    confident = jnp.take_along_axis(q, i_max, axis=-1) > 0.5
    xc = jnp.where(is_max, 0, dx - jnp.take_along_axis(dx, i_max, axis=-1))
    y = q * (xc - jnp.sum(q * xc, axis=-1, keepdims=True))
    centred = jnp.where(is_max, -jnp.sum(jnp.where(is_max, 0, y), axis=-1, keepdims=True), y)
    plain = q * (dx - jnp.sum(q * dx, axis=-1, keepdims=True))
    return jnp.where(confident, centred, plain)


@partial(jax.custom_jvp, nondiff_argnums=(1,))
def _softmax_last(x: jax.Array, opts: Options) -> jax.Array:
    """softmax over the last axis as e^{x − x_max} / Σ e^{x − x_max} (the shift exact with ``opts.exact_shift``):
    accurate as a value (q_max rounding to 1 loses nothing there); the derivative needs the fix, which the custom rule
    supplies. Computing the value as exp(log_softmax) instead would add |log q|·eps of relative error."""
    if x.shape[-1] == 0:
        return x
    if opts.exact_shift:
        x_max, _, _, _, e = _exact_shift(x)
    else:
        _, x_max, _, shifted = _argmax_parts(x)
        e = jnp.exp(shifted)
    return jnp.where(x_max == jnp.inf, jnp.nan, e / jnp.sum(e, axis=-1, keepdims=True))


@_softmax_last.defjvp
def _softmax_last_jvp(opts, primals, tangents):
    (x,), (dx,) = primals, tangents
    q = _softmax_last(x, opts)
    if x.shape[-1] == 0:
        return q, dx
    return q, _sm_tangent(x, q, dx, opts.rule)


@partial(jax.custom_jvp, nondiff_argnums=(1,))
def _log_softmax_last(x: jax.Array, opts: Options) -> jax.Array:
    """log_softmax over the last axis without adding 1.0 to the small exponentials (Eq. 5), as
    (x − x_max) − log1p(Σ_{j≠max} e^{x_j − x_max}), with lo added last under the exact shift: the paper's
    softplus(logsumexp(others)) is the same quantity, reached through a log and an exp more (relative error
    ε·|log s| on confident rows, and on flat ones about 1.7× native's worst)."""
    if x.shape[-1] == 0:
        return x
    if opts.exact_shift:
        x_max, is_max, hi, lo, e = _exact_shift(x)
        s = jnp.sum(jnp.where(is_max, 0, e), axis=-1, keepdims=True)  # Argmax excluded from the sum.
        out = (hi - jnp.log1p(s)) + lo
    else:
        _, x_max, is_max, shifted = _argmax_parts(x)
        s = jnp.sum(jnp.where(is_max, 0, jnp.exp(shifted)), axis=-1, keepdims=True)
        out = shifted - jnp.log1p(s)
    return jnp.where(x_max == jnp.inf, jnp.nan, out)  # +inf input gives NaN, as jax.nn documents.


@_log_softmax_last.defjvp
def _log_softmax_last_jvp(opts, primals, tangents):
    (x,), (dx,) = primals, tangents
    out = _log_softmax_last(x, opts)
    if x.shape[-1] == 0:
        return out, dx
    return out, _ls_tangent(x, _softmax_last(x, opts), dx, opts.rule)


@partial(jax.custom_jvp, nondiff_argnums=(2,))
def _nll_last(x: jax.Array, t: jax.Array, opts: Options) -> jax.Array:
    """−log_softmax(x)_t over the last axis, for in-range target indices ``t`` (keepdims)."""
    return -jnp.take_along_axis(_log_softmax_last(x, opts), t, axis=-1)[..., 0]


@_nll_last.defjvp
def _nll_last_jvp(opts, primals, tangents):
    """Tangent ⟨g, ẋ⟩ with g = q − e_t: q_j off the target and −Σ_{j≠t} q_j on it, a sum of positive terms, so the
    gradient is componentwise accurate (no cancellation, at any margin and on flat rows) and sums to zero by
    construction, as the paper's Eq. 6 makes it. Its derivative is that of the log_softmax rule's own gradient (the
    rule's VJP at −e_t): g = sg(g_target) + (g_rule − sg(g_rule)) has g_target's value exactly (x − x = 0) and
    g_rule's derivative, whose Hessian rows round better than differentiating −Σ_{j≠t} q_j would (measured)."""
    x, t = primals
    dx, _ = tangents
    q = _softmax_last(x, opts)
    is_t = jnp.arange(x.shape[-1]) == t
    sg = jax.lax.stop_gradient
    g_target = jnp.where(is_t, -jnp.sum(jnp.where(is_t, 0, q), axis=-1, keepdims=True), q)
    _, back = jax.vjp(lambda a: _log_softmax_last(a, opts), x)
    g_rule = back(-is_t.astype(x.dtype))[0]
    g = sg(g_target) + (g_rule - sg(g_rule))
    return _nll_last(x, t, opts), jnp.sum(g * dx, axis=-1)


def _ce_int_last(x: jax.Array, t: jax.Array, valid: jax.Array | None, opts: Options) -> jax.Array:
    """Hard-target cross-entropy −log_softmax(x)_t over the last axis, with optax's label and mask semantics."""
    k = x.shape[-1]
    tn = jnp.where(t < 0, t + k, t)[..., None]  # optax wraps negative labels through take_along_axis.
    x_t = jnp.take_along_axis(x, t[..., None], axis=-1)[..., 0]  # NaN for out-of-range labels, as in optax.
    xm = x if valid is None else jnp.where(valid, x, -jnp.inf)
    loss = _nll_last(xm, jnp.clip(tn, 0, k - 1), opts)
    loss = jnp.where(jnp.isnan(x_t), jnp.nan, loss)
    if valid is None:
        return loss
    # A masked-out target is absent from optax's normaliser: the loss is then LSE_valid(x) − x_t. Its value as (m − x_t)
    # + log1p(s), both exact where they can be (worst 171 → 9 ulp in fp64, 57 → 10.6 in fp32 on wide rows, against
    # optax's form; VALIDATION.md, "Probes cited in the code"). Its derivatives, of every order, are the logsumexp
    # form's (optax's): autodiff of log1p(s) gives the argmax entry as 1 − s/(1 + s), which cancels on flat rows (the
    # same section). Where both values are finite, new + (old − old): the new value exactly, with the old one's
    # derivatives; elsewhere, the old one.
    sg = jax.lax.stop_gradient
    t_valid = jnp.take_along_axis(valid, jnp.clip(tn, 0, k - 1), axis=-1)[..., 0]
    old = _safe_lse(xm)[..., 0] - x_t
    m, l1p, _ = _lse_parts(x, valid)
    hi, lo = _two_diff(m[..., 0], x_t)
    new = (hi + l1p[..., 0]) + lo
    ok = jnp.isfinite(old) & jnp.isfinite(new)
    loss = jnp.where(t_valid, loss, jnp.where(ok, sg(new) + (old - sg(old)), old))
    return jnp.where(jnp.any(valid, axis=-1), loss, 0)


def _safe_lse(a: jax.Array) -> jax.Array:
    """logsumexp over the last axis (keepdims); an all −inf row gives −inf with zero gradient instead of NaN."""
    empty = jnp.all(a == -jnp.inf, axis=-1, keepdims=True)
    lse = jax.nn.logsumexp(jnp.where(empty, 0, a), axis=-1, keepdims=True)
    return jnp.where(empty, -jnp.inf, lse)


def _expm1mx(x: jax.Array) -> jax.Array:
    """e^x − 1 − x without cancellation: Taylor series to x¹⁶/16! for |x| < 0.5, direct difference elsewhere."""
    small = jnp.abs(x) < 0.5
    xs = jnp.where(small, x, 0)
    acc = jnp.ones_like(xs)
    for n in range(16, 2, -1):  # Horner form of x²/2 · (1 + x/3 · (1 + x/4 · (…))).
        acc = 1 + xs / n * acc
    xl = jnp.where(small, 1, x)
    return jnp.where(small, xs * xs / 2 * acc, jnp.expm1(xl) - xl)


def _log1pmx(y: jax.Array) -> jax.Array:
    """log(1 + y) − y without cancellation: Taylor series to y¹⁸/18 for |y| < 0.1, direct difference elsewhere."""
    small = jnp.abs(y) < 0.1
    ys = jnp.where(small, y, 0)
    acc = jnp.zeros_like(ys)
    for n in range(18, 1, -1):  # y² · (−1/2 + y·(1/3 + y·(−1/4 + …))).
        acc = (-1) ** (n + 1) / n + ys * acc
    yl = jnp.where(small, 1, y)
    return jnp.where(small, ys * ys * acc, jnp.log1p(yl) - yl)


def _log1m(x: jax.Array, log_comp: jax.Array) -> jax.Array:
    """log(1 − x): log1p(−x) below ½, else the caller's log(1 − x), evaluated from the complement's own terms."""
    low = x < 0.5
    return jnp.where(low, jnp.log1p(-jnp.where(low, x, 0)), log_comp)


def _lse_parts(z: jax.Array, mask: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """log Σ_{j∈mask} e^{z_j} over the last axis as m + log1p(s) (keepdims): m the masked maximum, s the sum of the
    other masked e^{z_j − m} with the shift exact (Eq. 5's form: nothing small is added to 1), and whether the mask is
    empty there (m and log1p(s) are then 0, for the caller to replace; no NaN reaches a gradient). A mask whose logits
    are all −inf gives m = −inf and log1p(s) = 0, computed on zeros as the empty one; a +inf maximum gives
    log1p(s) = 0, the sum being +inf whatever s is."""
    empty = ~jnp.any(mask, axis=-1, keepdims=True)
    dead = ~jnp.any(mask & (z != -jnp.inf), axis=-1, keepdims=True)  # The empty mask included.
    za = jnp.where(dead, 0, jnp.where(mask, z, -jnp.inf))
    m, is_max, _, _, e = _exact_shift(za)
    l1p = jnp.log1p(jnp.sum(jnp.where(is_max | ~mask, 0, e), axis=-1, keepdims=True))
    return jnp.where(dead & ~empty, -jnp.inf, m), jnp.where(dead | (m == jnp.inf), 0, l1p), empty


def _log_mass(z: jax.Array, mask: jax.Array) -> jax.Array:
    """log Σ_{j∈mask} softmax(z)_j from the logits: exact where the probabilities themselves underflow. Both sums in Eq.
    5's form with the exact shift, the difference of their maxima exact (TwoSum): worst 97 → 2.0–3.4 ulp against the
    difference of two logsumexps on wide rows (VALIDATION.md, "Probes cited in the code"). An empty mask, or one on −inf
    logits only, gives −inf, a row without a finite logit NaN, as that difference did."""
    m_a, l_a, empty_a = _lse_parts(z, mask)
    m_z, l_z, empty_z = _lse_parts(z, z != -jnp.inf)
    hi, lo = _two_diff(m_a, m_z)
    out = (hi + (l_a - l_z)) + lo
    return jnp.where(empty_z, jnp.nan, jnp.where(empty_a, -jnp.inf, out))


def _pin(x: jax.Array) -> jax.Array:
    """``x`` rounded to its own dtype: a no-op on the value it denotes, but XLA may hand a 16-bit producer's result
    over in float32 inside a fusion (excess precision), and exactness at z_S = z_T needs the stored values."""
    fi = jnp.finfo(x.dtype)
    return jax.lax.reduce_precision(x, exponent_bits=fi.nexp, mantissa_bits=fi.nmant)


def _compensated_b(z_s: jax.Array, z_t: jax.Array, both: jax.Array, k: jax.Array) -> jax.Array:
    """b = δ − δ_k* with δ = z_S − z_T on classes both distributions support, exact 0 at the teacher's mode k*.

    δ is formed with its exact rounding remainder (TwoSum), so a common offset between student and teacher logits
    cancels exactly instead of after rounding at the offset's magnitude. TwoSum needs δ to be one rounded value:
    XLA may keep 16-bit intermediates in float32 inside a fusion (excess precision) and round them only where they
    are stored, so δ's rounding to the dtype is made explicit (measured on GPU in bfloat16 without it: 2 770 ε
    student-gradient error in the ``offset`` regime).
    """
    zs0, zt0 = jnp.where(both, _pin(z_s), 0), jnp.where(both, _pin(z_t), 0)
    delta = _pin(zs0 - zt0)
    bv = delta - zs0
    err = (zs0 - (delta - bv)) - (zt0 + bv)  # delta + err == zs0 − zt0 exactly.
    take = partial(jnp.take_along_axis, indices=k, axis=-1)
    is_k = jnp.arange(z_t.shape[-1]) == k
    return jnp.where(is_k, 0, (delta - take(delta)) + (err - take(err)))


class _Soft(NamedTuple):
    """Quantities shared by the soft-target values and gradients over the last axis."""

    q: jax.Array
    p: jax.Array
    b: jax.Array  # Compensated δ − δ_k*.
    k: jax.Array  # Teacher's mode (keepdims).
    supported: jax.Array  # Every teacher class has a finite student logit (finite loss).
    close: jax.Array  # |b| ≤ 1: evaluated in log space.
    far: jax.Array  # Student mass elsewhere: evaluated from the probabilities themselves.
    pt: jax.Array  # Teacher weights renormalised on the close set.
    d: jax.Array  # b centred under pt (0 off the close set).
    p_close: jax.Array
    p_far: jax.Array
    lq: jax.Array  # log(1 − Q_far).
    lp: jax.Array  # log(1 − P_far).


def _soft(z_s: jax.Array, z_t: jax.Array, opts: Options) -> _Soft:
    """Split the classes into those within |b| ≤ 1 of the teacher's mode, where q − p and KL are evaluated in log
    space (Eq. 9), and the rest, whose own probabilities carry no cancellation. On the close set,
    log(q/p) = d − log1p(Σ p̃·expm1(d)) + log(1 − Q_far) − log(1 − P_far) exactly. Support is read from the logits,
    not from the probabilities: a class whose probability underflows still carries mass (its KL term is finite), and
    log(1 − Q_far) = log Q_close is taken in log space when the close set holds the smaller part of q."""
    z_s, z_t = _pin(z_s), _pin(z_t)
    q, p = _softmax_last(z_s, opts), _softmax_last(z_t, opts)
    pos_p, pos_q = z_t > -jnp.inf, z_s > -jnp.inf
    both = pos_p & pos_q
    supported = ~jnp.any(pos_p & ~pos_q, axis=-1, keepdims=True)
    k = jnp.argmax(z_t, axis=-1, keepdims=True)
    b = _compensated_b(z_s, z_t, both, k)
    close = both & (jnp.abs(b) <= 1)
    far = pos_q & ~close
    p_close = jnp.sum(jnp.where(close, p, 0), axis=-1, keepdims=True)
    p_far = jnp.sum(jnp.where(far, p, 0), axis=-1, keepdims=True)
    q_far = jnp.sum(jnp.where(far, q, 0), axis=-1, keepdims=True)
    pt = jnp.where(close, p, 0) / jnp.where(supported, p_close, 1)
    d = jnp.where(close, b - jnp.sum(pt * b, axis=-1, keepdims=True), 0)
    lq = _log1m(q_far, _log_mass(z_s, close))
    lp = _log1m(p_far, jnp.log(jnp.where(supported, p_close, 1)))  # P_close ≥ p_k* ≥ 1/K: never underflows.
    return _Soft(q, p, b, k, supported, close, far, pt, d, p_close, p_far, lq, lp)


def _student_grad_value(z_s: jax.Array, z_t: jax.Array, opts: Options) -> jax.Array:
    """q − p: Eq. 9 on the close set, with the HTML revision's |s| < 1 fallback and the zero-sum correction."""
    c = _soft(z_s, z_t, opts)
    s = c.d - jnp.log1p(jnp.sum(c.pt * jnp.expm1(c.d), axis=-1, keepdims=True)) + c.lq - c.lp  # log(q/p).
    small = c.supported & c.close & (jnp.abs(s) < 1)
    u = jnp.where(small, c.p * jnp.expm1(jnp.where(small, s, 0)), c.q - c.p)
    return u - c.q * jnp.sum(u, axis=-1, keepdims=True)


def _kl_value(z_s: jax.Array, z_t: jax.Array, opts: Options) -> jax.Array:
    """KL(p ‖ q) = P_close·(log Σ p̃·e^d − Σ p̃·d − log(1 − Q_far) + log(1 − P_far)) + Σ_far p·log(p/q), the far terms
    as log(p_k/q_k) = log(p_k*/q_k*) − b_k; +inf on rows whose loss is infinite. Σ p̃·expm1(d) is split into the
    cancellation-free Σ p̃·(expm1(d) − d) and the rounding residual r = Σ p̃·d; below 1, log1p(S′ + r) − r is
    evaluated as S′ + log1pmx(S′ + r), so r cancels analytically rather than in rounding."""
    c = _soft(z_s, z_t, opts)
    resid = jnp.sum(c.pt * c.d, axis=-1, keepdims=True)
    s_prime = jnp.sum(c.pt * _expm1mx(c.d), axis=-1, keepdims=True)
    y = s_prime + resid
    near = y < 1
    kl_core = jnp.where(near, s_prime + _log1pmx(jnp.where(near, y, 0)), jnp.log1p(y) - resid)
    r_star = jnp.log1p(y) - jnp.take_along_axis(c.d, c.k, axis=-1) - c.lq + c.lp  # log(p/q) at the mode.
    # Σ_far p_k·log(p_k/q_k) term by term: r*·P_far − Σ_far p_k·b_k would cancel two O(|b|) sums into the KL.
    far_terms = jnp.sum(jnp.where(c.far, c.p * (r_star - c.b), 0), axis=-1, keepdims=True)
    kl = c.p_close * (kl_core - c.lq + c.lp) + far_terms
    return jnp.where(c.supported, kl, jnp.inf)[..., 0]


def _ce_value(z_s: jax.Array, z_t: jax.Array, opts: Options) -> jax.Array:
    """−Σ p log q, with 0·log 0 := 0."""
    p = _softmax_last(z_t, opts)
    return -jnp.sum(p * jnp.where(p > 0, _log_softmax_last(z_s, opts), 0), axis=-1)


def _teacher_grad(z_s: jax.Array, z_t: jax.Array, kl: bool, opts: Options) -> jax.Array:
    """d/dz_T from logit identities (log q, log p differ from z_S, z_T by row constants): −p ⊙ (z_S − ⟨p, z_S⟩) for
    CE, −p ⊙ (b − ⟨p, b⟩) for KL (NaN where KL = ∞)."""
    p = _softmax_last(z_t, opts)
    k = jnp.argmax(z_t, axis=-1, keepdims=True)
    if not kl:
        zs_k = jnp.take_along_axis(z_s, k, axis=-1)
        a = jnp.where(p > 0, z_s - jnp.where(jnp.isfinite(zs_k), zs_k, 0), 0)
        return -p * (a - jnp.sum(p * a, axis=-1, keepdims=True))
    pos_p, pos_q = z_t > -jnp.inf, z_s > -jnp.inf  # Support from the logits, as in _soft.
    both = pos_p & pos_q
    supported = ~jnp.any(pos_p & ~pos_q, axis=-1, keepdims=True)
    b = _compensated_b(z_s, z_t, both, k)
    return jnp.where(supported, -p * (b - jnp.sum(p * b, axis=-1, keepdims=True)), jnp.nan)


@partial(jax.custom_jvp, nondiff_argnums=(2,))
def _student_grad(z_s: jax.Array, z_t: jax.Array, opts: Options) -> jax.Array:
    """q − p from the log-space evaluation. Its own derivative is that of q − p through the softmax rule:
    differentiating the log-space formula instead cancels O(1) terms into e^{−m}-sized curvature."""
    return _student_grad_value(z_s, z_t, opts)


@_student_grad.defjvp
def _student_grad_jvp(opts, primals, tangents):
    z_s, z_t = primals
    dz_s, dz_t = tangents
    q, p = _softmax_last(z_s, opts), _softmax_last(z_t, opts)
    tangent = _sm_tangent(z_s, q, dz_s, opts.rule) - _sm_tangent(z_t, p, dz_t, opts.rule)
    return _student_grad(z_s, z_t, opts), tangent


@partial(jax.custom_jvp, nondiff_argnums=(2,))
def _soft_ce_last(z_s: jax.Array, z_t: jax.Array, opts: Options) -> jax.Array:
    return _ce_value(z_s, z_t, opts)


@_soft_ce_last.defjvp
def _soft_ce_last_jvp(opts, primals, tangents):
    z_s, z_t = primals
    dz_s, dz_t = tangents
    tangent = jnp.sum(_student_grad(z_s, z_t, opts) * dz_s, axis=-1) + jnp.sum(
        _teacher_grad(z_s, z_t, False, opts) * dz_t, axis=-1
    )
    return _soft_ce_last(z_s, z_t, opts), tangent


@partial(jax.custom_jvp, nondiff_argnums=(2,))
def _kl_last(z_s: jax.Array, z_t: jax.Array, opts: Options) -> jax.Array:
    return _kl_value(z_s, z_t, opts)


@_kl_last.defjvp
def _kl_last_jvp(opts, primals, tangents):
    z_s, z_t = primals
    dz_s, dz_t = tangents
    tangent = jnp.sum(_student_grad(z_s, z_t, opts) * dz_s, axis=-1) + jnp.sum(
        _teacher_grad(z_s, z_t, True, opts) * dz_t, axis=-1
    )
    return _kl_last(z_s, z_t, opts), tangent


# ------------------------------------------------------------------------------------------------ public API
# Each public function reads the options when it is traced and passes them down explicitly; JAX keys the jit caches on
# them (see ``_OPTIONS``).
@partial(jax.jit, static_argnames=("axis",))
def log_softmax(x: ArrayLike, axis: Axis = -1, where: ArrayLike | None = None) -> jax.Array:
    """Drop-in for :func:`jax.nn.log_softmax`, accurate at any logit margin.

    Args:
        x: input array.
        axis: axis or axes to normalise over (int, tuple, or None for all).
        where: elements to include; masked-out outputs are −inf.
    """
    opts = get_options()
    return _apply_last(lambda y: _log_softmax_last(y, opts), x, axis, where, -jnp.inf)


@partial(jax.jit, static_argnames=("axis",))
def softmax(x: ArrayLike, axis: Axis = -1, where: ArrayLike | None = None) -> jax.Array:
    """Drop-in for :func:`jax.nn.softmax`; masked-out outputs are 0.

    Its gradient keeps the argmax entry's q(1 − q) term instead of rounding it to 0 at high margin; with the exact
    shift its value is accurate to a few ulp where native's carries the shift's rounding.
    """
    opts = get_options()
    return _apply_last(lambda y: _softmax_last(y, opts), x, axis, where, 0)


def _compute_dtype(dtype) -> jnp.dtype:
    """float16 / bfloat16 are computed in float32, as in softmaxfix.torch; other dtypes in their own precision."""
    return jnp.dtype(jnp.float32) if jnp.dtype(dtype) in (jnp.float16, jnp.bfloat16) else jnp.dtype(dtype)


def _apply_last(kernel, x: ArrayLike, axis: Axis, where: ArrayLike | None, fill: float) -> jax.Array:
    """Run a last-axis kernel over ``axis`` with jax.nn's promotion and ``where`` conventions, in the compute dtype;
    the result is in the input dtype."""
    x = _inexact(x)
    dtype = x.dtype
    valid = None if where is None else jnp.broadcast_to(jnp.asarray(where, bool), x.shape)
    if valid is not None:
        x = jnp.where(valid, x, -jnp.inf)
    if x.size == 0:
        return x
    y, perm, reduced = _to_last(x.astype(_compute_dtype(dtype)), _canonical_axes(axis, x.ndim))
    out = _from_last(kernel(y), perm, reduced).astype(dtype)
    return out if valid is None else jnp.where(valid, out, fill)


def _expand_where(where: ArrayLike | None, logits: jax.Array, axis: Axis) -> jax.Array | None:
    """optax's convention: a per-example mask (``where.ndim != logits.ndim``) gains the class axis."""
    if where is None:
        return None
    where = jnp.asarray(where, bool)
    if where.ndim != logits.ndim and axis is not None:
        where = jnp.expand_dims(where, axis)
    return jnp.broadcast_to(where, logits.shape)


def _check_floating(x: jax.Array, name: str) -> None:
    if not jnp.issubdtype(x.dtype, jnp.floating):
        raise ValueError(f"{name} must be a floating-point array, got dtype {x.dtype}")


@partial(jax.jit, static_argnames=("axis",))
def softmax_cross_entropy(
    logits: ArrayLike, labels: ArrayLike, axis: Axis = -1, where: ArrayLike | None = None
) -> jax.Array:
    """Drop-in for :func:`optax.losses.softmax_cross_entropy` (labels are probabilities).

    Uses the fixed :func:`log_softmax`, so one-hot labels get the exact target gradient. For soft labels the
    gradient Σp·q − p still subtracts two rounded probabilities; pass teacher logits to
    :func:`soft_target_cross_entropy` to avoid that.
    """
    opts = get_options()
    logits, labels = jnp.asarray(logits), jnp.asarray(labels)
    _check_floating(logits, "logits")
    dtype, cdt = jnp.result_type(logits, labels), _compute_dtype(logits.dtype)
    where = _expand_where(where, logits, axis)
    log_probs = _apply_last(lambda y: _log_softmax_last(y, opts), logits.astype(cdt), axis, where, -jnp.inf)
    return -jnp.sum(labels.astype(jnp.result_type(labels, cdt)) * log_probs, axis, where=where).astype(dtype)


@partial(jax.jit, static_argnames=("axis",))
def softmax_cross_entropy_with_integer_labels(
    logits: ArrayLike, labels: ArrayLike, axis: int | tuple[int, ...] = -1, where: ArrayLike | None = None
) -> jax.Array:
    """Drop-in for :func:`optax.losses.softmax_cross_entropy_with_integer_labels`, accurate at any margin.

    Computes −log_softmax(z)_t with the fixed :func:`log_softmax`, so neither the loss nor the target gradient
    rounds to 0. Rows with no valid entry give 0; negative labels wrap and out-of-range labels give NaN, as in
    optax.
    """
    logits = jnp.asarray(logits)
    labels = jnp.asarray(labels)
    _check_floating(logits, "logits")
    if not jnp.issubdtype(labels.dtype, jnp.integer):
        raise ValueError(f"labels must be an integer array, got dtype {labels.dtype}")
    if not isinstance(axis, (int, tuple)):  # ValueError, not TypeError: optax raises ValueError here.
        raise ValueError(f"Keyword argument 'axis' must be of type 'int' or 'tuple[int, ...]' but got {type(axis)}.")  # noqa: TRY004
    axes = _canonical_axes(axis, logits.ndim)
    if where is not None:
        # optax's logsumexp broadcasts the mask and the logits against each other (a per-example mask first gains
        # the class axes), so the logits may take the mask's batch.
        w = jnp.asarray(where, bool)
        w = w if w.ndim == logits.ndim else jnp.expand_dims(w, axis)
        if w.ndim > logits.ndim:
            raise ValueError(f"where of shape {jnp.shape(where)} adds axes to logits of shape {logits.shape}")
        shape = list(jnp.broadcast_shapes(logits.shape, w.shape))
        for a in axes:  # Only the batch broadcasts: the classes stay the logits' (a wider mask raises below).
            shape[a] = logits.shape[a]
        logits = jnp.broadcast_to(logits, tuple(shape))
    where = _expand_where(where, logits, axis)
    x, _, _ = _to_last(logits.astype(_compute_dtype(logits.dtype)), axes)
    valid = None if where is None else _to_last(where, axes)[0]
    if labels.ndim != x.ndim - 1:  # optax's take_along_axis needs one label axis per batch axis of the logits.
        raise ValueError(
            f"labels must have {x.ndim - 1} axes, one per batch axis of the logits, got shape {labels.shape} for "
            f"logits of shape {logits.shape} and axis {axis}"
        )
    # optax broadcasts the labels against the logits' batch (through take_along_axis); here both are broadcast to the
    # common batch first, so the derivative rule sees matching shapes and autodiff sums the cotangents back.
    batch = jnp.broadcast_shapes(x.shape[:-1], labels.shape)
    x = jnp.broadcast_to(x, (*batch, x.shape[-1]))
    valid = None if valid is None else jnp.broadcast_to(valid, (*batch, valid.shape[-1]))
    labels = jnp.broadcast_to(labels, batch)
    return _ce_int_last(x, labels, valid, get_options()).astype(logits.dtype)


def _soft_public(kernel, logits: ArrayLike, teacher_logits: ArrayLike, axis: Axis, where: ArrayLike | None):
    """Shared front end of the soft-target losses: dtype, shape, axes and masking."""
    zs, zt = jnp.asarray(logits), jnp.asarray(teacher_logits)
    _check_floating(zs, "logits")
    _check_floating(zt, "teacher_logits")
    if zs.shape != zt.shape:
        raise ValueError(f"logits and teacher_logits must have the same shape, got {zs.shape} and {zt.shape}")
    dtype = jnp.result_type(zs, zt)
    # Pinned in the input dtype before the upcast: under excess precision XLA may fold a 16-bit producer's rounding
    # and the convert to float32 away, handing over the unrounded value.
    zs, zt = (_pin(a.astype(dtype)).astype(_compute_dtype(dtype)) for a in (zs, zt))
    axes = _canonical_axes(axis, zs.ndim)
    where = _expand_where(where, zs, axis)
    zs, _, _ = _to_last(zs, axes)
    zt, _, _ = _to_last(zt, axes)
    if zs.shape[-1] == 0:
        return jnp.zeros(zs.shape[:-1], dtype)
    if where is None:
        return kernel(zs, zt).astype(dtype)
    valid, _, _ = _to_last(where, axes)
    any_valid = jnp.any(valid, axis=-1, keepdims=True)
    # Masked entries become −inf in both (they then contribute nothing); empty rows are evaluated on a finite
    # dummy and reported as 0, as optax does for sums over nothing.
    zs = jnp.where(any_valid, jnp.where(valid, zs, -jnp.inf), 0)
    zt = jnp.where(any_valid, jnp.where(valid, zt, -jnp.inf), 0)
    return jnp.where(any_valid[..., 0], kernel(zs, zt), 0).astype(dtype)


@partial(jax.jit, static_argnames=("axis",))
def soft_target_cross_entropy(
    logits: ArrayLike, teacher_logits: ArrayLike, axis: Axis = -1, where: ArrayLike | None = None
) -> jax.Array:
    """Soft-target cross-entropy −Σ p log q per example, with p = softmax(teacher_logits), q = softmax(logits).

    The student gradient q − p is computed from ``logits − teacher_logits`` in log space, so it is exactly 0 when
    the two are equal and stays accurate as they approach; the teacher gradient is −p ⊙ (log q − ⟨p, log q⟩).
    Derivatives of every order are correct. Same value as
    ``optax.losses.softmax_cross_entropy(logits, jax.nn.softmax(teacher_logits))``, except on classes whose logits
    are −inf in both, where this takes 0·log 0 := 0 and returns the finite limit instead of NaN.
    """
    opts = get_options()
    return _soft_public(lambda a, b: _soft_ce_last(a, b, opts), logits, teacher_logits, axis, where)


@partial(jax.jit, static_argnames=("axis",))
def kl_divergence_with_logits(
    logits: ArrayLike, teacher_logits: ArrayLike, axis: Axis = -1, where: ArrayLike | None = None
) -> jax.Array:
    """KL(p ‖ q) per example, with p = softmax(teacher_logits), q = softmax(logits).

    Evaluated as log Σ p·exp(d) − Σ p·d on the centred logit difference d, so it is exactly 0 at equality and
    small divergences are not rounding noise. Student gradient as in :func:`soft_target_cross_entropy`. Same
    value as ``optax.losses.kl_divergence(jax.nn.log_softmax(logits), jax.nn.softmax(teacher_logits))``, except on
    classes whose logits are −inf in both, where this returns the finite limit instead of NaN.
    """
    opts = get_options()
    return _soft_public(lambda a, b: _kl_last(a, b, opts), logits, teacher_logits, axis, where)
